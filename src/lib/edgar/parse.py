import asyncio
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
import re
import time
from traceback import TracebackException
from typing import cast, Literal

import aiometer
import httpx
from lxml import etree
import pandas as pd
from pandera.typing import DataFrame, Series
from parsel import Selector
import polars as pl

from lib.const import HEADERS
from lib.edgar.models import CikEntry, CikFrame
from lib.fin.models import (
  FinStatement,
  FinRecord,
  Duration,
  Instant,
  Member,
  Scope,
  FinData,
)
from lib.fin.statement import (
  statement_urls,
  upsert_merged_statements,
)
from lib.utils.validate import (
  validate_currency,
)
from lib.utils.string import insert_characters, replace_all
from lib.utils.time import (
  month_end,
  fiscal_quarter_monthly,
)
from lib.xbrl.filings import (
  fetch_xbrl,
  parse_xbrl_period,
  parse_xbrl_unit,
  add_record_to_findata,
)

type Docs = Literal["cal", "def", "htm", "lab", "pre"]


async def scrap_edgar_statements(cik: int, id: str):
  from lib.edgar.company import Company

  company = Company(cik)
  filings = await company.xbrl_urls()

  financials = await parse_statements(filings.tolist())
  upsert_merged_statements("statements.db", id, financials)


async def update_edgar_statements(cik: int, id: str, delta=120):
  from lib.edgar.company import Company

  url_pattern = "https://www.sec.gov/Archives/edgar/data/%"
  old_filings = statement_urls("statements.db", id, url_pattern)

  if old_filings is None:
    await scrap_edgar_statements(cik, id)
    return

  last_date = old_filings["date"].max()

  if relativedelta(dt.now(), last_date).days < delta:
    return

  company = Company(cik)
  new_filings = await company.xbrl_urls()

  if not new_filings:
    return

  old_urls = set(old_filings["url"])
  new_urls = set(new_filings)
  new_urls = new_urls.difference(old_urls)

  if not new_urls:
    return

  mask = new_filings.isin(new_urls)
  new_filings = new_filings.loc[mask]
  new_statements = await parse_statements(new_filings.tolist())
  if new_statements:
    upsert_merged_statements("statements.db", id, new_statements)


async def parse_xbrl_urls(cik: int, doc_ids: list[str], doc_type: Docs) -> Series[str]:
  tasks = [
    asyncio.create_task(parse_xbrl_url(cik, doc_id, doc_type)) for doc_id in doc_ids
  ]
  urls = await asyncio.gather(*tasks)

  result = pd.Series(urls, index=doc_ids)
  result = result.loc[result.notnull()]

  return cast(Series[str], result)


async def parse_xbrl_url(cik: int, doc_id: str, doc_type: Docs = "htm") -> str:
  url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{doc_id.replace('-', '')}/{doc_id}-index.html"
  async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
    response = await client.get(url, headers=HEADERS)
    if response.status_code != 200:
      raise httpx.RequestError(f"Error: {response.text}")
    dom = Selector(response.text)

  data_files = dom.xpath("//table[@summary='Data Files']")
  if doc_type == "htm":
    href = data_files.xpath(
      './/a[re:test(@href, "(?<!_(cal|def|lab|pre))\\.xml$")]//@href'
    ).get()

  else:
    href = data_files.xpath(f'.//a[re:test(@href, "_{doc_type}.xml$")]//@href').get()

  return f"https://www.sec.gov{href}"


async def parse_statements(urls: list[str], run_async=True) -> list[FinStatement]:
  if run_async:
    tasks = [partial(parse_xml_filing, url) for url in urls]
    try:
      financials = await aiometer.run_all(tasks, max_at_once=5, max_per_second=2)
    except* Exception as eg:
      for i, exc in enumerate(eg.exceptions, 1):
        tb = "".join(TracebackException.from_exception(exc).format())
        print(f"\n---- Sub‑exception #{i} ----\n{tb}")
      raise

    return financials

  financials = [await parse_xml_filing(url) for url in urls]
  return financials


async def parse_xml_filing(url: str) -> FinStatement:
  def parse_unit(unit_id: str) -> str:
    try:
      unit = parse_xbrl_unit(root, unit_id)
    except ValueError:
      pattern = r"^Unit_(Standard|Divide)_(\w+)_[A-Za-z0-9_-]{22}$"
      unit_match = re.match(pattern, unit_id)
      if unit_match is not None:
        unit = unit_match.group(2).lower()
      else:
        unit = unit_id.split("_")[-1].lower()

    replacements = {"iso4217": "", "dollar": "d", "euro": "eur"}
    unit = replace_all(unit, replacements)

    return unit

  root = await fetch_xbrl(url)
  if root.tag == "Error":
    cik, doc_id = url.split("/")[6:8]
    doc_id = insert_characters(doc_id, {"-": [10, 12]})
    time.sleep(1)
    url = await parse_xbrl_url(int(cik), doc_id)
    root = await fetch_xbrl(url)

  form = {"10-K": "annual", "20-F": "annual", "40-F": "annual", "10-Q": "quarterly"}
  namespaces: dict[str | None, str] = root.nsmap
  namespaces = {k: v for k, v in namespaces.items() if k is not None}

  scope = cast(
    Scope,
    form[cast(str, root.xpath("./dei:DocumentType/text()", namespaces=namespaces)[0])],
  )
  date = dt.strptime(
    cast(
      str, root.xpath("./dei:DocumentPeriodEndDate/text()", namespaces=namespaces)[0]
    ),
    "%Y-%m-%d",
  )

  fiscal_end = cast(
    str, root.xpath("./dei:CurrentFiscalYearEndDate/text()", namespaces=namespaces)[0]
  )
  fiscal_end = re.sub("^-+", "", fiscal_end)
  fiscal_pattern = r"(0[1-9]|1[0-2])-(0[1-9]|12[0-9]|3[01])"

  match = re.search(fiscal_pattern, fiscal_end)
  fiscal_end_month = int(cast(re.Match[str], match).group(1))

  fiscal_period = cast(
    str, root.xpath("./dei:DocumentFiscalPeriodFocus/text()", namespaces=namespaces)[0]
  )

  if scope == "quarterly":
    derived_quarter = fiscal_quarter_monthly(date.month, fiscal_end_month)
    stated_quarter = int(fiscal_period[1])
    if derived_quarter != stated_quarter:
      new_fiscal_month = (fiscal_end_month - 3 * stated_quarter) % 12
      new_fiscal_day = month_end(date.year, new_fiscal_month)
      fiscal_end = f"{new_fiscal_month}-{new_fiscal_day}"

  doc_id = url.split("/")[-2]
  currency: set[str] = set()

  periods: set[Duration | Instant] = set()
  units: set[str] = set()
  dims: set[str] = set()

  data: FinData = {}

  name_pattern = (
    r"((?<!Level)(Zero(?!Coupon)|One|Two|Three|Four|Five|Six|Seven|Eight|Nine))+\w+$"
  )

  items: list[etree._Element] = root.findall(".//*[@unitRef]")
  for item in items:
    if item.text is None:
      continue

    context_id = item.attrib["contextRef"]

    period = parse_xbrl_period(root, context_id, True)
    periods.add(period)

    unit = parse_unit(item.attrib["unitRef"])
    units.add(unit)
    if len(unit) == 3 and validate_currency(unit):
      currency.add(unit)

    value = float(item.text)
    if "scale" in item.attrib:
      value *= 10.0 ** int(item.attrib["scale"])

    context: etree._Element = root.xpath(
      f".//*[local-name()='context' and @id='{context_id}']"
    )[0]
    member_nodes: list[etree._Element] = context.xpath(
      ".//*[local-name()='explicitMember']"
    )

    if member_nodes:
      member = member_nodes[0]
      member_name = cast(str, member.text).split(":")[-1]
      member_name = re.sub(r"(Segment)?Member", "", member_name)
      dim = cast(str, member.attrib["dimension"]).split(":")[-1]
      dims.add(dim)
      record = FinRecord(
        members={
          member_name: Member(
            dim=dim,
            value=value,
            unit=unit,
          )
        }
      )
    else:
      record = FinRecord(value=value, unit=unit)

    item_name = cast(str, item.tag).split("}")[-1]
    item_name = re.sub(name_pattern, "", item_name)
    add_record_to_findata(data, item_name, period, record)

  return FinStatement(
    date=date.date(),
    fiscal_period=fiscal_period,
    fiscal_end=fiscal_end,
    sources=[url],
    currencies=currency,
    periods=periods,
    units=units,
    dimensions=dims,
    synonyms=dict(),
    data=data,
  )


async def parse_taxonomy(url: str) -> pd.DataFrame:
  namespace = {
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
  }

  def rename_sheet(txt: str) -> str:
    pattern = r"income|balance|cashflow"
    m = re.search(pattern, txt, flags=re.I)
    if m:
      txt = m.group().lower()

    return txt

  async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=HEADERS)
    root: etree._Element = etree.fromstring(response.content)

  url_pattern = r"^https?://www\..+/"
  el_pattern = r"(?<=_)[A-Z][A-Za-z]+(?=_)?"

  taxonomy = []
  for sheet in root.findall(".//link:calculationLink", namespaces=namespace):
    sheet_label = re.sub(url_pattern, "", sheet.attrib[f"{{{namespace['xlink']}}}role"])
    sheet_label = rename_sheet(sheet_label)

    for el in sheet.findall(".//link:calculationArc", namespaces=namespace):
      taxonomy.append(
        {
          "sheet": sheet_label,
          "gaap": cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f"{{{namespace['xlink']}}}to"]),
          ).group(),
          "parent": cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f"{{{namespace['xlink']}}}from"]),
          ).group(),
        }
      )

  df = pd.DataFrame.from_records(taxonomy)
  df.set_index("item", inplace=True)
  df.drop_duplicates(inplace=True)
  return df


async def get_isin(cik: str | int) -> str | None:
  """cik-isin.com is not hosted anymore"""
  url = f"https://cik-isin.com/cik2isin.php?cik={cik}"

  try:
    async with httpx.AsyncClient() as client:
      response = await client.get(url, headers=HEADERS)
      response.raise_for_status()
      dom = Selector(response.text)
  except (httpx.ConnectError, httpx.RequestError, httpx.HTTPStatusError) as e:
    print(f"Error fetching ISIN for CIK {cik}: {e}")
    return None

  isin_text = dom.xpath(
    'normalize-space(//p[b[text()="ISIN"]]/text()[normalize-space()])'
  ).get()
  if isin_text is None:
    return ""

  isin_pattern = r"[A-Z]{2}[A-Z0-9]{9}[0-9]"
  isin_match = re.search(isin_pattern, isin_text)

  return isin_match.group() if isin_match else ""


def get_ciks() -> DataFrame[CikFrame]:
  url = "https://www.sec.gov/files/company_tickers.json"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    data: dict[str, dict] = response.json()

  lf = pl.LazyFrame(list(data.values()))
  lf = lf.rename({"cik_str": "cik", "title": "name"})
  lf_grouped = lf.group_by("cik", maintain_order=False).agg(
    [
      pl.col("name").first(),
      pl.col("ticker").map_elements(lambda x: json.dumps(x)).alias("tickers"),
    ]
  )

  return cast(DataFrame[CikFrame], lf_grouped.collect())


def get_ciks_pandas() -> DataFrame[CikFrame]:
  def get_tickers(group: pd.DataFrame) -> str:
    tickers = group["ticker"].tolist()
    return json.dumps(tickers)

  url = "https://www.sec.gov/files/company_tickers.json"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    parse: dict[str, CikEntry] = response.json()

  df = pd.DataFrame.from_dict(parse, orient="index")
  rename = {"cik_str": "cik", "title": "name"}
  df.rename(columns=rename, inplace=True)
  cik = df.groupby("cik").agg({"cik": "first", "name": "first"}, include_groups=False)
  cik["tickers"] = df.groupby("cik").apply(get_tickers, include_groups=False)

  return cast(DataFrame[CikFrame], cik)
