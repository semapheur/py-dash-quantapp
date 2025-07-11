import asyncio
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
import re
import time
from typing import cast, Literal
import xml.etree.ElementTree as et

import aiometer
import hishel
import httpx
import pandas as pd
from pandera.typing import DataFrame, Series
from parsel import Selector

from lib.const import HEADERS
from lib.edgar.models import CikEntry, CikFrame
from lib.fin.models import (
  FinStatement,
  Instant,
  Duration,
  FinPeriodStore,
  UnitStore,
  FinRecord,
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
  month_difference,
  month_end,
  fiscal_quarter_monthly,
  exclusive_end_date,
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
    tasks = [partial(parse_statement, url) for url in urls]
    financials = await aiometer.run_all(tasks, max_per_second=5)

    return financials

  financials = [await parse_statement(url) for url in urls]
  return financials


async def parse_statement(url: str) -> FinStatement:
  def fix_data(
    data: FinData,
    period_lookup: dict[Duration | Instant, str],
    unit_lookup: dict[str, int],
  ) -> FinData:
    fixed: FinData = {}

    for k in sorted(data.keys()):
      temp: list[FinRecord] = []

      for item in data[k]:
        item["period"] = period_lookup[item["period"]]

        if "unit" in item:
          item["unit"] = unit_lookup[item["unit"]]

        members = item.get("members")
        if members is None:
          temp.append(item)
          continue

        if len(members) == 1:
          member = next(iter(members.values()))
          m_unit = member.get("unit")

          if "value" not in item:
            item["value"] = member.get("value")

          if m_unit is None:
            temp.append(item)
            continue

          member["unit"] = unit_lookup[m_unit]

          if "unit" not in item:
            item["unit"] = member["unit"]

          temp.append(item)
          continue

        value = 0.0
        units = set()
        for m in members.keys():
          value += members[m].get("value", 0)
          unit = members[m].get("unit")
          if unit is not None:
            members[m]["unit"] = unit_lookup[unit]
            units.add(unit)

        if len(units) == 1:
          item["unit"] = unit_lookup[units.pop()]

          if "value" not in item:
            item["value"] = value

        temp.append(item)

      fixed[k] = temp

    return fixed

  def parse_period(period: et.Element) -> Instant | Duration:
    def parse_date(date_text: str):
      match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
      if match is None:
        raise ValueError(f'"{date_text}" does not match format "%Y-%m-%d"')

      return dt.strptime(match.group(), "%Y-%m-%d").date()

    if (el := period.find("./{*}instant")) is not None:
      instant_date = parse_date(cast(str, el.text)) + relativedelta(days=1)
      return Instant(instant=instant_date)

    start_date = parse_date(
      cast(str, cast(et.Element, period.find("./{*}startDate")).text)
    )
    end_date = parse_date(cast(str, cast(et.Element, period.find("./{*}endDate")).text))
    months = month_difference(start_date, end_date)
    end_date = exclusive_end_date(start_date, end_date, months)
    interval = Duration(start_date=start_date, end_date=end_date, months=months)

    return interval

  def parse_unit(unit: str) -> str:
    if re.match(r"^U(nit)?\d+$", unit) is not None:
      unit_el = cast(et.Element, root.find(f'.{{*}}unit[@id="{unit}"]'))

      if unit_el is None:
        print(url)

      if (measure_el := unit_el.find(".//{*}measure")) is not None:
        unit_ = cast(str, measure_el.text).split(":")[-1].lower()

      elif (divide := unit_el.find(".//{*}divide")) is not None:
        numerator = (
          cast(str, cast(et.Element, divide.find(".//{*}unitNumerator/measure")).text)
          .split(":")[-1]
          .lower()
        )
        denominator = (
          cast(str, cast(et.Element, divide.find(".//{*}unitDenominator/measure")).text)
          .split(":")[-1]
          .lower()
        )
        unit_ = f"{numerator}/{denominator}"

    else:
      pattern = r"^Unit_(Standard|Divide)_(\w+)_[A-Za-z0-9_-]{22}$"
      if (m := re.search(pattern, unit)) is not None:
        unit_ = m.group(2).lower()
      else:
        unit_ = unit.split("_")[-1].lower()

    unit_ = replace_all(unit_, {"iso4217": "", "dollar": "d", "euro": "eur"})

    pattern = r"^[a-z]{3}$"
    m = re.search(pattern, unit_, flags=re.I)
    if m is not None:
      if validate_currency(m.group()):
        currency.add(m.group())

    return unit_

  def parse_member(item: et.Element, segment: et.Element) -> dict[str, Member]:
    def parse_name(name: str) -> str:
      name = re.sub(r"(Segment)?Member", "", name)
      name = re.sub(name_pattern, "", name)
      return name.split(":")[-1]

    unit = parse_unit(item.attrib["unitRef"])
    unit_store.add_unit(unit)

    return {
      parse_name(cast(str, segment.text)): Member(
        dim=segment.attrib["dimension"].split(":")[-1],
        value=float(cast(str, item.text)),
        unit=unit,
      )
    }

  async def fetch(url: str) -> et.Element:
    # cache_transport = hishel.AsyncCacheTransport(transport=httpx.AsyncHTTPTransport())
    async with hishel.AsyncCacheClient() as client:
      response = await client.get(url, headers=HEADERS)
      return et.fromstring(response.content)

  root = await fetch(url)
  if root.tag == "Error":
    cik, doc_id = url.split("/")[6:8]
    doc_id = insert_characters(doc_id, {"-": [10, 12]})
    time.sleep(1)
    url = await parse_xbrl_url(int(cik), doc_id)
    root = await fetch(url)

  form = {"10-K": "annual", "20-F": "annual", "40-F": "annual", "10-Q": "quarterly"}

  scope = cast(
    Scope, form[cast(str, cast(et.Element, root.find(".{*}DocumentType")).text)]
  )
  date = dt.strptime(
    cast(str, cast(et.Element, root.find(".{*}DocumentPeriodEndDate")).text), "%Y-%m-%d"
  )

  fiscal_end = cast(
    str, cast(et.Element, root.find(".{*}CurrentFiscalYearEndDate")).text
  )
  fiscal_end = re.sub("^-+", "", fiscal_end)
  fiscal_pattern = r"(0[1-9]|1[0-2])-(0[1-9]|12[0-9]|3[01])"

  match = re.search(fiscal_pattern, fiscal_end)
  fiscal_end_month = int(cast(re.Match[str], match).group(1))

  fiscal_period = cast(
    str, cast(et.Element, root.find(".{*}DocumentFiscalPeriodFocus")).text
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

  period_store = FinPeriodStore()
  unit_store = UnitStore()

  data: FinData = {}

  name_pattern = (
    r"((?<!Level)(Zero(?!Coupon)|One|Two|Three|Four|Five|Six|Seven|Eight|Nine))+\w+$"
  )

  for item in root.findall(".//*[@unitRef]"):
    if item.text is None:
      continue

    scrap = FinRecord()

    ctx = item.attrib["contextRef"]
    period_el = cast(
      et.Element,
      cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find("./{*}period"),
    )

    period = parse_period(period_el)
    period_store.add_period(period)

    scrap["period"] = period

    segment = cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find(
      ".//{*}segment/{*}explicitMember"
    )

    if segment is not None:
      scrap["members"] = parse_member(item, segment)
    else:
      scrap["value"] = float(item.text)
      unit = parse_unit(item.attrib["unitRef"])
      unit_store.add_unit(unit)
      scrap["unit"] = unit

    item_name = item.tag.split("}")[-1]
    item_name = re.sub(name_pattern, "", item_name)
    if item_name not in data:
      data[item_name] = [scrap]
      continue

    try:
      entry = next(i for i in data[item_name] if i["period"] == scrap["period"])

      if "members" in scrap:
        cast(dict[str, Member], entry.setdefault("members", {})).update(
          cast(dict[str, Member], scrap["members"])
        )
      else:
        entry.update(scrap)
    except Exception:
      data[item_name].append(scrap)

  # Sort items
  fin_periods, period_lookup = period_store.get_periods()
  units, unit_lookup = unit_store.get_units()

  data = fix_data(data, period_lookup, unit_lookup)
  return FinStatement(
    date=date.date(),
    fiscal_period=fiscal_period,
    fiscal_end=fiscal_end,
    url=[url],
    currency=currency,
    periods=fin_periods,
    units=units,
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
    root = et.fromstring(response.content)

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


async def get_ciks() -> DataFrame[CikFrame]:
  def get_tickers(group: pd.DataFrame) -> str:
    tickers = group["ticker"].tolist()
    return json.dumps(tickers)

  rename = {"cik_str": "cik", "title": "name"}

  url = "https://www.sec.gov/files/company_tickers.json"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    parse: dict[str, CikEntry] = response.json()

  df = pd.DataFrame.from_dict(parse, orient="index")
  df.rename(columns=rename, inplace=True)
  cik = df.groupby("cik").agg({"cik": "first", "name": "first"}, include_groups=False)
  cik["tickers"] = df.groupby("cik").apply(get_tickers, include_groups=False)

  return cast(DataFrame[CikFrame], cik)
