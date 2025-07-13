from datetime import datetime as dt
import re
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import TypedDict, cast

import aiometer
import httpx
from lxml import etree
import pandas as pd

from lib.const import HEADERS
from lib.fin.models import (
  FinData,
  FinPeriodStore,
  FinRecord,
  FinStatement,
  Instant,
  Duration,
  UnitStore,
  Member,
  fix_statement_data,
)
from lib.fin.statement import (
  statement_urls,
  upsert_merged_statements,
  upsert_statements,
)
from lib.utils.string import replace_all
from lib.utils.time import exclusive_end_date, month_difference
from lib.utils.validate import validate_currency


XMLNS = {
  "ix": "http://www.xbrl.org/2013/inlineXBRL",
  "xbrli": "http://www.xbrl.org/2003/instance",
  "xbrldi": "http://xbrl.org/2006/xbrldi",
}


class FilingInfo(TypedDict):
  slug: str
  date: str


def lei_filings(
  lei: str, page_size: int = 100, timeout: float = 1.0
) -> list[FilingInfo]:
  def fetch(lei: str, page_size: int, page: int, timeout: float = 1.0) -> dict:
    url = "https://filings.xbrl.org/api/filings"
    params = {
      "include": "entity",
      "filter": f'[{{"name":"entity.identifier","op":"eq","val":"{lei}"}}]',
      "sort": "-date_added",
      "page[size]": str(page_size),
      "page[number]": str(page),
    }

    with httpx.Client(timeout=timeout) as client:
      response = client.get(url, headers=HEADERS, params=params)
      return response.json()

  def parse_filings(parse: dict) -> list[FilingInfo]:
    result: list[FilingInfo] = []
    for filing in parse["data"]:
      if filing["type"] != "filing":
        continue

      filing_id: str = filing["attributes"]["fxo_id"]
      if filing_id in filing_ids:
        continue

      filing_ids.add(filing_id)
      slug: str | None = filing["attributes"].get("json_url")
      if slug is None:
        print(filing["attributes"])
        slug = filing["attributes"].get("report_url")
        if slug is None:
          continue

      result.append(
        FilingInfo(
          slug=slug,
          date=filing["attributes"].get("period_end"),
        )
      )

    return result

  parse = fetch(lei, page_size, 1, timeout)
  count = parse["meta"]["count"]

  filing_ids: set[str] = set()
  filing_infos = parse_filings(parse)
  if page_size >= count:
    return filing_infos

  pages = count // page_size + 1
  for page in range(2, pages + 1):
    parse = fetch(lei, page_size, page, timeout)
    filing_infos.extend(parse_filings(parse))

  return filing_infos


def parse_xbrl_period(
  xml: etree._Element, context: str, adjust_end_date: bool
) -> Instant | Duration:
  def parse_date(date_text: str):
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
    if match is None:
      raise ValueError(f'"{date_text}" does not match format "%Y-%m-%d"')

    return dt.strptime(match.group(), "%Y-%m-%d").date()

  period_nodes: list[etree._Element] = xml.xpath(
    f".//xbrli:context[@id='{context}']/xbrli:period", namespaces=XMLNS
  )

  if not period_nodes:
    raise ValueError(f"Period missing for context: {context}")

  period = period_nodes[0]

  # Check for instant tag
  instant_text: list[str] = period.xpath("./xbrli:instant/text()", namespaces=XMLNS)
  if instant_text:
    instant_date = parse_date(cast(str, instant_text[0]))
    if adjust_end_date:
      instant_date += relativedelta(days=1)
    return Instant(instant=instant_date)

  # Otherwise, expect start and end dates
  start_date_text: list[str] = period.xpath(
    "./xbrli:startDate/text()", namespaces=XMLNS
  )
  end_date_text: list[str] = period.xpath("./xbrli:endDate/text()", namespaces=XMLNS)

  if not (start_date_text and end_date_text):
    raise ValueError(f"Start or end date missing in period tag for context: {context}")

  start_date = parse_date(start_date_text[0])
  end_date = parse_date(end_date_text[0])

  months = month_difference(start_date, end_date)
  end_date = exclusive_end_date(start_date, end_date, months)

  return Duration(start_date=start_date, end_date=end_date, months=months)


def parse_xbrl_unit(xml: etree._Element, unit_id: str) -> str:
  unit_nodes: list[etree._Element] = xml.xpath(
    f'.//xbrli:unit[@id="{unit_id}"]', namespaces=XMLNS
  )

  if not unit_nodes:
    raise ValueError(f"Unit missing for unit_id: {unit_id}")
  unit = unit_nodes[0]

  # Check for single measure tag
  measure: list[str] = unit.xpath("./xbrli:measure/text()", namespaces=XMLNS)
  if measure:
    return measure[0].split(":")[-1].lower()

  # Otherwise, expect divide tag
  divides_nodes: list[etree._Element] = unit.xpath("./xbrli:divide", namespaces=XMLNS)
  if not divides_nodes:
    raise ValueError(f"Divide missing for unit_id: {unit_id}")

  divide = divides_nodes[0]

  numerator_text: list[str] = divide.xpath(
    "./xbrli:unitNumerator/xbrli:measure/text()", namespaces=XMLNS
  )[0]
  denominator_text: list[str] = divide.xpath(
    "./xbrli:unitDenominator/xbrli:measure/text()", namespaces=XMLNS
  )

  if not (numerator_text and denominator_text):
    raise ValueError(
      f"Numerator or denominator missing in divide tag for unit_id: {unit_id}"
    )

  numerator = numerator_text[0].split(":")[-1].lower()
  denominator = denominator_text[0].split(":")[-1].lower()
  return f"{numerator}/{denominator}"


def parse_xhtml_filing(filing_slug: str, date: str):
  url = f"https://filings.xbrl.org{filing_slug}"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    xml = response.content

  replacements = {" ": "", ",": ""}
  data: FinData = {}
  period_store = FinPeriodStore()
  unit_store = UnitStore()
  currencies: set[str] = set()

  root: etree._Element = etree.fromstring(xml)
  items: list[etree._Element] = root.xpath(".//ix:nonFraction", namespaces=XMLNS)
  for item in items:
    value_text: str | None = item.text
    if value_text is None:
      continue

    scrap = FinRecord()

    try:
      value = float(replace_all(value_text, replacements))
    except ValueError:
      value = None

    if value is not None and "scale" in item.attrib:
      value *= 10.0 ** int(item.attrib["scale"])

    period = parse_xbrl_period(root, item.attrib["contextRef"], False)
    period_store.add_period(period)
    scrap["period"] = period

    unit = parse_xbrl_unit(root, item.attrib["unitRef"])
    unit_store.add_unit(unit)
    if len(unit) == 3 and validate_currency(unit):
      currencies.add(unit)

    ctx = item.attrib["contextRef"]
    member_nodes: list[etree._Element] = root.xpath(
      f".//xbrli:context[@id='{ctx}']/xbrli:scenario/xbrldi:explicitMember",
      namespaces=XMLNS,
    )
    if member_nodes:
      member = member_nodes[0]
      member_name = cast(str, member.text).split(":")[-1]
      member_name = re.sub(r"(Segment)?Member", "", member_name)
      scrap["members"] = {
        member_name: Member(
          dim=cast(str, member.attrib["dimension"]).split(":")[-1],
          value=value,
          unit=unit,
        )
      }
    else:
      scrap["value"] = value
      scrap["unit"] = unit

    item_name = item.attrib["name"].split(":")[-1]
    if item_name not in data:
      data[item_name] = [scrap]
      continue

    entry = next((i for i in data[item_name] if i["period"] == scrap["period"]), None)
    if entry is None:
      data[item_name].append(scrap)
      continue

    if "members" in scrap:
      cast(dict[str, Member], entry.setdefault("members", {})).update(
        cast(dict[str, Member], scrap["members"])
      )
    else:
      entry.update(scrap)

  periods, period_lookup = period_store.get_periods()
  units, unit_lookup = unit_store.get_units()

  data = fix_statement_data(data, period_lookup, unit_lookup)

  statement = FinStatement(
    date=dt.strptime(date, "%Y-%m-%d").date(),
    fiscal_period="FY",
    fiscal_end=date[5:],
    url=[url],
    currency=currencies,
    periods=periods,
    units=units,
    data=data,
  )
  return statement


async def parse_json_filing(filing_slug: str, date: str) -> FinStatement:
  def parse_period(period_text: str) -> Instant | Duration:
    dates = period_text.split("/")

    if len(dates) == 1:
      date = dt.strptime(dates[0], "%Y-%m-%dT%H:%M:%S").date()
      return Instant(instant=date)

    start_date = dt.strptime(dates[0], "%Y-%m-%dT%H:%M:%S").date()
    end_date = dt.strptime(dates[1], "%Y-%m-%dT%H:%M:%S").date()
    months = month_difference(start_date, end_date)
    end_date = exclusive_end_date(start_date, end_date, months)

    return Duration(start_date=start_date, end_date=end_date, months=months)

  url = f"https://filings.xbrl.org{filing_slug}"

  async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=HEADERS)
    parse = response.json()

  data: FinData = {}
  period_store = FinPeriodStore()
  unit_store = UnitStore()
  currencies: set[str] = set()

  for entry in cast(dict, parse["facts"]).values():
    unit: str | None = entry["dimensions"].get("unit")
    if unit is None:
      continue

    scrap = FinRecord()

    item_name = entry["dimensions"]["concept"].split(":")[-1]

    value = 0.0 if entry["value"] is None else float(entry["value"])
    scrap["value"] = value

    period = parse_period(entry["dimensions"]["period"])
    period_store.add_period(period)
    scrap["period"] = period

    unit = unit.split("/")[0].split(":")[-1].lower()
    unit_store.add_unit(unit)
    scrap["unit"] = unit

    axis: str | None = next(
      (key for key in entry["dimensions"] if key.startswith("ifrs-full:")), None
    )
    if axis is not None:
      dim = axis.split(":")[-1]
      member = cast(str, entry["dimensions"][axis]).split(":")[-1]
      member = re.sub(r"(Segment)?Member", "", member)
      scrap["members"] = {
        member: Member(
          dim=dim,
          value=value,
          unit=unit,
        )
      }

    if len(unit) == 3 and validate_currency(unit):
      currencies.add(unit)

    if item_name not in data:
      data[item_name] = [scrap]
      continue

    entry = next((i for i in data[item_name] if i["period"] == scrap["period"]), None)
    if entry is None:
      data[item_name].append(scrap)
      continue

    if "members" in scrap:
      cast(dict[str, Member], entry.setdefault("members", {})).update(
        cast(dict[str, Member], scrap["members"])
      )
    else:
      entry.update(scrap)

  periods, period_lookup = period_store.get_periods()
  units, unit_lookup = unit_store.get_units()

  data = fix_statement_data(data, period_lookup, unit_lookup)

  statement = FinStatement(
    date=dt.strptime(date, "%Y-%m-%d").date(),
    fiscal_period="FY",
    fiscal_end=date[5:],
    url=[url],
    currency=currencies,
    periods=periods,
    units=units,
    data=data,
  )
  return statement


async def parse_xbrl_statements(filings: list[FilingInfo]) -> list[FinStatement]:
  tasks = [partial(parse_json_filing, f["slug"], f["date"]) for f in filings]
  financials = await aiometer.run_all(tasks, max_per_second=5)

  return financials


async def scrap_xbrl_statements(lei: str, id: str):
  filings = lei_filings(lei)

  financials = await parse_xbrl_statements(filings)
  upsert_merged_statements("statements.db", id, financials)


async def update_xbrl_statements(lei: str, id: str, delta=120):
  url_pattern = "https://filings.xbrl.org/%"
  old_filings = statement_urls("statements.db", id, url_pattern)

  if old_filings is None:
    await scrap_xbrl_statements(lei, id)
    return

  last_date = old_filings["date"].max()

  if relativedelta(dt.now(), last_date).days < delta:
    return

  new_filings = pd.DataFrame.from_records(lei_filings(lei))
  new_filings["date"] = pd.to_datetime(new_filings["date"], format="%Y-%m-%d")
  new_filings = new_filings.loc[new_filings["date"] >= last_date, :]

  if not new_filings:
    return

  new_filings["url"] = new_filings["slug"].apply(
    lambda x: f"https://filings.xbrl.org/{x}"
  )
  new_urls = set(new_filings["url"])
  old_urls = set(old_filings["url"])

  new_urls = new_urls.difference(old_urls)

  if not new_urls:
    return

  mask = new_filings["url"].isin(new_urls)
  new_filings = new_filings.loc[mask, ["slug", "date"]].to_dict("records")
  new_statements = await parse_xbrl_statements(new_filings)
  if new_statements:
    upsert_statements("statements.db", id, new_statements)
