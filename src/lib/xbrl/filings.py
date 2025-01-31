from datetime import datetime as dt

import httpx

from lib.const import HEADERS
from lib.fin.models import FinData, FinRecord, Instant, Interval
from lib.utils import month_difference


def get_filings(country: str, page_size: int = 10000):
  def fetch(country: str, page_size: int, page: int) -> dict:
    url = (
      "https://filings.xbrl.org/api/filings?include=entity"
      f'&filter=[{{"name":"country","op":"eq","val":"{country}"}}]'
      f"&sort=-date_added&page[size]={page_size}&page[number]={page}"
    )

    with httpx.Client(timeout=20.0) as client:
      rs = client.get(url, headers=HEADERS)
      return rs.json()

  def parse_filings(filings_by_company: dict, parse: dict):
    id_map: dict[str, str] = {}

    for entity in parse["included"]:
      if entity["type"] != "entity":
        continue

      id = entity["attributes"]["identifier"]
      name = entity["attributes"]["name"]

      id_map[id] = name

    for filing in parse["data"]:
      if filing["type"] != "filing":
        continue

      filing_id: str = filing["attributes"]["fxo_id"]
      company_id = filing_id.split("-")[0]
      company_name = id_map.get(company_id, company_id)

      company_filings = filings_by_company.setdefault(company_name, {})

      if filing_id in company_filings:
        continue

      company_filings[filing_id] = {
        "json_url": filing["attributes"].get("json_url"),
        "date": filing["attributes"].get("period_end"),
      }

  parse = fetch(country, page_size, 1)
  count = parse["meta"]["count"]

  filings_by_company: dict[str, dict] = {}
  parse_filings(filings_by_company, parse)
  if page_size >= count:
    return filings_by_company

  pages = count // page_size + 1
  for page in range(2, pages + 1):
    parse = fetch(country, page_size, page)
    parse_filings(filings_by_company, parse)

  return filings_by_company


def get_statement(filing_slug: str):
  def parse_period(period_text: str) -> Instant | Interval:
    dates = period_text.split("/")

    if len(dates) == 1:
      date = dt.strptime(dates[0], "%Y-%m-%dT%H:%M:%S").date()
      return Instant(instant=date)

    start_date = dt.strptime(dates[0], "%Y-%m-%dT%H:%M:%S").date()
    end_date = dt.strptime(dates[1], "%Y-%m-%dT%H:%M:%S").date()
    months = month_difference(start_date, end_date)

    return Interval(start_date=start_date, end_date=end_date, months=months)

  url = f"https://filings.xbrl.org/{filing_slug}"

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    parse = rs.json()

  data: FinData = {}
  currencies: set[str] = set()

  for entry in parse["facts"].values():
    unit: str | None = entry["dimensions"].get("unit")
    if unit is None:
      continue

    scrap = FinRecord()

    item_name = entry["dimensions"]["concept"].split(":")[-1]

    scrap["value"] = float(entry["value"])
    scrap["period"] = parse_period(entry["dimensions"]["period"])
    unit = unit.split("/")[0].split(":")[-1]
    scrap["unit"] = unit
    if len(unit) == 3:
      currencies.add(unit)

    data.setdefault(item_name, []).append(scrap)

  return data, currencies
