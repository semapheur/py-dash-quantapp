from datetime import datetime as dt

import httpx

from lib.const import HEADERS
from lib.fin.models import FinData, FinRecord, Instant, Interval
from lib.utils import month_difference


def get_filings(country: str, page_size: int = 10000):
  url = (
    "https://filings.xbrl.org/api/filings?include=entity,language"
    f'&filter=[{{"name"%3A"country"%2C"op"%3A"eq"%2C"val"%3A"{country}"}}]'
    f"&sort=-date_added&page[size]={page_size}&page[number]=1&_=1738274381858"
  )

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    parse = rs.json()

  return parse


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

  for entry in parse["facts"].values():
    unit = entry["dimensions"].get("unit")
    if unit is None:
      continue

    scrap = FinRecord()

    item_name = entry["dimensions"]["concept"].split(":")[-1]

    scrap["value"] = float(entry["value"])
    scrap["period"] = parse_period(entry["dimensions"]["period"])
    scrap["unit"] = unit.split(":")[-1]

    data.setdefault(item_name, []).append(scrap)

  return data
