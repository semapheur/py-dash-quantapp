from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import TypedDict

import aiometer
import httpx
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
)
from lib.fin.statement import (
  statement_urls,
  upsert_merged_statements,
  upsert_statements,
)
from lib.utils import exclusive_end_date, month_difference


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
      rs = client.get(url, headers=HEADERS, params=params)
      return rs.json()

  def parse_filings(parse: dict):
    result: list[FilingInfo] = []
    for filing in parse["data"]:
      if filing["type"] != "filing":
        continue

      filing_id: str = filing["attributes"]["fxo_id"]
      if filing_id in filing_ids:
        continue

      filing_ids.add(filing_id)

      result.append(
        FilingInfo(
          slug=filing["attributes"].get("json_url"),
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


async def parse_statement(filing_slug: str, date: str) -> FinStatement:
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

  def fix_data(
    data: FinData,
    period_lookup: dict[Duration | Instant, str],
    unit_lookup: dict[str, int],
  ) -> FinData:
    return {
      k: [
        {
          **item,
          "period": period_lookup[item["period"]],
          "unit": unit_lookup[item["unit"]],
        }
        for item in v
      ]
      for k, v in sorted(data.items())
    }

  url = f"https://filings.xbrl.org/{filing_slug}"

  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    parse = rs.json()

  data: FinData = {}
  period_store = FinPeriodStore()
  unit_store = UnitStore()
  currencies: set[str] = set()

  for entry in parse["facts"].values():
    unit: str | None = entry["dimensions"].get("unit")
    if unit is None:
      continue

    scrap = FinRecord()

    item_name = entry["dimensions"]["concept"].split(":")[-1]

    scrap["value"] = float(entry["value"])
    period = parse_period(entry["dimensions"]["period"])
    period_store.add_period(period)
    scrap["period"] = period
    unit = unit.split("/")[0].split(":")[-1].lower()
    unit_store.add_unit(unit)
    scrap["unit"] = unit

    if len(unit) == 3:
      currencies.add(unit)

    data.setdefault(item_name, []).append(scrap)

  periods, period_lookup = period_store.get_periods()
  units, unit_lookup = unit_store.get_units()

  data = fix_data(data, period_lookup, unit_lookup)

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
  tasks = [partial(parse_statement, f["slug"], f["date"]) for f in filings]
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
