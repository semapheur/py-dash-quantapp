from datetime import datetime as dt, date as Date
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import TypedDict

import aiometer
import httpx
import pandas as pd

from lib.const import HEADERS
from lib.fin.models import FinData, FinRecord, FinStatement, Instant, Interval
from lib.fin.statement import df_to_statements, load_statements, upsert_statements
from lib.utils import month_difference


class FilingInfo(TypedDict):
  json_url: str
  date: str


def lei_filings(lei: str, page_size: int = 100, timeout: float = 1.0):
  def fetch(lei: str, page_size: int, page: int, timeout: float = 1.0) -> dict:
    url = "https://filings.xbrl.org/api/filings"
    params = {
      "include": "entity",
      "filter": f'[{{"name":"entity.identifier","op":"eq","val":{lei}}}]',
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
          json_url=filing["attributes"].get("json_url"),
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

  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
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

  statement = FinStatement(
    url=[url],
    scope="annual",
    date=dt.strptime(date, "%Y-%m-%d").date(),
    fiscal_period="FY",
    fiscal_end=date[5:],
    currency=currencies,
    data=data,
  )
  return statement


async def parse_statements(filings: list[FilingInfo]) -> list[FinStatement]:
  tasks = [partial(parse_statement, f["json_url"], f["date"]) for f in filings]
  financials = await aiometer.run_all(tasks, max_per_second=5)

  return financials


async def scrap_statements(lei: str, id: str) -> list[FinStatement]:
  filings = lei_filings(lei)

  financials = await parse_statements(filings)
  upsert_statements("statements.db", id, financials)
  return financials


async def update_statements(
  lei: str, id: str, delta=120, date: Date | None = None
) -> list[FinStatement]:
  df = await load_statements(id, date)

  if df is None:
    return await scrap_statements(lei, id)

  last_date = df["date"].max()

  if relativedelta(dt.now(), last_date).days < delta:
    return df_to_statements(df)

  new_filings = lei_filings(lei)
  new_filings = pd.DataFrame.from_records(new_filings)
  new_filings["date"] = pd.to_datetime(new_filings["date"])
  new_filings = new_filings[new_filings["date"] > last_date]

  if not new_filings:
    return df_to_statements(df)

  old_filings = set(df["id"])
  filings_diff = set(new_filings.index).difference(old_filings)

  if not filings_diff:
    return df_to_statements(df)

  new_fin = await parse_statements(new_filings.tolist())
  if new_fin:
    upsert_statements("statements.db", id, new_fin)

  return [*new_fin, *df_to_statements(df)]
