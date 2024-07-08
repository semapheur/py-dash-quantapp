import json
import logging
import time
from textwrap import dedent
from typing import cast

import pandas as pd
from tqdm import tqdm

from lib.db.lite import read_sqlite, upsert_sqlite, get_tables
from lib.edgar.parse import update_statements
from lib.fin.fundamentals import update_fundamentals

from lib.log.setup import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def get_currency(exchange: str) -> str:
  query = "SELECT currency FROM (SELECT DISTINCT mic, currency FROM stock WHERE mic = :exchange)"
  currency = read_sqlite("ticker.db", query, {"exchange": exchange})

  if currency is None:
    raise ValueError(f"No currency found for {exchange}")

  return cast(str, currency.loc[0, "currency"])


async def seed_edgar_financials(exchange: str) -> None:
  query = dedent(
    """
    SELECT DISTINCT stock.company_id AS company_id, edgar.cik AS cik FROM stock
    INNER JOIN edgar ON edgar.isin = stock.isin
    WHERE stock.mic = :exchange OR stock.company_id IN (
      SELECT DISTINCT company_id FROM stock
      WHERE mic = :exchange
    )
    """
  )
  df = read_sqlite("ticker.db", query, {"exchange": exchange})
  if df is None:
    raise ValueError(f"No tickers found for {exchange}")

  faulty: list[str] = []
  for id, cik in zip(df["company_id"], df["cik"]):
    try:
      _ = await update_statements(int(cik), id)
      time.sleep(1)

    except Exception as e:
      print(e)
      faulty.append(id)
      print(f"{id} failed")

  if not faulty:
    return

  with open("logs/seed_fail.json", "w+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_financials"] = faulty
    json.dump(content, f)


async def seed_fundamentals(exchange: str):
  query = "SELECT id, company_id, name FROM stock WHERE mic = :exchange"
  tickers = read_sqlite("ticker.db", query, params={"exchange": exchange})

  if tickers is None:
    raise ValueError(f"No tickers found for {exchange}")

  seeded_companies = set(tickers["company_id"]).intersection(
    get_tables("financials.db")
  )
  if not seeded_companies:
    raise ValueError(f"No financials found for {exchange}")

  currency = get_currency(exchange)
  faulty: list[str] = []
  stored: list[dict[str, str]] = []
  for company in tqdm(seeded_companies):
    try:
      ticker_ids = tickers.loc[tickers["company_id"] == company, "id"].tolist()
      _ = await update_fundamentals(company, ticker_ids, currency)
      stored.append({"id": company, "currency": currency})

    except Exception as e:
      logger.error(e, exc_info=True)
      print(f"{company} failed")

  if stored:
    df = pd.DataFrame.from_records(stored)
    df.set_index(("id", "currency"), inplace=True)
    upsert_sqlite(df, "tickers.db", "fundamentals")

  if not faulty:
    return

  with open("logs/seed_fail.json", "r+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_financials"] = faulty
    json.dump(content, f)
