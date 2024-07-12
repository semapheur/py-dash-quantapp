import json
import logging
import sqlite3
import time
from typing import cast

import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm

from lib.db.lite import read_sqlite, upsert_sqlite, get_tables
from lib.edgar.parse import update_statements
from lib.fin.fundamentals import update_fundamentals
from lib.ticker.fetch import get_primary_securities
from lib.log.setup import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def update_primary_securities(
  company_id: str, securities: list[str], currency: str
) -> None:
  from lib.const import DB_DIR

  db_path = DB_DIR / "ticker.db"
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  securities_json = json.dumps(securities)

  # Replace the entire array
  cursor.execute(
    """
      UPDATE company
      SET primary_securities = json(:securities), currency = :currency
      WHERE company_id = :company_id?
    """,
    {"securities": securities_json, "currency": currency, "company_id": company_id},
  )
  conn.commit()
  conn.close()


def select_menu(options) -> list[int]:
  for i, option in enumerate(options, 1):
    print(f"{i}. {option}")

  prompt = "Enter your choices: "
  while True:
    try:
      choices = [int(choice.strip()) for choice in input(prompt).split(",")]

      if all(1 <= choice <= len(options) for choice in choices):
        return [choice - 1 for choice in choices]
      else:
        print("Invalid choice(s). Please try again.")
    except ValueError:
      print("Please enter valid numbers separated by commas.")


async def seed_edgar_financials(exchange: str) -> None:
  query = """SELECT DISTINCT stock.company_id AS company_id, edgar.cik AS cik FROM stock
    INNER JOIN edgar ON edgar.isin = stock.isin
    WHERE stock.mic = :exchange OR stock.company_id IN (
      SELECT DISTINCT company_id FROM stock
      WHERE mic = :exchange
    )
  """

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
  query = "SELECT DISTINCT company_id FROM stock WHERE mic = :exchange"
  companies = read_sqlite("ticker.db", query, params={"exchange": exchange})

  if companies is None:
    raise ValueError(f"No tickers found for {exchange}")

  seeded_companies = set(companies["company_id"]).intersection(
    get_tables("financials.db")
  )
  if not seeded_companies:
    raise ValueError(f"No companies seeded for {exchange}")

  faulty: list[str] = []
  stored_company: list[dict[str, str]] = []
  for company in tqdm(seeded_companies):
    securities = get_primary_securities(company)
    currencies = securities["currency"].unique()

    update = False
    while len(securities) > 1 and len(currencies) > 1:
      print(
        f"{company} has multiple primary securities with different currencies. Select the correct ones."
      )
      options = [
        f"{t}.{e} ({c})"
        for t, e, c in zip(
          securities["ticker"], securities["mic"], securities["currency"]
        )
      ]
      ix = select_menu(options)
      securities = cast(DataFrame, securities.iloc[ix])
      currencies = securities["currency"].unique()
      update = True

    ticker_ids = securities["security_id"].tolist()
    currency = cast(str, currencies[0])
    if update:
      update_primary_securities(company, ticker_ids, currency)

    try:
      _ = await update_fundamentals(company, ticker_ids, currency)
      stored_company.append({"company_id": company})

    except Exception as e:
      logger.error(e, exc_info=True)
      print(f"{company} failed")

  if stored_company:
    df = pd.DataFrame.from_records(stored_company, index="company_id")
    upsert_sqlite(df, "ticker.db", "financials")

    df = pd.DataFrame.from_records({"mic": [exchange]}, index="mic")
    upsert_sqlite(df, "ticker.db", "stored_exchanges")

  if not faulty:
    return

  with open("logs/seed_fail.json", "r+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_financials"] = faulty
    json.dump(content, f)
