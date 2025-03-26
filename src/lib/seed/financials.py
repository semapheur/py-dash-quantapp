from contextlib import closing
import json
from functools import partial
import logging
import sqlite3
import time
from typing import cast

import aiometer
import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm

from lib.db.lite import fetch_sqlite, read_sqlite, upsert_strings, get_tables
from lib.edgar.parse import update_edgar_statements
from lib.fin.fundamentals import update_fundamentals
from lib.ticker.fetch import get_primary_securities, update_company_lei
from lib.gleif.fetch import lei_by_isin
from lib.log.setup import setup_logging
from lib.xbrl.filings import update_xbrl_statements


setup_logging()
logger = logging.getLogger(__name__)


def update_primary_securities(
  company_id: str, securities: list[str], currency: str
) -> None:
  from lib.const import DB_DIR

  db_path = DB_DIR / "ticker.db"
  with closing(sqlite3.connect(db_path)) as con:
    cursor = con.cursor()

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
    con.close()


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


async def seed_xbrl_statements(exchange: str) -> None:
  query = """
    SELECT DISTINCT
      stock.company_id AS company_id,
      stock.isin AS isin,
      company.lei AS lei
    FROM stock
    INNER JOIN company ON company.company_id = stock.company_id
    WHERE stock.mic = :exchange
  """

  df = read_sqlite("ticker.db", query, {"exchange": exchange})

  missing_leis = df.loc[pd.isnull(df["lei"])]
  if not missing_leis.empty:
    tasks = [partial(lei_by_isin, isin) for isin in missing_leis["isin"]]
    leis = await aiometer.run_all(tasks, max_per_second=5)
    missing_leis["lei"] = leis
    missing_leis = missing_leis.loc[pd.notnull(missing_leis["lei"])]

    update_company_lei(missing_leis[["company_id", "lei"]].to_dict(orient="records"))
    df.update(missing_leis)

  faulty: list[str] = []
  for id, lei in zip(df["company_id"], df["lei"]):
    try:
      await update_xbrl_statements(lei, id)
      time.sleep(1)

    except Exception as e:
      print(e)
      faulty.append(id)
      print(f"{id} failed")

  if not faulty:
    return

  with open("logs/seed_fail.json", "w+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_statements_xbrl"] = faulty
    json.dump(content, f)


async def seed_edgar_statements(exchange: str) -> None:
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
      await update_edgar_statements(int(cik), id)
      time.sleep(1)

    except Exception as e:
      print(e)
      faulty.append(id)
      print(f"{id} failed")

  if not faulty:
    return

  with open("logs/seed_fail.json", "w+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_statements_edgar"] = faulty
    json.dump(content, f)


async def seed_company_statements(company_id: str) -> None:
  query_cik = """SELECT e.cik AS cik FROM edgar e
    INNER JOIN stock s ON s.isin = e.isin
    WHERE s.company_id = :company_id
  """

  result = fetch_sqlite("ticker.db", query_cik, {"company_id": company_id})
  cik = result[0][0] if result else None
  if cik is not None:
    await update_edgar_statements(int(cik), company_id)

  query_isin_lei = """
    SELECT
      s.isin AS isin,
      c.lei 
    FROM stock s
    INNER JOIN company c ON c.company_id = s.company_id
    WHERE s.company_id = :company_id
  """

  result = fetch_sqlite("ticker.db", query_isin_lei, {"company_id": company_id})
  if not result:
    return

  isin, lei = result[0]

  if lei is None:
    lei = await lei_by_isin(isin)
    if lei is not None:
      update_company_lei([{"company_id": company_id, "lei": lei}])

  if lei is None:
    return

  await update_xbrl_statements(lei, company_id)


async def seed_exchange_statements(exchange: str) -> None:
  await seed_xbrl_statements(exchange)
  await seed_edgar_statements(exchange)


async def seed_fundamentals(exchange: str):
  query = "SELECT DISTINCT company_id FROM stock WHERE mic = :exchange"
  companies = read_sqlite("ticker.db", query, params={"exchange": exchange})

  if companies is None:
    raise ValueError(f"No tickers found for {exchange}")

  seeded_companies = set(companies["company_id"]).intersection(
    get_tables("statements.db")
  )
  if not seeded_companies:
    raise ValueError(f"No companies seeded for {exchange}")

  faulty: list[str] = []
  stored_company: list[str] = []
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
      stored_company.append(company)

    except Exception as e:
      logger.error(e, exc_info=True)
      print(f"{company} failed")

  if stored_company:
    upsert_strings("ticker.db", "financials", "company_id", stored_company)
    upsert_strings("ticker.db", "stored_exchanges", "mic", [exchange])

  if not faulty:
    return

  with open("logs/seed_fail.json", "r+") as f:
    content: dict = json.load(f)
    content[f"{exchange}_fundamentals"] = faulty
    json.dump(content, f)
