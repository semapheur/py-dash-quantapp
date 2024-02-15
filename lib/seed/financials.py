from functools import partial
import json
import time

import pandas as pd
from tqdm import tqdm

from lib.db.lite import read_sqlite, upsert_sqlite, get_tables
from lib.edgar.parse import update_statements
from lib.fin.fundamentals import update_fundamentals
from lib.morningstar.ticker import Stock

faulty = ['0P0000BV6H', '0P0001691U', '0P0000BRJU', '0P0000C80Q']
empty = [
  '0P0001O69E',
  '0P0000J4UJ',
  '0P0001PCD9',
  '0P000X9JZ',
  '0P0000AOB5',
  '0P0001MX0L',
]

SCREENER_CURRENCIES = {'XOSL': 'NOK'}


async def seed_edgar_financials(exchange: str) -> None:
  query = f"""
    SELECT stock.id AS id, edgar.cik AS cik FROM stock
    INNER JOIN edgar ON
      edgar.ticker = stock.ticker
    WHERE stock.mic = "{exchange}" AND stock.ticker IN (SELECT ticker FROM edgar)
    """

  df = read_sqlite('ticker.db', query)
  if df is None:
    raise ValueError(f'No tickers found for {exchange}')

  faulty: list[str] = []
  for id, cik in zip(df['id'], df['cik']):
    try:
      _ = await update_statements(int(cik), id)
      time.sleep(60)

    except Exception as e:
      print(e)
      faulty.append(id)
      print(f'{id} failed')

  if not faulty:
    return

  with open('logs/seed_fail.json', 'w+') as f:
    content: dict = json.load(f)
    content[f'{exchange}_financials'] = faulty
    json.dump(content, f)


async def seed_fundamentals(exchange: str):
  query = 'SELECT id, name FROM stock WHERE mic = :exchange'
  tickers = read_sqlite('ticker.db', query, params={'exchange': exchange})

  if tickers is None:
    raise ValueError(f'No tickers found for {exchange}')

  seeded_ids = set(tickers['id']).intersection(get_tables('financials.db'))
  if not seeded_ids:
    raise ValueError(f'No financials found for {exchange}')

  currency = SCREENER_CURRENCIES['XOSL']
  faulty: list[str] = []
  stored: list[dict[str, str]] = []
  for id in tqdm(seeded_ids):
    try:
      ohlcv_fetcher = partial(Stock(id, currency).ohlcv)
      _ = await update_fundamentals(id, currency, ohlcv_fetcher)
      stored.append({'id': id, 'currency': currency})

    except Exception as e:
      faulty.append(id)
      print(f'{id} failed')
      print(e)

  if stored:
    df = pd.DataFrame.from_records(stored)
    df.set_index(('id', 'currency'), inplace=True)
    upsert_sqlite(df, 'tickers.db', 'fundamentals')

  if not faulty:
    return

  with open('logs/seed_fail.json', 'r+') as f:
    content: dict = json.load(f)
    content[f'{exchange}_financials'] = faulty
    json.dump(content, f)
