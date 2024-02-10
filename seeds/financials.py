import json
import time

import pandas as pd

from lib.db.lite import read_sqlite, insert_sqlite, get_tables
from lib.edgar.financials import update_financials, financials_table
from lib.ticker.fetch import get_fundamentals

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
    SELECT stock.id, edgar.cik FROM stock
    INNER JOIN edgar ON
      edgar.ticker = stock.ticker
    WHERE stock.mic = "{exchange}" AND stock.ticker IN (SELECT ticker FROM edgar)
    """

  df = read_sqlite('ticker.db', query)
  df.set_index('id', inplace=True)

  tables = get_tables('financials_scrap.db')
  new_ids = set(df.index).difference(tables)

  faulty: list[str] = []

  for id in new_ids:
    cik = df.at[id, 'cik']

    try:
      _ = await update_financials(int(cik), id)
      time.sleep(60)

    except Exception as _:
      faulty.append(id)
      print(f'{id} failed')

  if not faulty:
    return

  with open('seed_fail.json', 'r+') as f:
    content: dict = json.load(f)
    content[f'{exchange}_financials'] = faulty
    json.dump(content, f)


async def seed_fundamentals(exchange: str):
  query = 'SELECT id, name FROM stock WHERE mic=:=exchange'
  stocks = read_sqlite('ticker.db', query, params={'exchange': exchange})

  stock_ids = set(stocks['id']).intersection(get_tables('financials.db'))
  currency = SCREENER_CURRENCIES.get(exchange)

  seeded_ids = set(get_tables('fundamentals.db'))

  tables: dict[str, pd.DataFrame] = {}
  for i in stock_ids:
    if (table := f'{i}_{currency}') in seeded_ids:
      query = f'SELECT * FROM "{table}"'
      df = read_sqlite('fundamentals.db', query)
      if df.empty:
        continue

      tables[i] = df
    else:
      df = await financials_table(i, currency)

      insert_sqlite(df.reset_index(), 'financials.db', table, '')
      tables.append(df)

      financials_fetcher = partial(Company(cik).financials_to_df)
      ohlcv_fetcher = partial(Ticker(_id, 'stock', 'USD').ohlcv)

      fundamentals = asyncio.run(
        get_fundamentals(_id, financials_fetcher, ohlcv_fetcher)
      )

      fundamentals.index = fundamentals.index.set_levels(
        fundamentals.index.levels[0].strftime('%Y-%m-%d'), level='date'
      )
      fundamentals.reset_index(inplace=True)

  return fundamentals.to_dict('records')
