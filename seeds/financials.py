import json
import time

from lib.db.lite import read_sqlite, get_tables
from lib.edgar.financials import scrap_financials

faulty = ['0P0000BV6H', '0P0001691U', '0P0000BRJU', '0P0000C80Q']


async def xosl_financials() -> None:
  query = """
    SELECT stock.id, edgar.cik FROM stock
    INNER JOIN edgar ON
      edgar.ticker = stock.ticker
    WHERE stock.mic = "XOSL" AND stock.ticker IN (SELECT ticker FROM edgar)
    """

  df = read_sqlite('ticker.db', query)
  df.set_index('id', inplace=True)

  tables = get_tables('financials_scrap.db')
  new_ids = set(df.index).difference(tables)

  faulty: list[str] = []

  for id in new_ids:  # cik in zip(df['id'], df['cik']):
    cik = df.at[id, 'cik']

    try:
      await scrap_financials(int(cik), id)
      time.sleep(60)

    except Exception as _:
      faulty.append(id)
      print(f'{id} failed')

  if not faulty:
    return

  with open('seed_fail.json', 'r+') as f:
    content: dict = json.load(f)
    content['xosl_financials'] = faulty

    json.dump(content, f)
