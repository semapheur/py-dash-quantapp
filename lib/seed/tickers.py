from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks


async def seed_stock_tickers():
  tickers = await get_tickers('stock')

  insert_sqlite(tickers, 'ticker.db', 'stock', 'replace', False)


async def seed_ciks():
  ciks = await get_ciks()

  insert_sqlite(ciks, 'ticker.db', 'edgar', 'replace', False)
