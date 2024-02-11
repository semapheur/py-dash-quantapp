from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers


async def seed_stock_tickers():
  tickers = await get_tickers('stock')

  insert_sqlite(tickers, 'tickers.db', 'stock', 'replace', False)
