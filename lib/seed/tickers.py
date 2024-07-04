import hashlib

from lib.db.lite import insert_sqlite, read_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks
from lib.mic import get_mics


def hash_companies(companies: list[list[str]], hash_length=10) -> dict[str, list[str]]:
  result: dict[str, list[str]] = {}
  hashes: set[str] = set()

  def generate_hash(company: str, suffix=''):
    base = company + suffix
    return hashlib.sha256(base.encode()).hexdigest()[:hash_length]

  for company in companies:
    name = min(company, key=len)
    hash_value = generate_hash(name)
    suffix = 0

    while hash_value in result:
      suffix += 1
      hash_value = generate_hash(name, str(suffix))

    hashes.add(hash_value)
    result[hash_value] = company

  return result


def find_index(nested_list: list[list[str]], query: str) -> int:
  for i, sublist in enumerate(nested_list):
    if query in sublist:
      return i

  return -1


async def seed_stock_tickers():
  tickers = await get_tickers('stock')
  insert_sqlite(tickers, 'ticker.db', 'stock', 'replace', False)


async def seed_ciks():
  ciks = await get_ciks()

  insert_sqlite(ciks, 'ticker.db', 'edgar', 'replace', False)


async def seed_exchanges():
  mics = get_mics()

  query = 'SELECT DISTINCT mic, currency FROM stock'
  currencies = read_sqlite('ticker.db', query)
  if currencies is None:
    await seed_stock_tickers()
    currencies = read_sqlite('ticker.db', query)

  mics = mics.merge(currencies, on='mic', how='left')

  insert_sqlite(mics, 'ticker.db', 'exchange', 'replace', False)
