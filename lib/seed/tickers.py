import hashlib
from rapidfuzz import fuzz

from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks
from lib.fuzz import group_fuzzy_matches

trim_words = [  # (?!^)
  r'\s[.-:]+\s',
  r'\d(\.\d+)?\s?%',
  r'\d/\d+(th)?',
  r'-([a-z]|\d+?)-',
  r'd/d+(th)?',
  'ab',
  r'a\.?dr?',
  'ag',
  'alien market',
  r'a/?sa?',
  'bearer',
  'bhd',
  'brdr',
  r'\(?buyback\)?',
  'cad',
  r'c?dr',
  'cedear',
  r'(one-(half)? )?cl(as)?s -?[a-z]-?',
  r'dep(osits?)?',
  r'(((brazili|canadi|kore)an|taiwan) )?deposit(a|o)ry (interests?|receipts?)',
  r'exch(angeable)?',
  'fixed',
  'fltg',
  'foreign',
  'fxdfr ',
  'gbp',
  'gmbh',
  r'\(?[a-z]{3} hedged\)?',
  'inc',
  r'int(terests?)?',
  'into',
  'jsc',
  r'kc?sc',
  'kgaa',
  'lp',
  'maturity',
  'na',
  r'\(new\)',
  r'(non)?-?conv(ert((a|i)ble)?)?',
  r'(non)?-?cum',
  r'(limited|non|sub(ord)?)?-?vo?t(in)?g',
  r'nv(dr)?',
  r'ord(inary)?',
  'partly paid',
  'pcl',
  r'perp(etual)?( [a-z]{3})?',
  'pfd',
  'php',
  'plc',
  'pref',
  'prf',
  'psc',
  'red',
  r'registere?d',
  r'repr(\.|esents)?',
  'restricted',
  r'r(ig)?ht?s?',
  r'\(?rs\.\d{1,2}(\.\d{2})?\)?',
  'rt',
  r's\.?a\.?',
  'sae',
  'sak',
  'saog',
  r'ser(ies?)? -?[a-z0-9]-?',
  r'sh(are)?s?',
  'spa',
  'sr',
  'sub',
  'tao',
  'tbk',
  r'(unitary )?(144a/)?reg s',
  r'units?',
  r'undated( [a-z]{3})',
  r'(un)?sponsored',
  r'(\d )?vote',
  r'(one(-half)? )?war(rant)?s?',
  'without',
]


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
