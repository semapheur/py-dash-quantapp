from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks
from lib.fuzz import group_fuzzy_matches

trim_words = [  # (?!^)
  r'\s[.-:]+\s' r'\d(\.\d+)?\s?%',
  r'-([a-z]|\d+?)-',
  r'd/d+(th)?',
  r'1(/\d)? r(ig)?ht?s?',
  'ab',
  r'a\.?dr?',
  'ag',
  'alien market',
  r'a/?sa?',
  'bearer',
  'bhd',
  'brdr',
  'cad',
  r'c?dr',
  'cedear',
  r'(one-(half)? )?cl(as)?s [a-z]',
  r'co(rp)?',
  r'dep(osits)?',
  r'depository (interest|receipts?)',
  r'exch(angeable)?',
  'fixed',
  'fltg',
  'foreign',
  'gbp',
  'gmbh',
  r'\(?[a-z]{3} hedged\)?',
  'inc',
  'into',
  'jsc',
  r'kc?sc',
  'kgaa',
  'ltd',
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
  'repr',
  'restricted',
  r'\(?rs\.\d{1,2}(\.\d{2})?\)?',
  'rt',
  r's\.?a\.?',
  'sae',
  'sak',
  'saog',
  r'ser(ie)?s? [a-z0-9]',
  r'sh(are)?s?',
  'spa',
  'sr',
  'sub',
  'tao',
  r'(unitary )?(144a/)?reg s',
  r'units?',
  r'undated( [a-z]{3})',
  r'(\d )?vote',
  r'(one(-half)? )?war(rant)?s?',
  'without',
]


def find_index(nested_list: list[list[str]], query: str) -> int:
  for i, sublist in enumerate(nested_list):
    if query in sublist:
      return i

  return -1


async def seed_stock_tickers(group_companies: bool = False):
  tickers = await get_tickers('stock')

  if group_companies:
    companies = group_fuzzy_matches(
      tickers['name'].sort_value().unique(), trim_words=trim_words
    )
    tickers['company_id'] = tickers['name'].apply(lambda x: find_index(companies, x))

  insert_sqlite(tickers, 'ticker.db', 'stock', 'replace', False)


async def seed_ciks():
  ciks = await get_ciks()

  insert_sqlite(ciks, 'ticker.db', 'edgar', 'replace', False)
