import asyncio
from datetime import date as Date, datetime as dt
from enum import Enum
from functools import partial
import re
import sqlite3
from typing import cast, Literal, TypeAlias
import xml.etree.ElementTree as et

import aiometer
import httpx
import bs4 as bs
import pandas as pd
from pandera.typing import DataFrame, Series
from tinydb import TinyDB

# Local
from lib.const import DB_DIR, HEADERS
from lib.db.lite import read_sqlite, sqlite_name
from lib.edgar.models import (
  CikEntry,
  CikFrame,
  Financials,
  Instant,
  Interval,
  Item,
  Member,
  Scope,
  FiscalPeriod,
  FinData,
  StockSplit,
)
from lib.utils import (
  combine_duplicate_columns,
  df_time_difference,
  insert_characters,
  month_difference,
  fiscal_quarter,
)


UNIT_BLACKLIST = {
  'contract',
  'country',
  'county',
  'customer',
  'day',
  'item',
  'month',
  'number',
  'obligation',
  'performanceobligation',
  'pure',
  'shares',
  'store',
  'subsidiary',
  'vendor',
  'usdpershare',
  'usdpershareitemtype',
  'year',
}


class ScopeEnum(Enum):
  quarterly = 3
  annual = 12


Docs: TypeAlias = Literal['cal', 'def', 'htm', 'lab', 'pre']


async def xbrl_urls(cik: int, doc_ids: list[str], doc_type: Docs) -> Series[str]:
  tasks = [asyncio.create_task(xbrl_url(cik, doc_id, doc_type)) for doc_id in doc_ids]
  urls = await asyncio.gather(*tasks)

  result = pd.Series(urls, index=doc_ids)
  result = result.loc[result.notnull()]

  return result


async def xbrl_url(cik: int, doc_id: str, doc_type: Docs = 'htm') -> str:
  url = (
    'https://www.sec.gov/Archives/edgar/data/'
    f"{cik}/{doc_id.replace('-', '')}/{doc_id}-index.htm"
  )
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    parse = bs.BeautifulSoup(rs.text, 'lxml')

  if doc_type == 'htm':
    pattern = r'(?<!_(cal|def|lab|pre)).xml$'
  else:
    pattern = rf'_{doc_type}.xml$'

  a_node = parse.find('a', href=re.compile(pattern))
  assert a_node is not None, 'a tag containing XBRL href not found'

  href = cast(bs.Tag, a_node).get('href')
  return f'https://www.sec.gov{href}'


async def parse_statements(urls: list[str]) -> list[Financials]:
  tasks = [partial(parse_statement, url) for url in urls]
  financials = await aiometer.run_all(tasks, max_per_second=10)

  return financials


async def parse_statement(url: str) -> Financials:
  def fix_data(data: FinData) -> FinData:
    fixed: FinData = {}

    for k in sorted(data.keys()):
      temp: list[Item] = []

      for item in data[k]:
        if 'value' in item or (members := item.get('members')) is None:
          temp.append(item)
          continue

        if len(members) == 1:
          member = next(iter(members.values()))
          temp.append(
            Item(
              period=item['period'],
              value=cast(float | int, member.get('value')),
              unit=cast(str, member.get('unit')),
            )
          )
          continue

        value = 0.0
        units = set()
        for m in members.values():
          value += m.get('value', 0)
          if (unit := m.get('unit')) is not None:
            units.add(unit)

        if len(units) == 1:
          temp.append(
            Item(period=item['period'], value=value, unit=units.pop(), members=members)
          )

      fixed[k] = temp

    return fixed

  def parse_period(period: et.Element) -> Instant | Interval:
    if (el := period.find('./{*}instant')) is not None:
      return Instant(instant=dt.strptime(cast(str, el.text), '%Y-%m-%d').date())

    start_date = dt.strptime(
      cast(str, cast(et.Element, period.find('./{*}startDate')).text), '%Y-%m-%d'
    )
    end_date = dt.strptime(
      cast(str, cast(et.Element, period.find('./{*}endDate')).text), '%Y-%m-%d'
    )

    months = month_difference(start_date, end_date)

    return Interval(
      start_date=start_date.date(), end_date=end_date.date(), months=months
    )

  def parse_unit(unit: str) -> str:
    if '_' not in unit:
      return unit.lower()

    return unit.split('_')[-1].lower()

  def parse_member(item: et.Element, segment: et.Element) -> dict[str, Member]:
    def parse_name(name: str) -> str:
      name = re.sub(r'(Segment)?Member', '', name)
      return name.split(':')[-1]

    unit = parse_unit(item.attrib['unitRef'])
    if unit not in UNIT_BLACKLIST:
      currency.add(unit)

    return {
      parse_name(cast(str, segment.text)): Member(
        dim=segment.attrib['dimension'].split(':')[-1],
        value=float(cast(str, item.text)),
        unit=unit,
      )
    }

  async def fetch(url: str) -> et.Element:
    async with httpx.AsyncClient() as client:
      rs = await client.get(url, headers=HEADERS)
      return et.fromstring(rs.content)

  root = await fetch(url)
  if root.tag == 'Error':
    cik, doc_id = url.split('/')[6:8]
    doc_id = insert_characters(doc_id, {'-': [10, 12]})
    url = await xbrl_url(int(cik), doc_id)
    root = await fetch(url)

  form = {'10-K': 'annual', '10-Q': 'quarterly'}

  scope = cast(
    Scope, form[cast(str, cast(et.Element, root.find('.{*}DocumentType')).text)]
  )
  date = dt.strptime(
    cast(str, cast(et.Element, root.find('.{*}DocumentPeriodEndDate')).text), '%Y-%m-%d'
  )
  fiscal_end = cast(
    str, cast(et.Element, root.find('.{*}CurrentFiscalYearEndDate')).text
  )[1:]

  if (el := root.find('.{*}DocumentFiscalPeriodFocus')) is not None:
    fiscal_period = cast(FiscalPeriod, el.text)
  elif scope == 'annual':
    fiscal_period = 'FY'
  else:
    pattern = r'(\d{2})-(\d{2})'

    match = re.search(pattern, fiscal_end)
    month = int(cast(re.Match[str], match).group(1))
    day = int(cast(re.Match[str], match).group(2))

    fiscal_period = cast(FiscalPeriod, fiscal_quarter(date, month, day))

  doc_id = url.split('/')[-2]
  currency = set()
  data: FinData = {}

  for item in root.findall('.//*[@unitRef]'):
    if item.text is None:
      continue

    scrap = Item()

    ctx = item.attrib['contextRef']
    period_el = cast(
      et.Element,
      cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find('./{*}period'),
    )

    scrap['period'] = parse_period(period_el)

    segment = cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find(
      './/{*}segment/{*}explicitMember'
    )

    if segment is not None:
      scrap['members'] = parse_member(item, segment)
    else:
      scrap['value'] = float(item.text)
      unit = parse_unit(item.attrib['unitRef'])
      if unit not in UNIT_BLACKLIST:
        currency.add(unit)

      scrap['unit'] = unit

    item_name = item.tag.split('}')[-1]
    if item_name not in data:
      data[item_name] = [scrap]
      continue

    try:
      entry = next(i for i in data[item_name] if i['period'] == scrap['period'])

      if 'members' in scrap:
        cast(dict[str, Member], entry.setdefault('members', {})).update(
          cast(dict[str, Member], scrap['members'])
        )
      else:
        entry.update(scrap)
    except Exception:
      data[item_name].append(scrap)

  # Sort items
  data = fix_data(data)
  return Financials(
    id=doc_id,
    date=date.date(),
    scope=scope,
    period=fiscal_period,
    fiscal_end=fiscal_end,
    currency=currency,
    data=data,
  )


async def parse_taxonomy(url: str) -> pd.DataFrame:
  namespace = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink',
  }

  def rename_sheet(txt: str) -> str:
    pattern = r'income|balance|cashflow'
    m = re.search(pattern, txt, flags=re.I)
    if m:
      txt = m.group().lower()

    return txt

  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    root = et.fromstring(rs.content)

  url_pattern = r'^https?://www\..+/'
  el_pattern = r'(?<=_)[A-Z][A-Za-z]+(?=_)?'

  taxonomy = []
  for sheet in root.findall('.//link:calculationLink', namespaces=namespace):
    sheet_label = re.sub(url_pattern, '', sheet.attrib[f'{{{namespace["xlink"]}}}role'])
    sheet_label = rename_sheet(sheet_label)

    for el in sheet.findall('.//link:calculationArc', namespaces=namespace):
      taxonomy.append(
        {
          'sheet': sheet_label,
          'gaap': cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}to']),
          ).group(),
          'parent': cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}from']),
          ).group(),
        }
      )

  df = pd.DataFrame.from_records(taxonomy)
  df.set_index('item', inplace=True)
  df.drop_duplicates(inplace=True)
  return df


def statement_to_df(financials: Financials) -> pd.DataFrame:
  def parse_date(period: Instant | Interval) -> Date:
    if isinstance(period, Interval):
      return period.end_date

    return period.instant

  fin_date = financials.date
  fin_scope = financials.scope
  fin_period = financials.period

  df_data: dict[tuple[Date, FiscalPeriod, int], dict[str, int | float]] = {}

  for item, entries in financials.data.items():
    for entry in entries:
      date = parse_date(entry['period'])
      if date != fin_date:
        continue

      if isinstance(entry['period'], Interval):
        months = entry['period'].months
      else:
        months = ScopeEnum[fin_scope].value

      period = fin_period
      if fin_period == 'FY' and months < 12:
        period = 'Q4'

      if value := entry.get('value'):
        df_data.setdefault((fin_date, period, months), {})[item] = value

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[item] = value

      if (members := entry.get('members')) is None:
        continue

      for member, m_entry in members.items():
        if (m_value := m_entry.get('value')) is None:
          continue

        dim = '.' + d if (d := m_entry.get('dim')) else ''
        key = f'{item}{dim}.{member}'
        df_data.setdefault((fin_date, period, months), {})[key] = m_value

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or m_entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period', 'months']
  return df


def get_stock_splits(fin_data: FinData) -> list[StockSplit]:
  data: list[StockSplit] = []

  name = 'StockholdersEquityNoteStockSplitConversionRatio1'
  splits = fin_data.get(name)
  if splits is None:
    return data

  for entry in splits:
    value = cast(float, entry.get('value'))

    data.append(
      StockSplit(
        date=cast(Interval, entry['period']).start_date,
        stock_split_ratio=value,
      )
    )
  return data


def fix_financials_df(df: pd.DataFrame) -> pd.DataFrame:
  query = """
    SELECT json_each.value AS gaap, item FROM items 
    JOIN JSON_EACH(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  """
  items = read_sqlite('taxonomy.db', query)

  df = df[list(set(df.columns).intersection(set(items['gaap'])))]

  rename = {k: v for k, v in zip(items['gaap'], items['item'])}
  df.rename(columns=rename, inplace=True)
  df = combine_duplicate_columns(df)

  query = 'SELECT item FROM items WHERE period = "duration"'
  items = read_sqlite('taxonomy.db', query)
  diff_items = list(set(items['item']).intersection(set(df.columns)))

  conditions = (('Q1', 3), ('Q2', 6), ('Q3', 9), ('FY', 12))

  period = df.index.get_level_values('period')
  months = df.index.get_level_values('months')
  for i in range(1, len(conditions)):
    mask = (period == conditions[i - 1][0]) & (months == conditions[i - 1][1]) | (
      period == conditions[i][0]
    ) & (months == conditions[i][1])
    _df = df.loc[mask, diff_items]
    _df.sort_index(level='date', inplace=True)

    _df.loc[:, 'month_diff'] = df_time_difference(
      cast(pd.DatetimeIndex, _df.index.get_level_values('date')), 30, 'D'
    ).array

    _df.loc[:, diff_items] = _df[diff_items].diff()
    _df = _df.loc[_df['month_diff'] == 3, diff_items]

    _df = _df.loc[(slice(None), conditions[i][0], conditions[i][1]), :]
    _df.reset_index(level='months', inplace=True)
    _df.loc[:, 'months'] = 3
    _df.set_index('months', append=True, inplace=True)

    if conditions[i][0] == 'FY':
      _df.reset_index(level='period', inplace=True)
      _df.loc[:, 'period'] = 'Q4'
      _df.set_index('period', append=True, inplace=True)
      _df = _df.reorder_levels(['date', 'period', 'months'])

    df = df.combine_first(_df)

  mask = (months == 3) & (period.isin({'Q1', 'Q2', 'Q3', 'Q4'})) | (months == 12) & (
    period == 'FY'
  )

  # df.reset_index(level='months', inplace=True)
  # df.loc[df['months'] == 12,'months'] = 'a'
  # df.loc[df['months'] == 3,'months'] = 'q'
  # df.rename(columns={'months': 'scope'}, inplace=True)
  # df.set_index('scope', append=True, inplace=True)

  df = df.loc[mask, :]

  return df.copy()


def upsert_financials(
  db_name: str,
  table: str,
  financials: list[Financials],
):
  db_path = DB_DIR / sqlite_name(db_name)

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  cur.execute(
    f"""CREATE TABLE IF NOT EXISTS "{table}"(
    id TEXT PRIMARY KEY,
    scope TEXT,
    date TEXT,
    period TEXT,
    fiscal_end TEXT,
    currency JSON,
    data JSON
  )"""
  )

  query = f"""INSERT INTO 
    "{table}" VALUES (:id, :scope, :date, :period, :fiscal_end, :currency, :data)
    ON CONFLICT (id) DO UPDATE SET  
      data=json_patch(data, excluded.data),
      currency=(
        SELECT json_group_array(value)
        FROM (
          SELECT json_each.value
          FROM json_each(currency)
          WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded.currency))
        )
      )
  """
  cur.executemany(query, [f.model_dump() for f in financials])

  con.commit()
  con.close()


def select_financials(db_name: str, table: str) -> list[Financials]:
  db_path = DB_DIR / sqlite_name(db_name)

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()
    cur.row_factory = lambda cursor, row: Financials(**row)

    financials: list[Financials] = cur.execute(f'SELECT * FROM "{table}"').fetchall()

  return financials


def get_ciks() -> DataFrame[CikFrame]:
  rnm = {'cik_str': 'cik', 'title': 'name'}

  url = 'https://www.sec.gov/files/company_tickers.json'

  with httpx.Client() as s:
    rs = s.get(url, headers=HEADERS)
    parse: dict[str, CikEntry] = rs.json()

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return cast(DataFrame[CikFrame], df)


def all_gaap_items() -> set[str]:
  db_path = DB_DIR / 'edgar.json'
  db = TinyDB(db_path)

  items: set[str] = set()
  for t in db.tables():
    data = db.table(t).all()
    for i in data:
      items.update(i['data'].keys())

  return items
