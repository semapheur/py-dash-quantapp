import asyncio
from datetime import datetime as dt
from enum import Enum
from functools import partial
import re
import xml.etree.ElementTree as et

import aiometer
import httpx
import bs4 as bs
from glom import glom
import pandas as pd
from tinydb import TinyDB

# Local
from lib.const import DB_DIR, HEADERS
from lib.db.lite import read_sqlite
from lib.edgar.models import (
  Financials, 
  Instant, 
  Interval, 
  Item, 
  Member, 
  Meta
)
from lib.utils import (
  combine_duplicate_columns,
  df_time_difference, 
  insert_characters, 
  month_difference, 
  fiscal_quarter
)

class Scope(Enum):
  QUARTERLY = 3
  ANNUAL = 12

async def xbrl_urls(cik: int, doc_ids: list[str], doc_type:str) -> pd.Series:
  tasks = [asyncio.create_task(xbrl_url(cik, doc_id, doc_type)) for doc_id in doc_ids]
  urls = await asyncio.gather(*tasks)

  result = pd.Series(urls, index=doc_ids)
  result = result.loc[result.notnull()]

  return result 

async def xbrl_url(cik: int, doc_id: str, doc_type: str = 'htm') -> str:
  url = (
    'https://www.sec.gov/Archives/edgar/data/'
    f"{cik}/{doc_id.replace('-', '')}/{doc_id}-index.htm"
  )
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    parse = bs.BeautifulSoup(rs.text, 'lxml')

  if doc_type == 'htm':
    pattern = r'(?<!_(cal|def|lab|pre)).xml$'
  elif doc_type in ['cal', 'def', 'lab', 'pre']:
    pattern = rf'_{doc_type}.xml$'
  else:
    raise Exception('Invalid document type!')

  a_node = parse.find('a', href=re.compile(pattern))
  if a_node is None:
    return None
  
  href = a_node.get('href')
  return f'https://www.sec.gov{href}'

async def parse_statements(urls: list[str]) -> list[Financials]:
  tasks = [partial(parse_statement, url) for url in urls]
  financials = await aiometer.run_all(tasks, max_per_second=10)

  return financials

async def parse_statement(url: str) -> Financials:

  def parse_period(period: et.Element) -> Instant|Interval:
    if (el := period.find('./{*}instant')) is not None:
      return {'instant': dt.strptime(el.text, '%Y-%m-%d')}

    start_date = dt.strptime(period.find('./{*}startDate').text, '%Y-%m-%d')
    end_date = dt.strptime(period.find('./{*}endDate').text, '%Y-%m-%d')
    months = month_difference(start_date, end_date)

    return {
      'start_date': start_date ,
      'end_date': end_date,
      'months': months
    }

  def parse_unit(unit: str) -> str:
    if '_' not in unit:
      return unit
    
    return unit.split('_')[-1].lower()

  def parse_member(item: et.Element, segment: et.Element) -> Member:
    def parse_name(name: str) -> str|None:
      name = re.sub(r'(Segment)?Member', '', name)
      return name.split(':')[-1]
    
    return {
      parse_name(segment.text): {
        'dim': segment.attrib['dimension'].split(':')[-1],
        'value': float(item.text),
        'unit': parse_unit(item.attrib['unitRef'])
      }
    }
  
  async def fetch(url: str) -> et.Element:
    async with httpx.AsyncClient() as client:
      rs = await client.get(url, headers=HEADERS)
      return et.fromstring(rs.content)
  
  root = await fetch(url)
  if root.tag == 'Error':
    cik, doc_id = url.split('/')[6:8]
    doc_id = insert_characters(doc_id, {'-': [10, 12]})
    url = await xbrl_url(cik, doc_id)
    root = await fetch(url)

  form = {
    '10-K': 'annual',
    '10-Q': 'quarterly'
  }

  scope = form[root.find('.{*}DocumentType').text]
  date = dt.strptime(root.find('.{*}DocumentPeriodEndDate').text, '%Y-%m-%d')
  fiscal_end = root.find('.{*}CurrentFiscalYearEndDate').text[1:]

  if (el := root.find('.{*}DocumentFiscalPeriodFocus')) is not None:
    fiscal_period = el.text
  elif scope == 'annual':
    fiscal_period = 'FY'
  else:
    pattern = r'(\d{2})-(\d{2})'

    match = re.search(pattern, fiscal_end)
    month = int(match.group(1))
    day = int(match.group(2))

    fiscal_period = fiscal_quarter(dt.strptime(date, '%Y-%m-%d'), month, day)

  meta: Meta = {
    'id': url.split('/')[-2],
    'date': date,
    'scope': scope,
    'period': fiscal_period,
    'fiscal_end': fiscal_end
  }
  data: Financials = {
    'meta': meta,
    'data': {}
  }

  for item in root.findall('.//*[@unitRef]'):
    if item.text is None:
      continue
    
    scrap: Item = {}
    
    ctx = item.attrib['contextRef']
    period_el = root.find(f'./{{*}}context[@id="{ctx}"]').find('./{*}period')
    
    scrap['period'] = parse_period(period_el)
   
    segment = root \
      .find(f'./{{*}}context[@id="{ctx}"]') \
      .find('.//{*}segment/{*}explicitMember')
    
    if segment is not None:
      scrap['member'] = parse_member(item, segment)
    else:    
      scrap['value'] = float(item.text)
      scrap['unit'] = parse_unit(item.attrib['unitRef'])
    
    item_name = item.tag.split('}')[-1]
    if item_name not in data['data']:
      data['data'][item_name] = [scrap]
      continue

    try:
      entry: dict = next(i for i in data['data'][item_name]
        if i['period'] == scrap['period']
      )
      if 'member' in scrap:
        entry.setdefault('member', {}).update(scrap['member'])
      else:
        entry.update(scrap)
    except Exception:
      data['data'][item_name].append(scrap)    

  # Sort items
  data['data'] = {
    k: data['data'][k] 
    for k in sorted(data['data'].keys())
  }
  return data

async def parse_taxonomy(url: str) -> pd.DataFrame:
  namespace = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink'
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
      taxonomy.append({
        'sheet': sheet_label,
        'gaap': re.search(
          el_pattern, 
          el.attrib[f'{{{namespace["xlink"]}}}to']).group(),
        'parent': re.search(
          el_pattern, 
          el.attrib[f'{{{namespace["xlink"]}}}from']).group(),
      })
  
  df = pd.DataFrame.from_records(taxonomy)
  df.set_index('item', inplace=True)
  df.drop_duplicates(inplace=True)
  return df

def statement_to_df(financials: Financials) -> pd.DataFrame:
    
  def parse_date(period: dict[str, str]) -> tuple[int, dt]:
    date = period.get('instant', period.get('end_date'))
    return dt.strptime(date, '%Y-%m-%d')
        
  fin_date = dt.strptime(glom(financials, 'meta.date'), '%Y-%m-%d')
  fin_scope = glom(financials, 'meta.scope')
  fin_period = glom(financials, 'meta.period')
  
  df_data: dict[tuple(dt, str), dict[str, int]] = {}

  for item, entries in financials['data'].items():
    for entry in entries:
      date = parse_date(entry['period'])
      if date != fin_date:
        continue

      months = entry['period'].get('months', Scope[fin_scope.upper()].value)
      period = fin_period
      if fin_period == 'FY' and months < 12:
        period = 'Q4'
      
      if value := entry.get('value'):
        df_data.setdefault((fin_date, period, months), {})[item] = value

        cond = (fin_period == 'FY' and (
          'instant' in entry['period'] or
          entry.get('unit') == 'shares'
        ))
        if cond:
          df_data.setdefault((fin_date, 'Q4', 3), {})[item] = value
      
      if not (members := entry.get('member')):
        continue

      for member, m_entry in members.items():
        if not (m_value := m_entry.get('value')):
          continue

        dim = '.' + d if (d := m_entry.get('dim')) else ''
        key = f'{item}{dim}.{member}'
        df_data.setdefault((fin_date, period, months), {})[key] = m_value

        cond = (fin_period == 'FY' and (
          'instant' in entry['period'] or
          m_entry.get('unit') == 'shares'
        ))
        if cond:
          df_data.setdefault((fin_date, 'Q4', 3), {})[key] = m_value
  
  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period', 'months']
  return df

def get_stock_splits(financials: Financials) -> list[dict]:

  name = 'StockholdersEquityNoteStockSplitConversionRatio1'
  splits = glom(financials, f'data.{name}')
  if splits is None:
    return
  
  data: list[dict] = []
  for entry in splits:
    value = entry.get('value')
    if value is None:
      continue
    
    data.append({
      'date': dt.strptime(
        glom(entry, 'period.start_date'), '%Y-%m-%d'),
      'stock_split_ratio': value
    })
  
  return data

def fix_financials_df(df: pd.DataFrame) -> pd.DataFrame:
  
  query = '''
    SELECT json_each.value AS gaap, item FROM items 
    JOIN JSON_EACH(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  '''
  items = read_sqlite('taxonomy.db', query)
  
  df = df[list(set(df.columns).intersection(set(items['gaap'])))]

  rename = {k: v for k, v in zip(items['gaap'], items['item'])}
  df.rename(columns=rename, inplace=True)
  df = combine_duplicate_columns(df)

  query = 'SELECT item FROM items WHERE period = "duration"'
  diff_items = read_sqlite('taxonomy.db', query)
  diff_items = list(set(diff_items['item']).intersection(set(df.columns)))
  
  conditions = (
    ('Q1', 3),
    ('Q2', 6),
    ('Q3', 9),
    ('FY', 12)
  )
  
  period = df.index.get_level_values('period')
  months = df.index.get_level_values('months')
  for i in range(1, len(conditions)):

    mask = (
      (period == conditions[i-1][0]) & (months == conditions[i-1][1]) |
      (period == conditions[i][0]) & (months == conditions[i][1])
    )
    _df = df.loc[mask, diff_items] 
    _df.sort_index(level='date', inplace=True)

    _df.loc[:, 'month_diff'] = df_time_difference(
      _df.index.get_level_values('date'), 30, 'D'
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

  mask = (
    (months == 3) & (period.isin({'Q1', 'Q2', 'Q3', 'Q4'})) |
    (months == 12) & (period == 'FY')
  )

  #df.reset_index(level='months', inplace=True)
  #df.loc[df['months'] == 12,'months'] = 'a'
  #df.loc[df['months'] == 3,'months'] = 'q'
  #df.rename(columns={'months': 'scope'}, inplace=True)
  #df.set_index('scope', append=True, inplace=True)

  df = df.loc[mask, :]
  
  return df.copy()

def get_ciks():
  rnm = {'cik_str': 'cik', 'title': 'name'}
      
  url = 'https://www.sec.gov/files/company_tickers.json'

  with httpx.Client() as s:
    rs = s.get(url, headers=HEADERS)
    parse = rs.json()

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return df

def all_gaap_items() -> set[str]:
  db_path = DB_DIR / 'edgar.json'
  db = TinyDB(db_path)
  
  items: set[str] = set()
  for t in db.tables():
    data = db.table(t).all()
    for i in data:
      items.update(i['data'].keys())
          
  return items