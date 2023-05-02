from datetime import datetime as dt
import re
import xml.etree.ElementTree as et

import httpx
import bs4 as bs
from glom import glom

import pandas as pd
from tinydb import TinyDB

# Local
from lib.const import DB_DIR, HEADERS
from lib.db.lite import upsert_sqlite
from lib.db.tiny import insert_tinydb, read_tinydb

async def fetch_urls(cik:str, doc_ids:list, doc_type:str) -> list:
  tasks = [xbrl_url(cik, doc_id) for doc_id in doc_ids]
  result = await asyncio.gather(*tasks)
  return list(filter(None, result))

async def xbrl_url(cik, doc_id: str, doc_type='htm') -> str:
  url = f'https://www.sec.gov/Archives/edgar/data/{doc_id}/{doc_id}-index.htm'
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
  return f'https://www.sec.gov/{href}'

async def parse_statement(url: str, cik: str):

  def parse_period(period: et.Element) -> dict[str, str]:
    if (el := period.find('./{*}instant')) is not None:
      return {
        'instant': el.text
      }
    else: 
      return {
        'start_date': period.find('./{*}startDate').text,
        'end_date': period.find('./{*}endDate').text,
      }

  def parse_unit(text:str) -> str:
    if '_' in text:
      text = text.split('_')[-1].lower()
    return text

  def parse_member(item, segment) -> dict:
    def name(text:str) -> str|None:
      text = re.sub(r'(Segment)?Member', '', text)
      return text.split(':')[-1]
    
    return {
      name(segment.text): {
        'dim': segment.attrib['dimension'].split(':')[-1],
        'value': float(item.text),
        'unit': parse_unit(item.attrib['unitRef'])
      }
    }
  
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    root = et.fromstring(rs.content)

  form = {
    '10-K': 'annual',
    '10-Q': 'quarterly'
  }
  meta = {
    #'cik': int(cik),
    #'ticker': root.find('.{*}TradingSymbol').text,
    'id': url.split('/')[-2],
    'scope': form[root.find('.{*}DocumentType').text],
    'date': root.find('.{*}DocumentPeriodEndDate').text,
    'fiscal_end': root.find('.{*}CurrentFiscalYearEndDate').text[1:]
  }
  data = {
    'meta': meta,
    'data': {}
  }

  for item in root.findall('.//*[@unitRef]'):
    if item.text is None:
      continue
    
    temp = {}
    
    # Period
    ctx = item.attrib['contextRef']
    period_el = root.find(f'./{{*}}context[@id="{ctx}"]').find('./{*}period')
    temp['period'] = parse_period(period_el)

    # Segment
    seg = root.find(f'./{{*}}context[@id="{ctx}"]').find('.//{*}segment/{*}explicitMember')
    if seg is not None:
      temp['member'] = parse_member(item, seg)
    else:    
      # Numerical value
      temp['value'] = float(item.text)
      
      # Unit
      temp['unit'] = parse_unit(item.attrib['unitRef'])
    
    # Append scrapping
    item_name = item.tag.split('}')[-1]
    if not item_name in data['data']:
      data['data'][item_name] = [temp]
      continue

    try:
      entry = next(i for i in data['data'][item_name]
        if i['period'] == temp['period']
      )
      if 'member' in temp:
        if 'member' in entry:
          entry['member'].update(temp['member'])
        else:
          entry['member'] = temp['member']
      else:
        entry.update(temp)
    except:
      data['data'][item_name].append(temp)    

  # Sort items
  data['data'] = {
    k: data['data'][k] 
    for k in sorted(data['data'].keys())
  }
  return data

async def parse_taxonomy(url: str) -> pd.DataFrame:
  ns = 'http://www.w3.org/1999/xlink'

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
  for sheet in root.findall('.//{*}calculationLink'):
    sheet_label = re.sub(url_pattern, '', sheet.attrib[f'{{{ns}}}role'])
    sheet_label = rename_sheet(sheet_label)

    for el in sheet.findall('.//{*}calculationArc'):
      taxonomy.append({
        'sheet': sheet_label,
        'gaap': re.search(el_pattern, el.attrib[f'{{{ns}}}to']).group(),
        'parent': re.search(el_pattern, el.attrib[f'{{{ns}}}from']).group(),
      })
  
  df = pd.DataFrame.from_records(taxonomy)
  df.set_index('item', inplace=True)
  df.drop_duplicates(inplace=True)
  return df

def load_fin_items(source: str) -> pd.DataFrame:
  df = pd.read_csv('fin_items.csv', usecols=[source, 'member', 'item'])
  df.dropna(subset=source, inplace=True)
  df.set_index(source, inplace=True)
  return df

def statement_to_df(data: dict) -> pd.DataFrame:
    
  def parse_period(period: dict[str, str]) -> dt:
    if 'instant' in period:
      return dt.strptime(period['instant'], '%Y-%m-%d')
    
    return dt.strptime(period['end_date'], '%Y-%m-%d')
    
  def insert_value(df_data:dict, col:str, val:float, date:dt, scope:str):
    if (date, scope) not in df_data:
      df_data[(date, scope)] = {}
        
    df_data[(date, scope)][col] = val
  
  fin_items = load_fin_items('gaap')
  mask = fin_items['member'].isna()
  fin_date = dt.strptime(glom(data, 'meta.date'), '%Y-%m-%d')
  scope = glom(data, 'meta.scope')
  
  df_data = {
    (fin_date, scope[0]): {}
  }
  
  _fin_items = set(fin_items.index).intersection(set(data['data'].keys()))
  for i in _fin_items:
    entries: list = glom(data, f'data.{i}')
    
    for e in entries:
      if 'value' not in e:
        continue

      date = parse_period(e['period']) 
      if date != fin_date:
        continue
      
      col = fin_items.loc[mask, 'item'].loc[i]
      df_data[(date, scope[0])][col] = e['value']
      
      if 'member' not in e:
        continue

      mem = fin_items.loc[i,'member']
      if isinstance(mem, float): 
        continue
        
      _mem = set(mem).intersection(set(e['member'].keys()))
      if not _mem:
        continue
            
      for m in _mem:
        col = fin_items.loc[
          (fin_items.index == i) & 
          (fin_items['member'] == m), 
          'item'
        ].loc[i]
        df_data[(date, scope[0])][col] = e['value']
      
  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period']
  return df

def get_ciks():
  rnm = {'cik_str': 'cik', 'title': 'name'}
      
  url = 'https://www.sec.gov/files/company_tickers.json'

  with requests.Session() as s:
    rs = s.get(url, headers=HEADERS)
    parse = rs.json()

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return df

def gaap_items() -> set[str]:
  db_path = DB_DIR / 'edgar.json'
  db = TinyDB(db_path)
  
  items: set[str] = set()
  for t in db.tables():
    data = db.table(t).all()
    for i in data:
      items.update(i['data'].keys())
          
  return items