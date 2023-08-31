from datetime import datetime as dt
import re
import xml.etree.ElementTree as et

import asyncio
import httpx
import bs4 as bs
from glom import glom

import pandas as pd
from tinydb import TinyDB

# Local
from lib.const import DB_DIR, HEADERS
from lib.edgar.models import (
  Financials, 
  Instant, 
  Interval, 
  Item, 
  Member, 
  Meta
)

async def xbrl_urls(cik: int, doc_ids: list[str], doc_type:str) -> list[str]:
  tasks = [asyncio.create_task(xbrl_url(cik, doc_id, doc_type)) for doc_id in doc_ids]
  result = await asyncio.gather(*tasks)
  return list(filter(None, result))

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

async def parse_statement(url: str) -> Financials:

  def parse_period(period: et.Element) -> Instant|Interval:
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

  def parse_member(item: et.Element, segment: et.Element) -> Member:
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
  
  print(url)
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    root = et.fromstring(rs.content)

  form = {
    '10-K': 'annual',
    '10-Q': 'quarterly'
  }
  meta: Meta = {
    'id': url.split('/')[-2],
    'scope': form[root.find('.{*}DocumentType').text],
    'date': root.find('.{*}DocumentPeriodEndDate').text,
    'fiscal_end': root.find('.{*}CurrentFiscalYearEndDate').text[1:]
  }
  data: Financials = {
    'meta': meta,
    'data': {}
  }

  for item in root.findall('.//*[@unitRef]'):
    if item.text is None:
      continue
    
    temp: Item = {}
    
    # Period
    ctx = item.attrib['contextRef']
    period_el = root.find(f'./{{*}}context[@id="{ctx}"]').find('./{*}period')
    temp['period'] = parse_period(period_el)

    # Segment
    seg = root \
      .find(f'./{{*}}context[@id="{ctx}"]') \
      .find('.//{*}segment/{*}explicitMember')
    
    if seg is not None:
      temp['member'] = parse_member(item, seg)
    else:    
      # Numerical value
      temp['value'] = float(item.text)
      
      # Unit
      temp['unit'] = parse_unit(item.attrib['unitRef'])
    
    # Append scrapping
    item_name = item.tag.split('}')[-1]
    if item_name not in data['data']:
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
    except Exception:
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

def statement_to_df(data: Financials) -> pd.DataFrame:
    
  def parse_period(period: dict[str, str]) -> dt:
    if 'instant' in period:
      return dt.strptime(period['instant'], '%Y-%m-%d')
    
    return dt.strptime(period['end_date'], '%Y-%m-%d')
      
  fin_date = dt.strptime(glom(data, 'meta.date'), '%Y-%m-%d')
  scope = glom(data, 'meta.scope')
  
  df_data: dict[tuple[dt, str], dict[str, float]] = {
    (fin_date, scope[0]): {}
  }
  
  for item in data['data'].keys():
    entries: list = glom(data, f'data.{item}')
    
    for entry in entries:
      
      date = parse_period(entry['period']) 
      if date != fin_date:
        continue
      
      if value := entry.get('value'):
        df_data[(date, scope[0])][item] = value
      
      if 'member' not in entry:
        continue
      
      for member in glom(data, f'data.{item}.member'):
        if value := member.get('value'):
          df_data[(date, scope[0])][f'{item}.{member}'] = value
      
  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period']
  return df

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