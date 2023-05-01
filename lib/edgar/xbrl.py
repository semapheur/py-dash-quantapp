import re
import xml.etree.ElementTree as et

import httpx
import bs4 as bs

import pandas as pd

# Local
from lib.db import DB_DIR, upsert_sqlite, insert_tinydb, read_tinydb
from lib.scrap import HEADERS
#rom lib.finlib import finItemRenameDict

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

  href = parse.find('a', href=re.compile(pattern))
  if href is None:
    return None
  
  href = href.get('href')
  return f'https://www.sec.gov/{href}'

async def parse_xbrl(url: str, cik: str):

  def parse_period(period: et.Element) -> dict[str, str]:
    if period.find('./{*}instant') is not None:
      return {
        'instant': period.find('./{*}instant').text
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