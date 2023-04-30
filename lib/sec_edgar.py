import numpy as np
import pandas as pd

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import asyncio

# Web scrapping
import requests
import httpx
import bs4 as bs
import json
import xml.etree.ElementTree as et
from glom import glom

# Databasing
#from pymongo import MongoClient, DESCENDING
from tinydb import where

# Utils
import re

# Local
from lib.db import DB_DIR, upsert_sqlite, insert_tinydb, read_tinydb
#rom lib.finlib import finItemRenameDict

class Ticker():

  _headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'Sec-GPC': '1',
    'Cache-Control': 'max-age=0',
  }

  def __init__(self, cik: int):
    self._cik = str(cik).zfill(10)

  def filings(self, forms:list=[]):
      
    def json_to_df(dct):
      data = {
        'id': parse['accessionNumber'],
        'date': parse['reportDate'],
        'form': parse['form'],
        #'description': parse['primaryDescription']
      }
      df = pd.DataFrame(data)
      df.set_index('id', inplace=True)
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
      return df
    
    dfs = []
    
    try:
      url = f'https://data.sec.gov/submissions/CIK{self._cik}-submissions-001.json'
      with requests.Session() as s:
        rs = s.get(url, headers=self._headers)
        parse = json.loads(rs.text)
          
      dfs.append(json_to_df(parse)) 
    except:
      pass
    
    url = f'https://data.sec.gov/submissions/CIK{self._cik}.json'
    with requests.Session() as s:
      rs = s.get(url, headers=self._headers)
      parse = json.loads(rs.text)
    
    parse = parse['filings']['recent']
    dfs.append(json_to_df(parse))
    
    if len(dfs) == 1:
      df = dfs.pop()
        
    else:
      df = pd.concat(dfs)
      df.drop_duplicates(inplace=True)
      df.sort_values('date', ascending=True, inplace=True)

    if forms:
      mask = df['form'].isin(forms)
      df = df.loc[mask]
        
    return df

  def get_financials(self, delta=120):

    def new_financials(start_date:dt|str='') -> pd.DataFrame:
      docs = self.filings(['10-K', '10-Q'])

      if start_date:
        docs = docs.loc[docs['date'] > start_date]

      return docs

    # Load financials
    financials = read_tinydb('edgar.json', None, int(self._cik))

    if financials:
      dates = [glom(f, 'meta.date') for f in financials]
      dates = pd.to_datetime(dates, format='%Y-%m-%d')
      last_date = dates.max()
      
      if relativedelta(dt.now(), last_date).days > delta:
        new_forms = new_financials(last_date)

        if not new_forms:
          return financials

        old_docs = {glom(f, 'meta.id') for r in financials}
        new_docs = set(new_forms.index).difference(old_docs)

        if not new_docs:
          return financials
    else:
      new_docs = new_financials().index

    new_financials = []
    for d in new_docs:
      url = xbrl_url(self._cik, d)
      if not url:
        continue
      new_financials.append(xbrl_to_dict(url, self.cik))

    if new_financials:
      insert_tinydb(new_financials, 'edgar.json', int(self._cik))
    
    return [*financials, *new_financials]

  def financials(self):
    #period = {'10-Q': 'q', '10-K': 'a'}
    
    # MongoDB
    #client = MongoClient('mongodb://localhost:27017/')
    #db = client['finly']
    #coll = db['secFinancials']
    #query = {'meta.cik': self._cik}
    #record = next(docs)
    #lastDate = dt.strptime(record['meta']['date'], '%Y-%m-%d')
    #if relativedelta(dt.now(), lastDate).months > 3:
    #  self.scrapFinancials()
    #docs = coll.find(query, {'_id': False}).sort(
    #  'meta.date', DESCENDING)

    query = where('meta')['cik'] == self._cik
    df = read_tinydb('financials.json', query, 'edgar')

    dfs = []

    for x in docs:
      temp = financials_to_df(x)
      dfs.append(temp)
    
    df = dfs.pop(0)
    for i in dfs:
      df = df.combine_first(i)
      diffCols = i.columns.difference(df.columns).tolist()

      if diffCols:
        df = df.join(i[diffCols], how='outer')

    df.sort_index(level=0, ascending=True, inplace=True)

    # Convert multi-quarterly figures to quarterly ones
    excl = ['sh', 'shDil', 'taxRate']
    for p in range(2,5):
        
      # Extract multi-quarterly figures
      dfMq = df.loc[(slice(None), f'{p}q'), :].dropna(axis=1, how='all')
      dfMq = dfMq[dfMq.columns.difference(excl)]
      dfMq.reset_index('period', inplace=True)
      dfMq['period'] = 'q'
      dfMq.set_index('period', append=True, inplace=True)

      # Extract quarterly figures
      dates = dfMq.index.get_level_values('date')

      if p == 2:
          dfQ = df.loc[(slice(None), 'q'), dfMq.columns].shift(1)
      
      else:
        dfQ = df.loc[(slice(None), 'q'), dfMq.columns]\
          .rolling(p-1, min_periods=p-1).sum().shift(1)
      
      dfQ = dfQ.loc[(dates, slice(None)), :]

      # Calculate quarterly figures
      dfMq = dfMq - dfQ

      df.update(dfMq, overwrite=False) # Upsert
    
    if {'2q', '3q', '4q'}.intersection(set(df.index.get_level_values('period'))):
      df = df.loc[(slice(None), ['a', 'q']), :]
        
    # Additional items
    df['rvnEx'].fillna(df['rvn'] - df['grsPrft'], inplace=True)

    df['ebit'] = df['netInc'] + df['intEx'] + df['taxEx'] #df['ebit'] = df['rvn'] - df['rvnEx'] - df['opEx']
    df['ebitda'] = df['ebit'] + df['da']
    df['intCvg'] = (df['ebit'] / df['intEx']) # Interest coverage
    df['taxRate'] = df['taxEx'] / df['ebt'] # Tax rate

    df['cceStInv'].fillna(df['cce'] + df['stInv'], inplace=True)
    df['totNoCrtLbt'].fillna(df['totAst'] - df['totCrtAst'], inplace=True)
    df['totNoCrtLbt'].fillna(df['totLbt'] - df['totCrtLbt'], inplace=True)

    # Working capital
    for p in df.index.get_level_values('period').unique():
      msk = (slice(None), p)
      df.loc[msk, 'wrkCap'] = (
        df.loc[msk, 'totCrtAst'].rolling(2, min_periods=0).mean() -
        df.loc[msk, 'totCrtLbt'].rolling(2, min_periods=0).mean()
      )

      df.loc[msk, 'chgWrkCap'] = df.loc[msk, 'wrkCap'].diff()

    # Total debt
    df['totDbt'] = df['stDbt'] + df['ltDbt']

    #df['tgbEqt'] = (df['totEqt'] - df['prfEqt'] - df['itgbAst'] - df['gw'])

    # Capital expenditure
    for p in df.index.get_level_values('period').unique():
      msk = (slice(None), p)
      df.loc[msk, 'capEx'] = df.loc[msk, 'ppe'].diff() + df.loc[msk, 'da']

    # Free cash flow
    if 'freeCf' not in set(df.columns):
      df['freeCfFirm'] = (
        df['netInc'] + 
        df['da'] +
        df['intEx'] * (1 - df['taxRate']) - 
        df['chgWrkCap'] -
        df['capEx']
      )
      df['freeCf'] = (
        df['freeCfFirm'] + 
        df['totDbt'].diff() - 
        df['intEx'] * (1 - df['taxRate']))

    return df

  def get_calc_template(self, doc_id):
    ns = 'http://www.w3.org/1999/xlink'

    url = xbrl_url(self._cik, doc_id)
    with requests.Session() as s:
      rs = s.get(url, headers=HEADERS)
      root = et.fromstring(rs.content)

    url_pattern = r'https?://www\..+/'
    el_pattern = r'(?<=_)[A-Z][A-Za-z]+(?=_)'

    calc = dict()
    for sheet in root.findall('.//{*}calculationLink'):
      temp = dict()
      for el in sheet.findall('.//{*}calculationArc'):
        parent = re.search(el_pattern, el.attrib[f'{{{ns}}}from']).group()
        child = re.search(el_pattern, el.attrib[f'{{{ns}}}to']).group()
        
        if parent not in temp:
          temp[parent] = {}
        
        temp[parent].update({child: float(el.attrib['weight'])})
    
      label = re.sub(url_pattern, '', sheet.attrib[f'{{{ns}}}role'])
      calc[label] = temp

    return calc

  async def get_taxonomy(self):

    async def fetch():
      docs = self.filings(['10-K', '10-Q']).index
      tasks = []
      for doc in docs:
        url = await xbrl_url(self._cik, doc, 'cal')
        if not url:
          continue

        tasks.append(asyncio.create_task(taxonomy_to_df(url)))
      
      dfs = await asyncio.gather(*tasks)
      df = pd.concat(dfs)
      df = df.loc[~df.index.duplicated()]
      return df

    return asyncio.run(fetch())

def get_ciks():
  rnm = {'cik_str': 'cik', 'title': 'name'}

  headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
  }
      
  url = 'https://www.sec.gov/files/company_tickers.json'

  with requests.Session() as s:
    rs = s.get(url, headers=headers)
    parse = rs.json()

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return df

async def fetch_urls(cik:str, doc_ids:list, doc_type:str) -> list:
  tasks = [xbrl_url(cik, doc_id) for doc_id in doc_ids]
  result = await asyncio.gather(*tasks)
  return list(filter(None, result))

async def xbrl_url(cik, doc_id: str, doc_type='htm') -> str:
  url = f'https://www.sec.gov/Archives/edgar/data/{doc_id}/{doc_id}-index.htm'
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=Ticker._headers)
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

def xbrl_to_dict(url: str, cik: str) -> dict:

  with requests.Session() as s:
    rs = s.get(url, headers=Ticker._headers)
    root = et.fromstring(rs.content)

  form = {
    '10-K': 'annual',
    '10-Q': 'quarterly'
  }
  meta = {
    #'cik': int(cik),
    #'ticker': root.find('.{*}TradingSymbol').text,
    'id': url.split('/')[-2],
    'type': form[root.find('.{*}DocumentType').text],
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

    item_name = item.tag.split('}')[-1]
    if not item_name in data['data']:
      data['data'][item_name] = []
    
    temp = {}
    
    # Dates
    ctx = item.attrib['contextRef'] #item.get('contextRef')
    period = root.find(f'./{{*}}context[@id="{ctx}"]').find('./{*}period')
    
    if period.find('./{*}instant') is not None:
      temp['period'] = {
        'instant': period.find('./{*}instant').text
      }
    else:
      temp['period'] = {
        'start_date': period.find('./{*}startDate').text,
        'end_date': period.find('./{*}endDate').text,
      }
    # Segment
    seg = root.find(f'./{{*}}context[@id="{ctx}"]').find('.//{*}segment/{*}explicitMember')
    if seg is not None:
      temp['segment'] = seg.text
    
    # Numerical value
    temp['value'] = float(item.text)
    
    # Unit
    unit = item.attrib['unitRef']
    if '_' in unit:
      unit = unit.split('_')[-1].lower()
    temp['unit'] = unit
    
    # Append scrapping
    if temp not in data['data'][item_name]:
      data['data'][item_name].append(temp)

  # Sort items
  data['data'] = {
    k: data['data'][k] 
    for k in sorted(data['data'].keys())
  }
  return data

def financials_to_df(dct_raw):
	
  def parse_date(date):
    if isinstance(date, str):
      date = dt.strptime(i['period']['startDate'], '%Y-%m-%d')
    
    return date

  def insert_value(dct, col, val, end_date, period):
    if (end_date, period) not in df_data:
      dct[(end_date, period)] = {}
    
    if col not in df_data[(end_date, period)]:
      df_data[(end_date, period)][col] = val

    else:
      if not pd.isnull(df_data[(end_date, period)][col]):
        df_data[(end_date, period)][col] = val

    return dct

  fiscal_month = int(dct_raw['meta']['fiscal_end'].split('-')[1])

  df_data = {}

  for k, v in dct_raw['data'].items():
    for i in v:
      if ('segment' in i) and (not 'value' in i):
        continue
      
      if 'instant' in i['period']:
        end_date = parse_date(i['period']['instant'])

        if end_date.month == fiscal_month:
          for p in ['a', 'q']:
            df_data = insert_value(df_data, k, float(i['value']), end_date, p)

        else:
          df_data = insert_value(df_data, k, float(i['value']), end_date, 'q')
              
      else:
        start_date = parse_date(i['period']['startDate'])
        end_date = parse_Date(i['period']['endDate'])
        
        delta = relativedelta(end_date, start_date)
        cond_annual = (
          ((delta.months == 0 and delta.years == 1) or (delta.months > 10)) and 
          (end_date.month == fiscal_month)
        )
        if cond_annual:
          for p in ['a', '4q']:
            df_data = insert_value(df_data, k, float(i['value']), end_date, p)

        elif (delta.months < 4):
          df_data = insert_value(df_data, k, float(i['value']), end_date, 'q')

        elif (delta.months < 7):
          df_data = insert_value(df_data, k, float(i['value']), end_date, '2q')

        elif (delta.months < 10):
          df_data = insert_value(df_data, k, float(i['value']), end_date, '3q')

  # Construct dataframe
  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period']
  rnm = finItemRenameDict('GAAP')
  cols = set(rnm.keys()).intersection(set(df.columns))
  df = df[list(cols)]
  df.rename(columns=rnm, inplace=True)
  df.dropna(how='all', inplace=True)

  # Combine and remove duplicate columns
  temp = df.loc[:, df.columns.duplicated()]
  if not temp.empty:
    df = df.loc[:, ~df.columns.duplicated()]

    for c in temp:
      df[c].fillna(temp[c], inplace=True)

  return df

async def taxonomy_to_df(xml_url):
  ns = 'http://www.w3.org/1999/xlink'

  def rename_sheet(txt: str) -> str:
    pattern = r'income|balance|cashflow'
    m = re.search(pattern, txt, flags=re.I)
    if m:
      txt = m.group().lower()
    
    return txt

  async with httpx.AsyncClient() as client:
    rs = await client.get(xml_url, headers=Ticker._headers)
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

def process_taxonomy():
    df = read_sqlite('SELECT * FROM financials', 'taxonomy.db', 'item')
    if df is None:
        raise Exception('Taxonomy does not exist!')
    
    df.reset_index(inplace=True)
    df.rename(columns={'item': 'gaap'}, inplace=True)
    
    item = set(df['gaap'].unique())
    parent = set(df['parent'].unique())
    append = parent.difference(item)
    
    temp = df.loc[df['parent'].isin(append)]
    temp.loc[:,'gaap'] = temp['parent'].values
    temp.loc[:,'parent'] = ''
    
    df = pd.concat([df, temp])
    
    df['item'] = df['gaap'].astype(str).apply(lambda x: snake_abbreviate(x))
    df['label'] = df['gaap'].astype(str).apply(lambda x: ' '.join(camel_split(x)))
    df['parent_'] = df['parent'].astype(str).apply(lambda x: snake_abbreviate(x) if x else '')

    df = df[['sheet', 'item', 'parent_', 'parent', 'label', 'gaap']]
    df.to_csv('fin_taxonomy.csv', index=False)

def gaap_items():
  db_path = DB_DIR / 'edgar.json'
  db = TinyDB(db_path)
  
  items = set()
  for t in db.tables():
    data = db.table(t).all()
    for i in data:
      items.update(i['data'].keys())
          
  return items