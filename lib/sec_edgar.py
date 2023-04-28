import numpy as np
import pandas as pd

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

# Web scrapping
import requests
import bs4 as bs
import json
import xml.etree.cElementTree as et
from glom import glom

# Databasing
#from pymongo import MongoClient, DESCENDING
from tinydb import where

# Utils
import re

# Local
from lib.db import insert_tinydb, read_tinydb
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

  def filings(self):
      
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
        
    return df

  def get_filing(self, doc_id: str) -> dict:
    url = f'https://www.sec.gov/Archives/edgar/data/{doc_id}/{doc_id}-index.htm'
    with requests.Session() as s:
      rs = s.get(url, headers=Ticker._headers)
      parse = bs.BeautifulSoup(rs.text, 'lxml')

    pattern = r'(?<!_(cal|def|lab|pre)).xml$'
    href = parse.find('a', href=re.compile(pattern))
    if href is None:
      return None
    
    href = href.get('href')
    xml_url = f'https://www.sec.gov/{href}'

    return xml_to_dict(xml_url, self._cik)

  def get_financials(self, delta=90):

    def new_financials(start_date:dt|str='') -> pd.DataFrame:
      df = self.filings()
      mask = df['form'].isin(['10-K', '10-Q'])
      df = df.loc[mask]

      if start_date:
        df = df.loc[df['date'] > start_date]

      return df

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
      filing = self.get_filing(d)
      if not filing:
        continue
      new_financials.append(filing)

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

def xml_to_dict(url: str, cik: str) -> dict:

  with requests.Session() as s:
    rs = s.get(url, headers=Ticker._headers)
    root = et.fromstring(rs.content)

  form = {
    '10-K': 'annual',
    '10-Q': 'quarterly'
  }
  meta = {
    'cik': int(cik),
    'ticker': root.find('.{*}TradingSymbol').text,
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
        if isinstance(i['period']['instant'], str):
          end_date = dt.strptime(i['period']['instant'], '%Y-%m-%d')
        else:
          end_date = i['period']['instant']

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
    rc = s.get(url, headers=headers)
    parse = json.loads(rc.text)

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return df