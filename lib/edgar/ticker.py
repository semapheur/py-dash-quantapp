import numpy as np
import pandas as pd

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import asyncio

# Web scrapping
import requests

import json
import xml.etree.ElementTree as et
from glom import glom

# Databasing
#from pymongo import MongoClient, DESCENDING
from tinydb import where

# Utils
import re
from dataclasses import dataclass

# Local
from lib.db.lite import upsert_sqlite
from lib.db.tiny import insert_tinydb, read_tinydb
from lib.edgar.parse import fetch_urls, xbrl_url, parse_xbrl, parse_taxonomy
from lib.const import HEADERS

class Ticker():

  def __init__(self, cik:int):
    self._cik = cik

  def padded_cik(self):
    return str(self._cik).zfill(10)

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
      url = f'https://data.sec.gov/submissions/CIK{self.padded_cik()}-submissions-001.json'
      with requests.Session() as s:
        rs = s.get(url, headers=HEADERS)
        parse = json.loads(rs.text)
          
      dfs.append(json_to_df(parse)) 
    except:
      pass
    
    url = f'https://data.sec.gov/submissions/CIK{self.padded_cik()}.json'
    with requests.Session() as s:
      rs = s.get(url, headers=HEADERS)
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

  async def get_financials(self, delta=120):

    def new_financials(start_date:dt|str='') -> pd.DataFrame:
      docs = self.filings(['10-K', '10-Q'])

      if start_date:
        docs = docs.loc[docs['date'] > start_date]

      return docs

    async def fetch(docs):
      tasks = []
      for d in docs:
        url = await xbrl_url(self.padded_cik(), d)
        if not url:
          continue
        tasks.append(asyncio.create_task(parse_xbrl(url, self.padded_cik())))
      
      financials = await asyncio.gather(*tasks)
      return financials

    # Load financials
    financials = read_tinydb('edgar.json', None, str(self._cik))

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

    new_financials = asyncio.run(fetch(new_docs))
    if new_financials:
      insert_tinydb(new_financials, 'edgar.json', str(self._cik))
    
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
        tasks.append(asyncio.create_task(parse_taxonomy(url)))
      
      dfs = await asyncio.gather(*tasks)
      df = pd.concat(dfs)
      df = df.loc[~df.index.duplicated()]
      return df

    return asyncio.run(fetch())

def process_taxonomy():
    df = read_sqlite('SELECT * FROM financials', 'taxonomy.db', 'item')
    if df is None:
        raise Exception('Taxonomy does not exist!')
    
    df.reset_index(inplace=True)
    df.rename(columns={'item': 'gaap'}, inplace=True)
    
    item = set(df['gaap'].values)
    parent = set(df['parent'].values)
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