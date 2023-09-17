import asyncio
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import re
from typing import Optional
import xml.etree.ElementTree as et

from glom import glom
import pandas as pd
import httpx

# Local
from lib.const import HEADERS
from lib.db.lite import read_sqlite
from lib.db.tiny import insert_tinydb, read_tinydb
from lib.edgar.models import Financials
from lib.edgar.parse import (
  fix_financials_df,
  xbrl_url,
  xbrl_urls, 
  parse_statements, 
  parse_taxonomy,
  statement_to_df
)
from lib.fin.utils import Taxonomy
from lib.utils import camel_split, snake_abbreviate

class Company():
  __slots__ = ('_cik')

  def __init__(self, cik: int):
    self._cik = cik

  def padded_cik(self):
    return str(self._cik).zfill(10)

  def filings(
    self, 
    forms: Optional[list[str]] = None,
    date: Optional[dt|str] = None,
    filter_xbrl: bool = False
  ) -> pd.DataFrame:
    
    def fetch(url: str):
      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS)
        parse = rs.json()

      return parse

    def json_to_df(filings: dict[str, list[str]]):
      data = {
        'id': filings['accessionNumber'],
        'date': filings['reportDate'],
        'form': filings['form'],
        'primary_document': filings['primaryDocument'],
        'is_XBRL': filings['isXBRL']
      }
      df = pd.DataFrame(data)
      df.set_index('id', inplace=True)
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
      return df
    
    dfs = []
        
    url = f'https://data.sec.gov/submissions/CIK{self.padded_cik()}.json'
    parse = fetch(url)
    filings = parse['filings']['recent']
    dfs.append(json_to_df(filings))

    if files := parse['filings'].get('files'):
      for f in files:
        url = f"https://data.sec.gov/submissions/{f['name']}"
        filings = fetch(url)
        dfs.append(json_to_df(filings))
    
    if len(dfs) == 1:
      df = dfs.pop()
        
    else:
      df = pd.concat(dfs)
      df.drop_duplicates(inplace=True)
      df.sort_values('date', ascending=True, inplace=True)

    if forms:
      df = df.loc[df['form'].isin(forms)]

    if date:
      if isinstance(date, str):
        date = dt.strptime(date, '%Y-%m-%d')
      
      df = df.loc[df['date'] > date]

    if filter_xbrl:
      df = df.loc[df['is_XBRL'].astype(bool)]
         
    return df
  
  def xbrls(self, date: Optional[dt|str] = None) -> list[str]:
    filings = self.filings(['10-K', '10-Q'], date, True)

    if filings['date'].max() < dt(2020, 7, 1):
      raise Exception('Not possible to find XBRL names')
    
    prefix ='https://www.sec.gov/Archives/edgar/data/'

    filings.sort_values('date', ascending=False, inplace=True)
    filings.reset_index(inplace=True)

    ticker: str = filings['primary_document'].iloc[0]
    ticker = ticker.split('-')[0]

    filings['xbrl'] = (
      prefix + str(self._cik) + '/' + filings['id'].str.replace('-', '') + '/'
    )
    mask = filings['date'] >= dt(2020, 7, 1)
    filings.loc[~mask, 'xbrl'] += (
      ticker + '-' + filings['date'].dt.strftime('%Y%m%d') + '.xml'
    )
    filings.loc[mask, 'xbrl'] += (
      filings.loc[mask, 'primary_document'].str.replace('.htm', '_htm.xml')
    )
    filings.set_index('id', inplace=True)
    return filings['xbrl']
  
  async def financials(self, delta=120) -> list[Financials]:
  
    async def fetch_urls(date: Optional[str|dt] = None) -> pd.Series:
      try:
        urls = self.xbrls(date)
      
      except Exception:
        filings = self.filings(['10-Q', '10-K'], date, True)
        urls = await xbrl_urls(self._cik, filings.index, 'htm')

      return urls
    
    # Load financials
    financials = read_tinydb('data/edgar.json', None, str(self._cik))

    if financials:
      dates = [glom(f, 'meta.date') for f in financials]
      dates = pd.to_datetime(dates, format='%Y-%m-%d')
      last_date = dates.max()
      
      if relativedelta(dt.now(), last_date).days < delta:
        return financials

      new_filings = await fetch_urls(last_date)

      if not new_filings:
        return financials

      old_filings = {glom(f, 'meta.id') for f in financials}
      new_filings = set(new_filings.index).difference(old_filings)

      if not new_filings:
        return financials
    else:
      new_filings = await fetch_urls()

    new_financials = await parse_statements(new_filings.tolist())
    if new_financials:
      insert_tinydb(new_financials, 'data/edgar.json', str(self._cik))
    
    return [*financials, *new_financials]

  async def financials_to_df(self,
    taxonomy: Optional[Taxonomy] = None,
  ) -> pd.DataFrame:
    #period = {'10-Q': 'q', '10-K': 'a'}
    
    financials = await self.financials()
    financials = sorted(financials, key=lambda x: glom(x, 'meta.date'), reverse=True)

    dfs = []

    for f in financials:
      dfs.append(statement_to_df(f))
    
    df = pd.concat(dfs, join='outer')
    df.sort_index(level=0, ascending=True, inplace=True)

    if taxonomy:
      df = fix_financials_df(df, taxonomy)

    return df

  def get_calc_template(self, doc_id):
    ns = 'http://www.w3.org/1999/xlink'

    url = xbrl_url(self._cik, doc_id)
    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS)
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
    
    result = await fetch()
    return result

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
    
    df['item'] = df['gaap'].astype(str).apply(
      lambda x: snake_abbreviate(x)
    )
    df['label'] = df['gaap'].astype(str).apply(
      lambda x: ' '.join(camel_split(x))
    )
    df['parent_'] = df['parent'].astype(str).apply(
      lambda x: snake_abbreviate(x) if x else ''
    )

    df = df[['sheet', 'item', 'parent_', 'parent', 'label', 'gaap']]
    df.to_csv('fin_taxonomy.csv', index=False)

'''
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

df['ebit'] = df['netInc'] + df['intEx'] + df['taxEx'] 
#df['ebit'] = df['rvn'] - df['rvnEx'] - df['opEx']
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

'''