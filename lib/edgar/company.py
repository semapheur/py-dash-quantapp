import asyncio
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import re
from typing import cast, Literal, Optional, TypedDict
import xml.etree.ElementTree as et

import pandas as pd
import pandera as pa
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Index, Series, DateTime
from pydantic import BaseModel
import httpx

# Local
from lib.const import HEADERS
from lib.db.lite import read_sqlite
from lib.edgar.models import Financials, Filings, Recent, StockSplit, FinData
from lib.edgar.parse import (
  fix_financials_df,
  get_stock_splits,
  upsert_financials,
  xbrl_url,
  xbrl_urls,
  parse_statements,
  parse_taxonomy,
  statement_to_df,
)
from lib.fin.calculation import stock_split_adjust
from lib.utils import camel_split, snake_abbreviate

FIELDS = {
  'id': 'TEXT',
  'date': 'TEXT',
  'scope': 'TEXT',
  'period': 'TEXT',
  'fiscal_end': 'TEXT',
  'currency': 'JSON',
  'data': 'JSON',
}


class FilingsJSON(BaseModel):
  accessionNumber: list[str]
  filingDate: list[str]
  reportDate: list[str]
  acceptanceDateTime: list[str]
  act: list[str]
  form: list[str]
  fileNumber: list[str]
  filmNumber: list[str]
  items: list[str]
  size: list[int]
  isXBRL: list[Literal[0, 1]]
  isInlineXBRL: list[Literal[0, 1]]
  primaryDocument: list[str]
  primaryDocDescription: list[str]


class FilingsFrame(pa.DataFrameModel):
  id: Index[str]
  date: Series[DateTime]
  form: Series[str]
  primary_document: Series[str]
  is_XBRL: Series[Literal[0, 1]]


class Company:
  __slots__ = '_cik'

  def __init__(self, cik: int):
    self._cik = cik

  def padded_cik(self):
    return str(self._cik).zfill(10)

  def filings(
    self,
    forms: Optional[list[str]] = None,
    date: Optional[dt] = None,
    filter_xbrl: bool = False,
  ) -> DataFrame[FilingsFrame]:
    def fetch(url: str) -> dict:
      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS)
        parse: dict = rs.json()

      return parse

    def json_to_df(filings: Recent) -> DataFrame[FilingsFrame]:
      data = {
        'id': filings.get('accessionNumber'),
        'date': filings.get('reportDate'),
        'form': filings.get('form'),
        'primary_document': filings.get('primaryDocument'),
        'is_XBRL': filings.get('isXBRL'),
      }
      df = pd.DataFrame(data)
      df.set_index('id', inplace=True)
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
      return cast(DataFrame[FilingsFrame], df)

    dfs: list[DataFrame[FilingsFrame]] = []

    url = f'https://data.sec.gov/submissions/CIK{self.padded_cik()}.json'
    parse = fetch(url)
    if (filings := cast(Filings | None, parse.get('filings'))) is None:
      raise ValueError(f'JSON has no filings: {parse}')

    recent = filings['recent']
    dfs.append(json_to_df(recent))

    if (files := filings.get('files')) is not None:
      for f in files:
        url = f'https://data.sec.gov/submissions/{f.get("name")}'
        recent = cast(Recent, fetch(url))
        dfs.append(json_to_df(recent))

    if len(dfs) == 1:
      df = dfs.pop()

    else:
      df = cast(DataFrame[FilingsFrame], pd.concat(dfs))
      df.drop_duplicates(inplace=True)
      df.sort_values('date', ascending=True, inplace=True)

    if forms:
      df = cast(DataFrame[FilingsFrame], df.loc[df['form'].isin(forms)])

    if date:
      df = cast(DataFrame[FilingsFrame], df.loc[df['date'] > date])

    if filter_xbrl:
      df = cast(DataFrame[FilingsFrame], df.loc[df['is_XBRL'].astype(bool)])

    return df

  def xbrls(self, date: Optional[dt] = None) -> Series[str]:
    filings = self.filings(['10-K', '10-Q'], date, True)

    if filings['date'].max() < dt(2020, 7, 1):
      raise Exception('Not possible to find XBRL names')

    prefix = 'https://www.sec.gov/Archives/edgar/data/'

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
    filings.loc[mask, 'xbrl'] += filings.loc[mask, 'primary_document'].str.replace(
      '.htm', '_htm.xml'
    )
    filings.set_index('id', inplace=True)
    return cast(Series[str], filings['xbrl'])

  def stock_splits(self) -> Series[float]:
    field = 'StockholdersEquityNoteStockSplitConversionRatio1'
    query = (
      f'SELECT data FROM {self._cik} WHERE json_extract(data, "$.{field}") IS NOT NULL'
    )
    df_parse = cast(
      DataFrame[str], read_sqlite('financials_scrap', query, dtype={'data': str})
    )
    fin_data = cast(list[FinData], df_parse['data'].apply(json.loads).to_list())

    df_data: list[StockSplit] = []
    for data in fin_data:
      df_data.extend(get_stock_splits(data))

    df = pd.DataFrame(df_data)
    df.drop_duplicates(inplace=True)
    df.set_index('date', inplace=True)

    return cast(Series[float], df['stock_split_ratio'])

  async def financials(self, delta=120, date: Optional[str] = None):
    async def fetch_urls(date: Optional[dt] = None) -> pd.Series:
      try:
        urls = self.xbrls(date)

      except Exception:
        filings = self.filings(['10-Q', '10-K'], date, True)
        urls = await xbrl_urls(self._cik, filings.index.to_list(), 'htm')

      return urls

    # def df_to_financials(df: DataFrame) -> list[Financials]:
    #  df.loc['date'] = df.loc[:, 'date'].apply(pd.to_datetime)
    #  df.loc['currency'] = df.loc[:, 'currency'].apply(json.loads)
    #  df.loc['data'] = df.loc[:, 'data'].apply(json.loads)
    #  return [Financials(**row) for row in df.to_dict('records')]

    # Load financials
    query = f'SELECT * FROM "{self._cik}" ORDER BY date ASC'
    if date:
      query += f' WHERE date >= {dt.strptime(date, "%Y-%m-%d")}'

    financials = read_sqlite('financials_scrap.db', query, parse_dates=True)

    if not financials.empty:
      last_date = financials['date'].max()

      if relativedelta(dt.now(), last_date).days < delta:
        return financials

      new_filings = await fetch_urls(last_date)

      if not new_filings:
        return financials

      old_filings = set(financials['id'])
      filings_diff = set(new_filings.index).difference(old_filings)

      if not filings_diff:
        return financials
    else:
      new_filings = await fetch_urls()

    new_financials = await parse_statements(new_filings.tolist())
    if new_financials:
      upsert_financials('financials_scrap.db', str(self._cik), new_financials)

    return new_financials

  async def financials_to_df(self) -> pd.DataFrame:
    financials = await self.financials()

    dfs = []

    for f in financials:
      dfs.append(statement_to_df(f))

    df = pd.concat(dfs, join='outer')
    df.sort_index(level=0, ascending=True, inplace=True)
    df = fix_financials_df(df)

    ratios = self.stock_splits()
    if ratios is not None:
      df = stock_split_adjust(df, ratios)

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
  query = 'SELECT * FROM financials'
  df = read_sqlite('taxonomy.db', query, index_col='item')
  if df is None:
    raise Exception('Taxonomy does not exist!')

  df.reset_index(inplace=True)
  df.rename(columns={'item': 'gaap'}, inplace=True)

  item = set(df['gaap'].values)
  parent = set(df['parent'].values)
  append = parent.difference(item)

  temp = df.loc[df['parent'].isin(append)]
  temp.loc[:, 'gaap'] = temp['parent'].values
  temp.loc[:, 'parent'] = ''

  df = pd.concat([df, temp])

  df['item'] = df['gaap'].astype(str).apply(lambda x: snake_abbreviate(x))
  df['label'] = df['gaap'].astype(str).apply(lambda x: ' '.join(camel_split(x)))
  df['parent_'] = (
    df['parent'].astype(str).apply(lambda x: snake_abbreviate(x) if x else '')
  )

  df = df[['sheet', 'item', 'parent_', 'parent', 'label', 'gaap']]
  df.to_csv('fin_taxonomy.csv', index=False)


"""
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

"""
