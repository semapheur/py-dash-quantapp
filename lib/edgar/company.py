import asyncio
from dataclasses import dataclass
from datetime import datetime as dt
import re
from typing import cast, Literal, Optional
import xml.etree.ElementTree as et

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series
from pandera.dtypes import Timestamp
from pydantic import BaseModel
import httpx

# Local
from lib.const import HEADERS
from lib.db.lite import read_sqlite
from lib.edgar.models import (
  RawFinancials,
  Filings,
  Recent,
)
from lib.edgar.parse import (
  xbrl_url,
  xbrl_urls,
  parse_statements,
  parse_taxonomy,
)

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
  date: Series[Timestamp]
  form: Series[str]
  primary_document: Series[str]
  is_XBRL: Series[Literal[0, 1]]


@dataclass(slots=True)
class Company:
  cik: int

  def padded_cik(self):
    return str(self.cik).zfill(10)

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
      prefix + str(self.cik) + '/' + filings['id'].str.replace('-', '') + '/'
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

  async def xbrl_urls(self, date: Optional[dt] = None) -> Series[str]:
    try:
      urls = self.xbrls(date)

    except Exception:
      filings = self.filings(['10-Q', '10-K'], date, True)
      urls = await xbrl_urls(self.cik, filings.index.to_list(), 'htm')

    return urls

  async def get_financials(self, date: Optional[dt] = None) -> list[RawFinancials]:
    xbrls = await self.xbrl_urls(date)
    return await parse_statements(xbrls.tolist())

  def get_calc_template(self, doc_id):
    ns = 'http://www.w3.org/1999/xlink'

    url = xbrl_url(self.cik, doc_id)
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

  async def get_taxonomy(self) -> DataFrame:
    async def fetch():
      docs = self.filings(['10-K', '10-Q']).index
      tasks = []
      for doc in docs:
        url = await xbrl_url(self.cik, doc, 'cal')
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
