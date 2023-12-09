import asyncio
import re
from typing import Literal, Optional

import httpx
import numpy as np
import pandas as pd
import pycountry

from lib.const import HEADERS
from lib.utils import replace_all

PARAMS = {
  'page': '1',
  'pageSize': '1',
  'sortOrder': 'Name asc',
  'outputType': 'json',
  'version': '1',
  'languageId': 'en-US',
  'currencyId': 'NOK',
  'filters': '',
  'term': '',
  'securityDataPoints': '',
  'subUniverseId': '',
}

async def fetch_api(
  params: dict[str, str], 
  timeout: Optional[float|int] = None
) -> dict:
  url = (
    'https://tools.morningstar.co.uk/api/rest.svc'
    '/dr6pz9spfi/security/screener'
  )
  client_timeout = httpx.Timeout(timeout) if timeout else None

  async with httpx.AsyncClient(timeout=client_timeout) as client:
    rs = await client.get(url, headers=HEADERS, params=params)
    return rs.json()

async def get_currency(id: str) -> str:
  params: dict[str, str] = PARAMS.copy()
  params['term'] = id

  parse = await fetch_api(params)
  return parse['rows'][0].get('PriceCurrency', 'USD')

async def get_tickers(
  security: Literal['stock', 'etf', 'index', 'fund', 'fund_category', 'ose']
) -> pd.DataFrame:

  rename = {
    'SecId': 'id',
    'LegalName': 'legal_name',
    'ExchangeId': 'mic',
    'SectorName': 'sector',
    'IndustryName': 'industry',
    'CategoryName': 'category',
  }
  fields = {
    'stock': (
      'SecId', 'Ticker', 'Name', 'LegalName', 'ExchangeId', 'Currency', 
      'SectorName', 'IndustryName', 'ClosePrice'
    ),
    'eft': (
      'SecId' 'Ticker', 'Name', 'ExchangeId', 'Currency', 
      'CategoryName',  'ClosePrice'
    ),
    'index': ('SecId', 'Name', 'Currency'),
    'fund': ('SecId', 'LegalName', 'Currency', 'CategoryName', 'ClosePrice'),
    'func_category': ('name', 'id')
  }
  universe = {
    'stock': 'E0WWE$$ALL',
    'etf': 'ETEXG$XOSE|ETEUR$$ALL',
    'index': 'IXMSX$$ALL',
    'fund': 'FONOR$$ALL',
    'func_category': 'FONOR$$ALL'
  }

  def parse_data(
    data: dict, 
    security: str,
  ) -> list[dict[str, str|float]]:

    if security == 'fund_category':
      return data['filters'][0][0]['CategoryId']

    return data['rows']

  async def fetch_data(
    params: dict[str, str], 
    pages: int
  ) -> list[dict]:

    tasks = []
    for p in range(2, pages + 1):
      params['page'] = str(p)
      tasks.append(asyncio.create_task(
        fetch_api(params.copy(), 60)
      ))
    data = await asyncio.gather(*tasks, return_exceptions=True)
    return data

  async def fallback(
    params: dict[str, str],
    bound: int,
    price: float
  ) -> list[dict[str, str]]:

    params.update({
      'page': '1', 
      'filters': f'ClosePrice:LTE:{price}'
    })

    scrap: list[dict[str, str]] = []
    while price > 0 and len(scrap) < bound:
      data = await fetch_api(params, 60)
      scrap.extend(parse_data(data, security))
      price = scrap[-1].get('ClosePrice', 0.0)
      params['filters'] = f'ClosePrice:LTE:{price}'

    return scrap

  page_size = 50000
  params: dict[str, str] = PARAMS.copy()
  sort_order = 'ClosePrice desc' if 'ClosePrice' in fields[security] else 'Name asc'
  params.update({
    'pageSize': str(page_size),
    'sortOrder': sort_order,
    'securityDataPoints': '|'.join(fields[security]),
    'universeIds': universe[security]
  })
  if security == 'fund_category':
    params['filterDataPoints'] = 'CategoryId'

  data = await fetch_api(params, 60)
  total: int = data['total']
  scrap = parse_data(data, security)

  if total > page_size:
    pages = int(np.ceil(total / page_size))
    data = await fetch_data(params, pages)

    error = False
    for d in data:
      if isinstance(d, Exception):
        error = True
        continue
      
      scrap.extend(parse_data(d, security))      

    if error:
      price: float = scrap[-1].get('ClosePrice', 0.0)
      if price > 0.0:
        temp = await fallback(params, total - len(scrap), price)
        scrap.extend(temp)

  df = pd.DataFrame.from_records(scrap)
  df.drop_duplicates(inplace=True)

  if 'ClosePrice' in df.columns:
    df = df[df['ClosePrice'] > 0.0]
    df.drop('ClosePrice', axis=1, inplace=True)

  df.rename(columns=rename, inplace=True)
  df.columns = df.columns.str.lower()
  
  if security == 'stock':
    # Extract MIC
    pattern = r'^EX(\$+|TP\$+)'
    df['mic'] = df['mic'].str.replace(pattern, '', regex=True)

  return df

async def fund_data():
    
  equity_style = ('NA', 
    'Value-Big Cap', 'Mix-Big Cap', 'Growth-Big Cap',
    'Value-Mid Cap', 'Mix-Mid Cap', 'Growth-Mid Cap',
    'Value-Small Cap', 'Mix-Small Cap', 'Growth-Small Cap'
  )
  bond_style = ('NA',
    'Hi CQ-Lo IS', 'Hi CQ-Mi IS', 'Hi CQ-Hi IS',
    'Mi CQ-Lo IS', 'Mi CQ-Mi IS', 'Mi CQ-Hi IS',
    'Lo CQ-Lo IS', 'Lo CQ-Mi IS', 'Lo CQ-Hi IS'
  )
  fields = {
    'SecId': 'id', 
    'LegalName': 'name', 
    'CategoryName': 'category',
    'PriceCurrency': 'currency',
    'EquityStyleBox': 'equity_style', 
    'BondStyleBox': 'bond_style',
    'AverageCreditQualityCode': 'avg_credit_quality',
    'StarRatingM255': 'rating', 
    'SustainabilityRank': 'sustainaility',
    'GBRReturnM0': 'return:ty',
    'GBRReturnM12': 'return:1y', 
    'GBRReturnM36': 'return:3y', 
    'GBRReturnM60': 'return:5y', 
    'GBRReturnM120': 'return:10',
    'InitialPurchase': 'init_purch',
    'AlphaM36': 'alpha:3y',
    'BetaM36': 'beta:3y', 
    'R2M36': 'r2:3y',
    'StandardDeviationM36': 'std:36', 
    'SharpeM36': 'sharpe:3y', 
    'SortinoM36': 'sortino:3y', 
    'PERatio': 'pe_ratio', 
    'PBRatio': 'pb_ratio'
  }
  params: dict[str, str] = PARAMS.copy()
  params.update({
    'pageSize': '50000',
    'sortOrder': 'LegalName asc',
    'universeIds': 'FONOR$$ALL',
    'securityDataPoints': '|'.join(fields.keys()),
  })
  parse = await fetch_api(params)
  data = parse['rows']
  df = pd.DataFrame.from_records(data)
  df.rename(columns=fields, inplace=True)
  df['equity_style'] = df['equity_style'].apply(lambda x: equity_style[x])
  df['bond_style'] = df['bond_style'].apply(lambda x: bond_style[x])
  return df

def index_data(
  scope: Literal['country', 'region', 'sector-us', 'sector-global']):
    
  scope_list = {
    'country': 'countryReturnData',
    'region': 'regionReturnData',
    'sector-us': 'sectorReturnDataUS',
    'sector-global': 'sectorReturnDataGlobal'
  }
  params = {
    'top': '!MSTAR',
    'clientId': 'undefined',
    'benchmarkId': 'category',
    'version': '3.35.0',
  }
  url = (
    'https://api.morningstar.com/sal-service/v1/'
    f'index/valuation/{scope_list[scope]}'
  )
  
  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS, params=params)
    parse = rs.json()
      
  scrap = []
  for i in parse['gmbValuationDataList']:
    scrap.append({
      'id': i.get('performanceId'),
      'ticker': i.get('ticker'),
      'name': i.get('indexName'),
      'pfv_ratio:mr': i.get('pfvMostRecent'),
      'pfv_ratio:3m': i.get('pfvMost3M'),
      'pfv_ratio:1y': i.get('pfvMost1Y'),
      'pfv_ratio:5y': i.get('pfvMost5Y'),
      'pfv_ratio:10y': i.get('pfvMost10Y'),
    })
      
  df = pd.DataFrame.from_records(scrap)
  
  def alpha3(s):
    pattern = r'^Morningstar [a-zA-Z.]+\s?[a-zA-Z]* Inde'
    match = re.search(pattern, s)
    
    country = match.group()
    replace = {
      'Morningstar ': '', ' Inde': '', ' Market': '',
      'Czech Republic': 'Czechia', 'Korea': 'Korea, Republic of', 
      'Russia': 'Russian Federation',
      'Taiwan': 'Taiwan, Province of China',
      'UK': 'United Kingdom', 'U.S.': 'United States',
    }
    country = replace_all(country, replace)
    alpha3 = pycountry.countries.get(name=country).alpha_3
    
    return alpha3
          
  df['country'] = df['name'].apply(alpha3)
      
  return df

'''
    # Trim tickers (for merging)
    df['tickerTrim'] = df['ticker'].str.lower()
    patterns = {
      'XIST': r'\.e$',
      'XCAI': r'\.ca\.?$',
      'XTAE': r'(-|\.)(m|l)$',
      'MISX': r'-rm(dr)?$',
      'XCSE': r'\sdkk$',
      'XBLB': r'-r-a$',
      'XZIM': r'\.zw$',
      'XHKG': r'(^0+(?=[1-9][0-9a-z]*))'
    }
    for k, v in patterns.items():
      mask = df['mic'] == k
      df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'] \
        .str.replace(v, '', regex=True)

    # Trim ticker for suffixes and special characters 
    pattern = (
      r'((?<=\.p)r(?=\.?[a-z]$))|' # .PRx
      r'((?<=\.w)i(?=\.?[a-z]$))|' #.WIx
      r'((?<=\.(r|u|w))t(?=[a-z]?$))|' # .RT/UT/WT
      r'((\.|/)?[inpx][047]{4}$)|'
      r'(\s|\.|_|-|/|\')'
    )
    df['tickerTrim'] = df['tickerTrim'].str.replace(pattern, '', regex=True)
'''