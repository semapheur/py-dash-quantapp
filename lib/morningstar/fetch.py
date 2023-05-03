import re
from typing import Literal, Optional

import asyncio
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
    'LegalName': 'name',
    'ExchangeId': 'mic',
    'SectorName': 'sector',
    'IndustryName': 'industry',
    'CategoryName': 'category',
  }
  fields = {
    'stock': (
      'SecId', 'Ticker', 'Name', 'ExchangeId', 'Currency', 
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
    data = asyncio.run(fetch_data(params, pages))

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
    
  eStyle = {
    0: 'NA',
    1: 'Value-Big Cap',
    2: 'Mix-Big Cap',
    3: 'Growth-Big Cap',
    4: 'Value-Mid Cap',
    5: 'Mix-Mid Cap',
    6: 'Growth-Mid Cap',
    7: 'Value-Small Cap',
    8: 'Mix-Small Cap',
    9: 'Growth-Small Cap'
  }
  bStyle = {
    0: 'NA',
    1: 'Hi CQ-Lo IS',
    2: 'Hi CQ-Mi IS',
    3: 'Hi CQ-Hi IS',
    4: 'Mi CQ-Lo IS',
    5: 'Mi CQ-Mi IS',
    6: 'Mi CQ-Hi IS',
    7: 'Lo CQ-Lo IS',
    8: 'Lo CQ-Mi IS',
    9: 'Lo CQ-Hi IS'
  }
  fields = (
    'SecId', 'PriceCurrency', 'LegalName', 'CategoryName',
    'StarRatingM255', 'SustainabilityRank', 'GBRReturnM0',
    'GBRReturnM12', 'GBRReturnM36', 'GBRReturnM60', 'GBRReturnM120',
    'InitialPurchase', 'EquityStyleBox', 'BondStyleBox',
    'AverageCreditQualityCode', 'AlphaM36', 'BetaM36', 'R2M36',
    'StandardDeviationM36', 'SharpeM36', 'SortinoM36', 'PERatio', 'PBRatio'
  )
  params: dict[str, str] = PARAMS.copy()
  params.update({
    'pageSize': '50000',
    'sortOrder': 'LegalName asc',
    'universeIds': 'FONOR$$ALL',
    'securityDataPoints': '|'.join(fields),
  })
  
  parse = await fetch_api(params)
  
  scrap =  []
  for f in parse['rows']:
    scrap.append({
      'name': f.get('LegalName'),
      'id': f.get('SecId'),
      'currency': f.get('PriceCurrency'),
      'category': f.get('CategoryName'),
      'eqtStyle': eStyle[f.get('EquityStyleBox', 0)],
      'bndStyle': bStyle[f.get('BondStyleBox', 0)],
      'rating': f.get('StarRatingM255', 0),
      'sustainability': f.get('SustainabilityRank', np.nan),
      'return_ty': f.get('GBRReturnM0', np.nan),
      'return_1y': f.get('GBRReturnM12', np.nan),
      'return_3y': f.get('GBRReturnM36', np.nan),
      'return_5y': f.get('GBRReturnM60', np.nan),
      'avgCrdtQy': f.get('AverageCreditQualityCode', 'NA'),
      'std': f.get('StandardDeviationM36', np.nan),
      'sharpe': f.get('SharpeM36', np.nan),
      'sortino': f.get('SortinoM36', np.nan),
      'r2': f.get('R2M36', np.nan),
      'alpha': f.get('AlphaM36', np.nan),
      'beta': f.get('BetaM36', np.nan),
      'peRatio': f.get('PERatio', np.nan),
      'pbRatio': f.get('PBRatio', np.nan),
      'initPrchLimit': f.get('InitialPurchase', np.nan),
    })

  df = pd.DataFrame.from_records(scrap)
  return df

def index_data(scope):
    
  scopeList = {
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
    f'index/valuation/{scopeList[scope]}'
  )
  
  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS, params=params)
    parse = rs.json()
      
  scrap = []
  for i in parse['gmbValuationDataList']:
    scrap.append({
      'mornId': i.get('performanceId'),
      'ticker': i.get('ticker'),
      'name': i.get('indexName'),
      'pfvRatio_mr': i.get('pfvMostRecent'),
      'pfvRatio_3m': i.get('pfvMost3M'),
      'pfvRatio_1y': i.get('pfvMost1Y'),
      'pfvRatio_5y': i.get('pfvMost5Y'),
      'pfvRatio_10y': i.get('pfvMost10Y'),
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