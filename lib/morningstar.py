import numpy as np
import pandas as pd

# Web scrapping
import requests
import json
import bs4 as bs

# Date
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

# Utils
import re
import pycountry

# Local
from lib.db.lite import read_sqlite, upsert_sqlite
from lib.utils import replace_all

SCREENER_API = 'https://tools.morningstar.co.uk/api/rest.svc/dr6pz9spfi/security/screener'
#'https://lt.morningstar.com/api/rest.svc/klr5zyak8x/security/screener'

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.5',
  # 'Accept-Encoding': 'gzip, deflate, br',
  'DNT': '1',
  'Connection': 'keep-alive',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'cross-site',
  'Sec-GPC': '1',
  # Requests doesn't support trailers
  # 'TE': 'trailers',
}

class Ticker():

  def __init__(self, id: str, security: str, currency: str=''):
    self._id = id # Morningstar ID
    self._security = security
    self._currency = currency if currency else get_currency(id)

  def ohlcv(self, start_date='1970-01-01'):

    def parser(params):
      with requests.Session() as s:
        rs = s.get(url, headers=HEADERS, params=params)
        if rs.status_code != 200:
          return None
        parse = rs.json()
      
      scrap = []
      if self._security == 'stock':
        for d in parse:
          scrap.append({
            'date': d[0], # Unix in miliseconds
            'open': d[1],
            'high': d[2],
            'low': d[3],
            'close': d[4],
            'volume': d[5]
          })
      else:
        for d in parse:
          scrap.append({
            'date': d[0],
            'close': d[1]
          })
      
      df = pd.DataFrame.from_records(scrap)
      df['date'] = pd.to_datetime(df['date'], unit='ms')
      df.set_index('date', inplace=True)
      return df     

    # Check if date is given in string
    if isinstance(start_date, dt):
      start_date = start_date.strftime('%Y-%m-%d')
    
    #endDate = dt.now().strftime('%Y-%m-%d')

    params = {
      'currencyId': self._currency,
      'idtype': 'Morningstar',
      'frequency': 'daily',
      'startDate': start_date,
      #'endDate': endDate,
      'outputType': 'COMPACTJSON',
      'applyTrackRecordExtension': 'true',
    }

    if self._security == 'stock':
      params['performanceType'] = ''
      params['id'] = f'{self._id}]3]0]' # E0WWE$$ALL
      url = 'https://tools.morningstar.no/api/rest.svc/timeseries_ohlcv/dr6pz9spfi'

    else:
      params['priceType'] = ''
      url = 'https://tools.morningstar.no/api/rest.svc/timeseries_price/dr6pz9spfi'

      if self._security == 'etf':
        params['performanceType'] = ''
        params['id'] = f'{self._id}]22]1]'

      elif self._security == 'fund':
        params['id'] = f'{self._id}]2]1]'

    df = parser(params)       
    return df

  def financials(self):
      # WIP

    params = {
      'tab': '10',
      'vw': 'bs',
      'SecurityToken': f'{self._id}]3]0]E0WWE$$ALL',
      'Id': f'{self._id}',
      'ClientFund': '0',
      'CurrencyId': self._currency,
    }

    url = 'https://tools.morningstar.no/no/stockreport/default.aspx'
    
    with requests.Session() as s:
      rs = s.get(url, headers=HEADERS, params=params)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    
    table = soup.find('table', {'class': 'right years5'})
    
    scrap = []
    
    tr = table.findAll('tr')
    
    rpl = {' ': '', ',': '.'}
    
    # Dates
    scrap_row = []
    for h in tr[0].findAll('th')[1:]:
      scrap_row.append(dt(year=int(h.text), month=12, day=31))
                        
    scrap.append(scrap_row)
    
    for row in tr[1:]:
        
      if row.get('class') != 'majorGrouping':
          
          scrap_row = []
          for col in row.findAll('td'):
              
            if col.text == '-':
              scrap_row.append(np.nan)
            else:
              scrap_row.append(int(replace_all(col.text, rpl)) * 1e6)
                  
          scrap.append(scrap_row)
                
    cols = [
      'Date', 'Revenue', 'CostOfRevenue', 'GrossProfit',
      'RnD', 'SGA', 'Salaries', 'Depreciation', 'OtherOperatingExpenses',
      'OperatingExpense', 'EBIT', 'TaxableIncome', 'EBT', 'FinancialResult', 
      'NetIncomeFromContinuingOperations', 'NetIncome', 
      'NetIncomeCommonStockholders', 'BasicEPS', 'DilutedEPS'
    ]
    
    df = pd.DataFrame(np.array(scrap).T, columns=cols)
    
    return df

  def fund_details(self):
    # WIP
    if self._security != 'fund':
      return None

    scrap = []

    # String replacements
    repDic = {
      ',': '.',
      '%': '',
      'n/a': '0',
      '1\xa0\n            NOK': '0'
    }

    url = 'https://www.morningstar.no/no/funds/snapshot/snapshot.aspx?id=' + self._id
    with requests.Session() as s:
      rs = s.get(url)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    # Fees
    tbl = soup.find('table', {'class': 'snapshotTextColor snapshotTextFontStyle snapshotTable overviewKeyStatsTable'})
    buyFee = tbl.findAll('tr')[7].findAll('td')[2].text
    annFee = tbl.findAll('tr')[8].findAll('td')[2].text
    buyFee = re.sub(r'^-$', '0', buyFee)
    annFee = re.sub(r'^-$', '0', annFee)
    buyFee = float(replace_all(buyFee, repDic))
    annFee = float(replace_all(annFee, repDic))

    # Reference index
    tbl = soup.find('table', {'class': 'snapshotTextColor snapshotTextFontStyle snapshotTable overviewBenchmarkTable2Cols'})
    vals = tbl.findAll('tr')[3].findAll('td')
    index = vals[0].text
    if vals[1].text != '-':
      index +='\n' + vals[1].text

    # Alfa/beta (best fit)
    tab = url + '&tab=2'
    with requests.Session() as s:
      rs = s.get(tab)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    tbl = soup.find('table', {'class': 'snapshotTextColor snapshotTextFontStyle snapshotTable ratingMptStatsTable'})
    beta = tbl.findAll('tr')[3].findAll('td')[2].text
    alfa = tbl.findAll('tr')[4].findAll('td')[2].text
    beta = re.sub(r'^-$', '0', beta)
    alfa = re.sub(r'^-$', '0', alfa)
    beta = float(replace_all(beta, repDic))
    alfa = float(replace_all(alfa, repDic))

    # Portfolio
    tab = url + '&tab=3'
    with requests.Session() as s:
      rs = s.get(tab)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    # Check if equity/bond fund
    label = 'snapshotTextColor snapshotTextFontStyle snapshotTable portfolioEquityStyleTable'
    tbl = soup.find('table', {'class': label}).findAll('table', {'class': label})
    if tbl:
      ratios = []
      for row in tbl[2].findAll('tr')[1:6]:
        portfolio = float(replace_all(row.findAll('td')[1].text, repDic))
        relative = float(replace_all(row.findAll('td')[2].text, repDic))
        ratios.extend([portfolio, relative])
      for row in tbl[2].findAll('tr')[7:]:
        portfolio = float(replace_all(row.findAll('td')[1].text, repDic))
        relative = float(replace_all(row.findAll('td')[2].text, repDic))
        ratios.extend([portfolio, relative])
    else:
      ratios = [0]*20

    # Append to scrap matrix
    scrap.extend([index, alfa, beta])
    scrap.extend(ratios)
    scrap.extend([buyFee, annFee])
        
    cols = ['bmIndex', 'alfa (bf)', 'beta (bf)', 'peRatio', 'peRatioRel', 
            'pbRatio', 'pbRatioRel', 'psRatio', 'psRatioRel', 'pfcfRatio', 'pfcfRatioRel', 
            'dyf', 'dyf (rel)', 'est rg', 'est rg (rel)', 'rg', 'rg (rel)', 
            'saleGrwth', 'saleGrwthRel', 'fcfGrwth', 'fcfGrwthRel', 'bvGrwth', 'bvGrwthRel', 
            'buyFee', 'anlFee']
    
    df = pd.DataFrame(scrap, columns=cols)
    return df

  def documents(self):
      
    p = 0
    params = [
      ('id', f'{self._id}]3]0]E0WWE`$`$ALL'),
      ('currencyId', 'NOK'),
      ('languageId', 'nb-NO'),
      ('pageSize', '15'),
      ('moduleId', '59'),
      ('documentCategory', 'financials'),
      ('pageNumber', f'{p}'),
    ]

    url = 'https://tools.morningstar.no/api/rest.svc/security_documents/dr6pz9spfi'
    with requests.Session() as s:
      rs = s.get(url, headers=HEADERS, params=params)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
        
    pages = soup.find('documents').get('totalnumberofpages')
    
    scrap = []
    while p <= int(pages):
        
      docs = soup.findAll('document')
      for d in docs:
                      
        att = d.find('attributes')
        if att is not None:
          repType = att.text.replace('\n', '')
        
        else:
          repType = 'N/A'
        
        scrap.append({
          'date': d.find('effectivedate').text,
          'type': repType,
          'language': d.find('languageid').text.replace('\n', ''),
          'format': d.find('format').text,
          'link': d.find('downloadurl').text.replace('amp;', '')
        })

      p += 1
      params[-1] = ('pageNumber', f'{p}'),

      with requests.Session() as s:
        rs = s.get(url, headers=HEADERS, params=params)
        soup = bs.BeautifulSoup(rs.text, 'lxml')
        
    df = pd.DataFrame.from_records(scrap)
    return df

def get_currency(id: str) -> str:
    params = {
      'page': '1',
      'pageSize': '1',
      'sortOrder': 'Name asc',
      'outputType': 'json',
      'version': '1',
      'languageId': 'en-US',
      'currencyId': 'USD',
      'filters': '',
      'term': id,
      'securityDataPoint': 'PriceCurrency',
      'subUniverseId': '',
    }

    with requests.Session() as s:
      rs = s.get(SCREENER_API, headers=HEADERS, params=params)
      parse = json.loads(rs.text)

    return parse['rows'][0].get('PriceCurrency', 'USD')

def get_ohlcv(
  id: str, 
  security: str, 
  currency='', 
  delta=1,
  cols: list[str]=[]
) -> pd.DataFrame:
  
  if not cols:
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']

  if 'date' not in cols:
    cols.append('date')
  
  query = f'SELECT {", ".join(cols)} FROM {security} WHERE id = "{id}"'
  ohlcv = read_sqlite(query, 'ohlcv.db', 'date', True)

  if ohlcv is None:
    ohlcv = Ticker(id, security, currency).ohlcv()
    ohlcv['id'] = id
    ohlcv.set_index(['id'], append=True, inplace=True)
    upsert_sqlite(ohlcv, 'ohlcv.db', security)
    ohlcv.reset_index(level='id', drop=True, inplace=True)

    return ohlcv[cols]
  
  last_date = ohlcv.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days > delta:
    new_ohlcv = Ticker(id, currency, security).ohlcv(last_date.strftime('%Y-%m-%d'))

    if new_ohlcv is None:
      return ohlcv

    ohlcv['id'] = id
    ohlcv.set_index(['id'], append=True, inplace=True)
    upsert_sqlite(ohlcv, 'ohlcv.db', security)
    ohlcv = read_sqlite(query, 'ohlcv.db', 'date', True)

  return ohlcv

def get_tickers(security: str) -> pd.DataFrame:        
  # scope: ose/stock/fund/fundCategory

  def fallback() -> list[dict[str, str]]:
    params = {
      'page': '1',
      'pageSize': '1',
      'sortOrder': 'ClosePrice desc',
      'outputType': 'json',
      'version': '1',
      'languageId': 'en-US',
      'currencyId': 'NOK',
      'filters': '',
      'term': '',
      'subUniverseId': '',
    }

    with requests.Session() as s:
      rs = s.get(SCREENER_API, headers=HEADERS, params=params)
      content = json.loads(rs.text)

    price = content['rows'][0]['ClosePrice']
    page_size = 1000
    params['pageSize'] = str(page_size)

    scrap = []
    while price > 0:
      temp, price = extract_data(params)
      scrap.extend(temp)
      params['filters'] = f'ClosePrice:LTE:{price}'

    return scrap
    
  def extract_data(params: dict[str, str], security: str) -> list[dict[str, str]]:
                    
    with requests.Session() as s:
      rs = s.get(SCREENER_API, headers=HEADERS, params=params)
      if params['page'] == '2':
        print(rs.text)
      content = json.loads(rs.text)
    
    scrap = []
    if security == 'fund':
      for f in content['rows']:
        scrap.append({
          'name': f.get('LegalName'),
          'category': f.get('CategoryName', 'NA'),
          'currency': f.get('PriceCurrency'),
          'id': f.get('SecId'),
        })
    elif security == 'fundCategory':
      for fc in content['filters'][0][0]['CategoryId']:
        scrap.append({
          'name': fc.get('name'), # Category name
          'id': fc.get('id'),
        })
    elif security == 'etf':
      for e in content['rows']:
        scrap.append({
          'ticker': e.get('Ticker', ''),
          'name': e.get('Name'),
          'currency': e.get('PriceCurrency'),
          'category': e.get('CategoryName'),
          'id': e.get('SecId')
        })
    elif security == 'index':
      for i in content['rows']:
        scrap.append({
          'name': i.get('Name'),
          'currency': i.get('PriceCurrency'),
          'id': i.get('SecId')
        })
    else: # Stock
      for s in content['rows']:
        price = s.get('ClosePrice', 0)
        if price > 0:
          scrap.append({
            'id': s.get('SecId'), # Morningstar ID
            'ticker': s.get('Ticker'),
            'name': s.get('Name'), # Equity name
            'mic': s.get('ExchangeId'),
            'currency': s.get('PriceCurrency'),
            'sector': s.get('SectorName'),
            'industry': s.get('IndustryName'),
          })
            
    return scrap

  params = {
		'page': '1',
    'pageSize': '1',
    'sortOrder': 'Name asc',
    'outputType': 'json',
    'version': '1',
    'languageId': 'en-US',
    'currencyId': 'USD',
    'filters': '',
    'term': '',
    'subUniverseId': '',
	}

  if security == 'stock':
    params['universeIds'] = 'E0WWE$$ALL'
    params['securityDataPoints'] = 'SecId|Name|IndustryName|SectorName|ExchangeId|Ticker|PriceCurrency|ClosePrice'
    
  elif security == 'etf':
    params['universeIds'] = 'ETEXG$XOSE|ETEUR$$ALL'
    params['securityDataPoints'] = 'SecId|Name|PriceCurrency|TenforeId|LegalName|Ticker|Currency|CategoryName'

  elif security == 'index':
    params['universeIds'] = 'IXMSX$$ALL'
    params['securityDataPoints'] = 'SecId|Name|PriceCurrency|TenforeId|LegalName'
      
  elif security == 'fundCategory':
    params['universeIds'] = 'FONOR$$ALL'
    params['filterDataPoints'] = 'CategoryId'
      
  else:
    if security == 'ose':
      args = ('E0EXG$XOSL', 'SecId|LegalName|Name|IndustryName|SectorName|TenforeId|Universe|ExchangeId|Ticker')
    elif security == 'fund':
      args = ('FONOR$$ALL', 'SecId|Name|PriceCurrency|TenforeId|LegalName|CategoryName')

    params['universeIds'] = args[0]
    params['securityDataPoints'] = args[1]

  with requests.Session() as s:
    rs = s.get(SCREENER_API, headers=HEADERS, params=params)
    content = json.loads(rs.text)

  total = content['total']
  page_size = 50000
  params['pageSize'] = str(page_size)

  if page_size > total:
    scrap = extract_data(params, security)
  
  else:
    scrap = []

    pages = int(np.ceil(total / page_size))

    try:
      for p in range(1, pages + 1):
        print(p)
        params['page'] = str(p)
        scrap.extend(extract_data(params, security))

    except:
      scrap = fallback()

  df = pd.DataFrame.from_records(scrap)
  df.drop_duplicates(inplace=True)

  if security == 'stock':
    # Extract MIC
    pattern = r'^EX(\$+|TP\$+)'
    df['mic'] = df['mic'].str.replace(pattern, '', regex=True)

    # Trim tickers (for merging)
    #df['tickerTrim'] = df['ticker'].str.lower()
    #patterns = {
    #  'XIST': r'\.e$',
    #  'XCAI': r'\.ca\.?$',
    #  'XTAE': r'(-|\.)(m|l)$',
    #  'MISX': r'-rm(dr)?$',
    #  'XCSE': r'\sdkk$',
    #  'XBLB': r'-r-a$',
    #  'XZIM': r'\.zw$',
    #  'XHKG': r'(^0+(?=[1-9][0-9a-z]*))'
    #}
    #for k, v in patterns.items():
    #  mask = df['mic'] == k
    #  df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'].str.replace(v, '', regex=True)

    # Trim ticker for suffixes and special characters 
    #pattern = (
    #  r'((?<=\.p)r(?=\.?[a-z]$))|' # .PRx
    #  r'((?<=\.w)i(?=\.?[a-z]$))|' #.WIx
    #  r'((?<=\.(r|u|w))t(?=[a-z]?$))|' # .RT/UT/WT
    #  r'((\.|/)?[inpx][047]{4}$)|'
    #  r'(\s|\.|_|-|/|\')'
    #)
    #df['tickerTrim'] = df['tickerTrim'].str.replace(pattern, '', regex=True)

  return df

def fund_data():
    
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
  params = (
    ('page', '1'),
    ('pageSize', '50000'),
    ('sortOrder', 'LegalName asc'),
    ('outputType', 'json'),
    ('version', '1'),
    ('languageId', 'nb-NO'),
    ('currencyId', 'NOK'),
    ('universeIds', 'FONOR$$ALL'),
    ('securityDataPoints', 'SecId|PriceCurrency|LegalName|CategoryName|StarRatingM255|SustainabilityRank|GBRReturnM0|GBRReturnM12|GBRReturnM36|GBRReturnM60|GBRReturnM120|InitialPurchase|EquityStyleBox|BondStyleBox|AverageCreditQualityCode|AlphaM36|BetaM36|R2M36|StandardDeviationM36|SharpeM36|SortinoM36|PERatio|PBRatio'),
    ('filters', ''),
    ('term', ''),
    ('subUniverseId', ''),
  )
  
  with requests.Session() as s:
    rs = s.get(SCREENER_API, headers=HEADERS, params=params)
    content = json.loads(rs.text)
  
  scrap =  []
  for f in content['rows']:
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

  params = (
    ('top', '!MSTAR'),
    ('clientId', 'undefined'),
    ('benchmarkId', 'category'),
    ('version', '3.35.0'),
  )

  url = f'https://api.morningstar.com/sal-service/v1/index/valuation/{scopeList[scope]}'
  
  with requests.Session() as s:
    rs = s.get(url, headers=HEADERS, params=params)
    content = json.loads(rs.text)
      
  scrap = []
  for i in content['gmbValuationDataList']:
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