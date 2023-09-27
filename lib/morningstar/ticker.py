from datetime import datetime as dt
import re
from typing import Literal, Optional

import bs4 as bs
import numpy as np
import requests
import pandas as pd

from lib.const import HEADERS
from lib.models import OHLCV
from lib.morningstar.fetch import get_currency
from lib.utils import replace_all

SCREENER_API = (
  'https://tools.morningstar.co.uk/api/rest.svc'
  '/dr6pz9spfi/security/screener'
)
#'https://lt.morningstar.com/api/rest.svc/klr5zyak8x/security/screener'

class Ticker():
  __slots__ = ('_id', '_security', '_currency')

  def __init__(self, 
    id: str, 
    security: Literal['stock', 'etf', 'fund'], 
    currency: Optional[str]=''
  ):
    self._id = id # Morningstar ID
    self._security = security
    self._currency = currency if currency else get_currency(id)

  def ohlcv(self, start_date:str='1970-01-01'):

    def parser(params):
      with requests.Session() as s:
        rs = s.get(url, headers=HEADERS, params=params)
        if rs.status_code != 200:
          return None
        parse = rs.json()
      
      scrap: list[OHLCV] = []
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
      url = (
        'https://tools.morningstar.no/api/rest.svc/'
        'timeseries_ohlcv/dr6pz9spfi'
      )
    else:
      params['priceType'] = ''
      url = (
        'https://tools.morningstar.no/api/rest.svc/'
        'timeseries_price/dr6pz9spfi'
      )
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
      if row.get('class') == 'majorGrouping':
        continue
          
      scrap_row = []
      for col in row.findAll('td'): 
        if col.text == '-':
          scrap_row.append(np.nan)
        else:
          scrap_row.append(int(replace_all(col.text, rpl)) * 1e6)
              
      scrap.append(scrap_row)
                
    cols = (
      'Date', 'Revenue', 'CostOfRevenue', 'GrossProfit',
      'RnD', 'SGA', 'Salaries', 'Depreciation', 'OtherOperatingExpenses',
      'OperatingExpense', 'EBIT', 'TaxableIncome', 'EBT', 'FinancialResult', 
      'NetIncomeFromContinuingOperations', 'NetIncome', 
      'NetIncomeCommonStockholders', 'BasicEPS', 'DilutedEPS'
    )
    
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
    tbl_cls = 'snapshotTextColor snapshotTextFontStyle snapshotTable'
    tbl = soup.find('table', {'class': f'{tbl_cls} overviewKeyStatsTable'})
    buyFee = tbl.findAll('tr')[7].findAll('td')[2].text
    annFee = tbl.findAll('tr')[8].findAll('td')[2].text
    buyFee = re.sub(r'^-$', '0', buyFee)
    annFee = re.sub(r'^-$', '0', annFee)
    buyFee = float(replace_all(buyFee, repDic))
    annFee = float(replace_all(annFee, repDic))

    # Reference index
    tbl = soup.find('table', {'class': f'{tbl_cls} overviewBenchmarkTable2Cols'})
    vals = tbl.findAll('tr')[3].findAll('td')
    index = vals[0].text
    if vals[1].text != '-':
      index +='\n' + vals[1].text

    # Alfa/beta (best fit)
    tab = url + '&tab=2'
    with requests.Session() as s:
      rs = s.get(tab)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    tbl = soup.find('table', {'class': f'{tbl_cls} ratingMptStatsTable'})
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

    attrs = {'class': f'{tbl_cls} portfolioEquityStyleTable'}
    tbl = soup.find('table', attrs).findAll('table', attrs)
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
        
    cols = (
      'bmIndex', 'alfa (bf)', 'beta (bf)', 'peRatio', 'peRatioRel', 
      'pbRatio', 'pbRatioRel', 'psRatio', 'psRatioRel', 'pfcfRatio', 
      'pfcfRatioRel', 'dyf', 'dyf (rel)', 'est rg', 'est rg (rel)', 
      'rg', 'rg (rel)', 'saleGrwth', 'saleGrwthRel', 'fcfGrwth', 
      'fcfGrwthRel', 'bvGrwth', 'bvGrwthRel', 'buyFee', 'anlFee'
    )
    
    df = pd.DataFrame(scrap, columns=cols)
    return df

  def documents(self):
      
    p = 0
    params = {
      'id': f'{self._id}]3]0]E0WWE`$`$ALL',
      'currencyId': 'NOK',
      'languageId': 'nb-NO',
      'pageSize': '15',
      'moduleId': '59',
      'documentCategory': 'financials',
      'pageNumber': f'{p}',
    }
    url = (
      'https://tools.morningstar.no/api/rest.svc/'
      'security_documents/dr6pz9spfi'
    )
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