from datetime import datetime as dt
from datetime import timezone as tz
import re
import time
from typing import Optional

import bs4 as bs
import httpx
import numpy as np
import pandas as pd

from lib.const import HEADERS
from lib.db.lite import read_sqlite

class Ticker:
    
  def __init__(self, ticker: str):
    self._ticker = ticker
  
  def ohlcv(self, 
    start_date: Optional[str|dt] = None, 
    period: Optional[str] = None
  ) -> pd.DataFrame:
      
    def parse_json(start_stamp: int, end_stamp: int) -> dict:
      params = {
        'formatted': 'true',
        #'crumb': '0/sHE2JwnVF',
        'lang': 'en-US',
        'region': 'US',
        'includeAdjustedClose': 'true',
        'interval': '1d',
        'period1': str(start_stamp),
        'period2': str(end_stamp),
        'events':'div|split',
        'corsDomain': 'finance.yahoo.com',
      }
      url = f'https://query2.finance.yahoo.com/v8/finance/chart/{self._ticker}'

      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        data = rs.json()
      
      return data['chart']['result'][0]
                  
    if start_date and (period is None):
      if isinstance(start_date, str):
        startDate = dt.strptime(start_date, '%Y-%m-%d')

      start_stamp = int(startDate.replace(tzinfo=tz.utc).timestamp())
        
    elif period == 'max':
      start_stamp = int(dt.today().timestamp())
      start_stamp -= 3600 * 24

    else:
      start_stamp = int(dt(2000, 1, 1).replace(tzinfo=tz.utc).timestamp())
    
    end_stamp = int(dt.now().replace(tzinfo=tz.utc).timestamp())
    end_stamp += 3600 * 24

    parse = parse_json(start_stamp, end_stamp)
    
    if period == 'max':
      start_stamp = parse['meta']['firstTradeDate']
      parse = parse_json(start_stamp, end_stamp)
    
    # Index
    ix = parse['timestamp']

    # OHLCV
    ohlcv = parse['indicators']['quote'][0]
    adj_close = parse['indicators']['adjclose'][0]['adjclose']

    data = {
      'open': ohlcv['open'],
      'high': ohlcv['high'],
      'low': ohlcv['low'],
      'close': ohlcv['close'],
      'adjusted_close': adj_close,
      'volume': ohlcv['volume'],
    }

    # Parse to DataFrame
    df = pd.DataFrame.from_dict(data, orient='columns')
    df.index = pd.to_datetime(ix, unit='s').floor('D') # Convert index from unix to date
    df.index.rename('date', inplace=True)

    return df

  def price_targets(self) -> pd.DataFrame:

    url = f'https://finance.yahoo.com/quote/{self._ticker}/analysis'
    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS)
      soup = bs.BeautifulSoup(rs.text, 'lxml')

    div = soup.find('div', {'id': 'Col2-9-QuoteModule-Proxy'})

    scrap = div.find('div', {'aria-label': True}).get('aria-label')

    pt = scrap.split(' ')[1::2]
    pt.insert(0, self._ticker)
    cols = ('ticker', 'low', 'current', 'average', 'high')
    df = pd.DataFrame(pt, columns=cols)
    return df

  def financials(self) -> pd.DataFrame:
            
    def parse(period: str):
      end_stamp = int(dt.now().timestamp()) + 3600*24

      params = {
        'lang': 'en-US',
        'region': 'US',
        'symbol': self._ticker,
        'padTimeSeries': 'true',
        'type': ','.join([period + i for i in items['yahoo']]),
        'merge': 'false',
        'period1': '493590046',
        'period2': str(end_stamp),
        'corsDomain': 'finance.yahoo.com',
      }
      url = ('https://query2.finance.yahoo.com/ws/'
        f'fundamentals-timeseries/v1/finance/timeseries/{self._ticker}')
      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        parse = rs.json()

      dfs = []
      pattern = r'^(annual|quarterly)'
      for r in parse['timeseries']['result']:
        item = r['meta']['type'][0]

        if item not in r:
          continue

        scrap = {}
        for e in list(filter(None, r[item])):
          date = dt.strptime(e['asOfDate'], '%Y-%m-%d')
          scrap[date] = e['reportedValue']['raw']

        df = pd.DataFrame.from_dict(
          scrap, orient='index', 
          columns=[re.sub(pattern, '', item)])

        dfs.append(df)

      df = pd.concat(dfs, axis=1)
      df['period'] = period[0]

      return df
    
    query = '''
      SELECT json_each.value AS yahoo, item FROM items 
      JOIN JSON_EACH(yahoo) ON 1=1
      WHERE yahoo IS NOT NULL
    '''
    items = read_sqlite('taxonomy.db', query)

    dfs = []
    for p in ('annual', 'quarterly'):
      dfs.append(parse(p))
        
    df = pd.concat(dfs)
    
    col_map = {
      k: v for k, v in zip(items['yahoo'], items['item'])
    }
    df.rename(columns=col_map, inplace=True)
    df.set_index('period', append=True, inplace=True)
    df.index.names = ['date', 'period']

    return df

  def option_chains(self) -> pd.DataFrame:

    def parse_option_chain(parse: dict, stamp):
        
      def get_entry(opt: dict, key: str):
        if key in opt:
          if isinstance(opt[key], dict):
            entry = opt[key].get('raw')
          elif isinstance(opt[key], bool):
            entry = opt[key]
        else:
          entry = np.nan
            
        return entry
      
      options = parse['optionChain']['result'][0]['options'][0]

      cols = ('strike', 'impliedVolatility', 'openInterest', 
        'lastPrice', 'ask', 'bid', 'inTheMoney')
      cells = np.zeros((len(cols), (len(options['calls']) + len(options['puts']))))
      
      # Calls
      for i, opt in enumerate(options['calls']):
        for j, c in enumerate(cols):                               
          cells[j, i] = get_entry(opt, c)
              
      # Puts
      for i, opt in enumerate(options['puts']):
        for j, c in enumerate(cols):
          cells[j, i+len(options['calls'])] = get_entry(opt, c)

      data = {k: v for k, v in zip(cols, cells)}
      data['optionType'] = np.array(
        ['call'] * len(options['calls']) + ['put'] * len(options['puts']))
      
      # Parse to data frame
      df = pd.DataFrame.from_records(data)

      # Add expiry date
      date = dt.utcfromtimestamp(stamp)
      df['expiry'] = date
              
      return df
    
    params = {
      'formatted': 'true',
      #'crumb': '2ztQhfMEzsm',
      'lang': 'en-US',
      'region': 'US',
      'corsDomain': 'finance.yahoo.com',
    }
    
    url = ('https://query1.finance.yahoo.com/'
      f'v7/finance/options/{self._ticker}')
    with httpx.Client() as client:
      rs= client.get(url, headers=HEADERS, params=params)
      parse = rs.json()
    
    # Maturity dates
    stamps = parse['optionChain']['result'][0]['expirationDates']

    # Parse first option chain to dataframe
    dfs = []
    dfs.append(parse_option_chain(parse, stamps[0]))

    # Parse remaining option chains
    for i in range(1, len(stamps)):
      time.sleep(1)
        
      params[-1] = ('date', stamps[i])
        
      #q = (i % 2) + 1
      #url = f'https://query{q}.finance.yahoo.com/v7/finance/options/{ticker}'
      
      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        parse = rs.json()

      dfs.append(parse_option_chain(parse, stamps[i]))

    # Concatenate
    df = pd.concat(dfs)
    
    return df
    
def batchOhlcv(tickers, startDate='', period=None):

    if isinstance(tickers, str):
        ticker = Ticker(tickers)
        return Ticker.ohlcv(startDate, period)

    elif isinstance(tickers, list):
        dfs = []
        for t in tickers:
            ticker = Ticker(t)
            
            df = ticker.ohlcv(startDate, period)

            cols = pd.MultiIndex.from_product([[t], [c for c in df.columns]])
            df.columns = cols

            dfs.append(df)

        return pd.concat(dfs, axis=1)
    else:
        return None