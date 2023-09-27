from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd

from lib.fin.calculation import applier

def f_score(df: pd.DataFrame) -> pd.DataFrame:
    
  ocf = df['operating_cashflow']
  
  df['piotroski_f_score'] = (
    np.heaviside(df['return_on_equity'], 0) +
    np.heaviside(applier(df['return_on_assets'], 'diff'), 0) +
    np.heaviside(ocf, 0) +
    np.heaviside(ocf / df['assets'] - df['return_on_assets'], 0) +
    np.heaviside(applier(df['debt'], 'diff'), 0) +
    np.heaviside(applier(df['quick_ratio'], 'diff'), 0) +
    np.heaviside(-applier(df['average_shares_outstanding_basic'], 'diff'), 1) +
    np.heaviside(applier(df['gross_profit_margin'], 'diff'), 0) + 
    np.heaviside(applier(df['asset_turnover'], 'diff'), 0)
  )

  return df

def z_score(df: pd.DataFrame) -> pd.DataFrame:
  
  assets = applier(df['assets'], 'avg')
  liabilities = applier(df['liabilities'], 'avg')

  df['altman_z_score'] = (
    1.2 * applier(df['operating_working_capital'], 'avg') / assets +
    1.4 * df['retained_earnings_accumulated_deficit'] / assets +
    3.3 * df['operating_cashflow'] / assets +
    0.6 * df['market_capitalization'] / liabilities +
    assets / liabilities
  )

  return df

def m_score(df: pd.DataFrame) -> pd.DataFrame:

  # Days Sales in Receivables Index
  dsri = df['average_current_trade_receivables'] / df['revenue']
  dsri /= applier(dsri, 'shift')
      
  # Gross Margin Index            
  gmi = applier(df['gross_profit_margin'], 'shift') / df['gross_profit_margin'] 
      
  # Asset Quality Index    
  aqi = (1 - (
    df['operating_working_capital'] + 
    df['productive_assets'] + 
    df['noncurrent_securities']) / df['assets'])
  
  aqi /= applier(aqi, 'shift')
  
  # Sales Growth Index
  sgi = df['revenue'] / applier(df['revenue'], 'shift')
  
  # Depreciation Index    
  depi = (df['depreciation'] / (
    df['productive_assets'] - df['depreciation'])
  )
  depi /= applier(depi, 'shift')
  
  # Sales General and Administrative Expenses Index       
  sgai = df['selling_general_administrative_expense'] / df['revenue']
  sgai /= applier(sgai, 'shift')
      
  # Leverage Index
  li = df['liabilities'] / df['assets']
  li /= applier(li, 'shift')
  
  # Total Accruals to Total Assets
  ocf = df['operating_cashflow']
  tata = (df['operating_income_loss'] - ocf) / df['totAst']
  
  # Beneish M-score
  df['beneish_m_score'] = (-4.84 +
    0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 
    0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * li
  )
  return df

# Weighted average cost of capital
'''
def wacc(
  df: pd.DataFrame, 
  quote_fetcher: partial[pd.DataFrame], 
  mktParser,
  rskFreeParser,
  tickerId='',
  betaPeriod=1,
  dbtMaturity=5
) -> pd.DataFrame: # yahoo: ^TNX/'^TYX; fred: DSG10
  
  if 'shDil' not in set(df.columns):
    raise ValueError

  # Load quotes
  startDate = df.index.get_level_values('date').min() - relativedelta(years=betaPeriod)
  startDate = dt.strftime(startDate, '%Y-%m-%d')

  if tickerId:
    dfOhlcv = getFinData(quoteParser, 'ohlcv.db', tickerId)
  else:
    dfOhlcv = quoteParser(startDate)

  if 'adjClose' in dfOhlcv.columns:
    dfOhlcv = dfOhlcv['adjClose']
  else:
    dfOhlcv = dfOhlcv['close']

  dfOhlcv.rename('close', inplace=True)
  dfOhlcv = dfOhlcv.resample('D').ffill()
  dfMkt = mktParser(startDate)['close'].resample('D').ffill()
  dfMkt.rename('market', inplace=True)
  dfRskFree = rskFreeParser(startDate).resample('D').ffill()

  if 'riskFree' not in set(dfRskFree.columns):
    raise ValueError

  dfReturns = pd.concat([dfOhlcv, dfMkt, dfRskFree], axis=1)#.dropna()
  cols = ['close', 'market']
  dfReturns[cols] = (dfReturns[cols].diff() / dfReturns[cols].abs().shift(1))
  dfReturns.dropna(inplace=True)

  if 'mktCap' not in set(df.columns):
    df['mktCap'] = df['shQuote'] * df['shDil']

  # Add capitalization class
  if df['mktCap'].isnull().all():
    df['capCls'] = df['totAst'].apply(
      lambda x: 'small' if x < 2e9 else 'large')
      
  else:
    df['capCls'] = df['mktCap'].apply(lambda x: 'small' 
      if x < 2e9 else 'large')
                                      
  df['crdtSprd'] = df.apply(lambda r: creditRating(r['intCvg'],
    r['capCls']), axis=1)
  
  beta = {
    'beta': [],
    'mktRet': [],
    'rfr': []
  }
  #delta = ttm['period'].apply(lambda x: relativedelta(years=-1) if x == 'A' 
  #                            else relativedelta(months=-3))

  ixDates = df.sort_values('date').index.get_level_values('date').unique()
  if dfReturns.index.min() > ixDates.min():
    ixDates = ixDates[ixDates.values > dfReturns.index.min()]
  
  for i in range(len(ixDates)):
    if i == 0:
      mask = (
        (dfReturns.index >= min(ixDates[i], dfReturns.index.min())) & 
        (dfReturns.index <= ixDates[i])
      )
    else:  
      mask = (dfReturns.index >= ixDates[i-1]) & (dfReturns.index <= ixDates[i])
    
    temp = dfReturns[mask]
    if not temp.empty:
      x = sm.add_constant(temp['market'])
      model = sm.OLS(temp['close'], x)
      results = model.fit()
      beta['beta'].append(results.params[-1])
      
      beta['mktRet'].append(temp['market'].mean() * 252)
      beta['rfr'].append(temp['riskFree'].mean() / 100)
        
    else:
      beta['beta'].append(np.nan)
      beta['mktRet'].append(np.nan)
      beta['rfr'].append(np.nan)        
  
  dfBeta = pd.DataFrame(data=beta, index=ixDates)
  df = df.reset_index().merge(dfBeta, on='date', how='left').set_index(['date', 'period'])

  df['betaLvr'] = df['beta'] * (1 + (1 - df['taxRate']) * 
    df['totDbt'] / df['totEqt'])
  
  # Cost of equity
  df['cstEqt'] = df['rfr'] + df['betaLvr'] * (
      df['mktRet'] - df['rfr'])
  # Cost of debt
  df['cstDbt'] = df['rfr'] + df['crdtSprd']
  
  intEx = df['intEx'].copy()
  if 'q' in df.index.get_level_values('period'):
    mask = (slice(None), 'q')
    intEx.loc[mask] = intEx.loc[mask].rolling(window=4, min_periods=4).sum()
    intEx = intEx.combine_first(df['intEx'] * 4)

  # Market value of debt
  df['mktValDbt'] = (intEx / df['cstDbt']) * (
    1 - (1 / (1 + df['cstDbt'])**dbtMaturity)) + (
      df['totDbt'] / (1 + df['cstDbt'])**dbtMaturity)

  df['wacc'] = (
    df['cstEqt'] * df['mktCap'] / (df['mktCap'] + df['mktValDbt']) +
    df['cstDbt'] * (1 - df['taxRate']) * df['totDbt'] / (
      df['mktCap'] + df['mktValDbt'])
  )
  excl = ['crdtSprd', 'mktRet', 'rfr']
  df = df[df.columns.difference(excl)]
  return 
'''

def credit_rating(icr: float, cap: Literal['small', 'large']):
  # ICR: Interest Coverage Ratio
  
  if np.isnan(icr):
    return np.nan
  
  elif np.isposinf(icr):
    return 0.004
  
  elif np.isneginf(icr):
    return 0.4
  
  icr_intervals = {
    'small': (
      -1e5, 0.5, 0.8, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 6, 7.5, 9.5, 12.5, 1e5),
    'large': (
      -1e5, 0.2, 0.65, 0.8, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4.25, 5.5, 6.5, 8.5, 1e5)
  }
      
  spread_list = (
    0.1512, 0.1134, 0.0865, 0.082, 0.0515, 0.0421, 0.0351, 0.024, 
    0.02, 0.0156, 0.0122, 0.0122, 0.0108, 0.0098, 0.0078, 0.0063
  )
  
  cap_intervals = icr_intervals[cap]

  if icr < cap_intervals[0]:
    spread = 0.2
      
  elif icr > cap_intervals[-1]:
    spread = 0.005
  
  else:
    for i in range(1, len(cap_intervals)):
      if icr >= cap_intervals[i-1] and icr < cap_intervals[i]:
        spread = spread_list[i-1]
        break
  
  return spread