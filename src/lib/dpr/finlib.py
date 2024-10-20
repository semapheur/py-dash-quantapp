import numpy as np
import pandas as pd

# JIT
from numba import jit

from scipy.optimize import root_scalar #brentq fixed_point

# Statistical analysis
import statsmodels.api as sm

# Date
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import sqlalchemy as sqla

# Utils
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from operator import and_
import textwrap

from lib.foos import fuzzMatch

FINITEMS = {
  'incomeStatement': {
    'rvn': {
        'label': 'Revenue',
        'children': {
          'saleRvn': {
            'label': 'Sales Revenue',
            'sign': 1,
            'children': {
              'domSaleRvn': {
                  'label': 'Domestic Sales Revenue',
                  'sign': 1
              },
              'frgnSaleRvn': {
                  'label': 'Foreign Sales Revenue',
                  'sign': 1
              }
            } 
          }, 
          'finRvn': {
            'label': 'Financing Revenue',
            'sign': 1
          }, 
          'othRvn': {
            'label': 'Other Revenue',
            'sign': 1
          }
        } 
    },
    'rvnEx': {
      'label': 'Cost of Revenue',
      'abbreviation': 'COGS',
      'children': {
        'saleRvnEx': {
          'label': 'Cost of Sales Revenue',
          'sign': 1
        }, 
        'finRvnEx': {
          'label': 'Cost of Financing Revenue',
          'sign': 1
        }, 
        'othRvnEx': {
          'label': 'Cost of Other Revenue',
          'sign': 1
        }
      }
    },
    'grsPrft': {
      'label': 'Gross Profit',
      'formula': 'rvn - rvnEx'
    },
    'opEx': {
      'label': 'Operating Expense',
      'formula': 'rdEx + sgaEx + dda + oth'
    },
    'rdEx': {
      'label': 'Research & Development Expense',
    },
    'sgaEx': {
      'label': 'Selling, General & Administrative Expense',
    },
    'dda': {
      'label': 'Depreciation, Depletion & Amortization',
      'abbreviation': 'DDA'
    },
    'othOpEx': {
      'label': 'Other Operating Expense',
    },
    'opInc': {
      'label': 'Operating Income/Loss',
      'formula': 'grsPrft -opEx' 
    },
    'noOpInc': {
      'label': 'Non-Operating Income/Loss',
      'formula': 'finInc +fxInc +othNoOpInc'
    },
    'ebit': {
      'label': 'Earnings Before Interest & Taxes',
      'abbreviation': 'EBIT',
      'formula': 'opInc +noOpInc'
    },
    'ebitda': {
      'label': 'Earnings before Interest, Taxes & DDA',
      'abbreviation': 'EBITDA',
      'formula': 'ebit +dda'
    },
    'noRcrInc': {
      'label': 'Non-recurring Income',
      'formula': 'slAstGl +slInvGl +slBizGl'
    },
    'noRcrEx': {
      'label': 'Non-recurring Expense',
      'formula': 'maEx +rstrEx +noRcDrv +acqRd + astWoff +extgDbt +lglStlm +isrcStlm +acnChg +prvDbtAcnt'
    },
    'impr': {
      'label': 'Impairment',
      'formula': 'imprTgbAst +imprItgbAst +imprGw'
    },
    'ebt': {
      'label': 'Earnings Before Taxes',
      'abbreviation': 'EBT',
      'formula': 'ebit -intEx +noRcrGl'
    },
    'taxEx': {
      'label': 'Income Tax Expense',
    },
    'netIncCntOp': {
        'label': 'Net Income from Continuing Operations'
    },
    'netIncDscOp': {
      'label': 'Net Income from Discontinued Operations'
    },
    'netInc': {
      'label': 'Net Income',
      'formula': 'ebt - taxEx'
    },
    'sh': {
      'label': ' Weighted Average Shares Outstanding (basic)'
    },
    'shDil': {
      'label': 'Weighted Average Shares Outstanding (diluted)'
    },
    'eps': {
      'label': 'Earnings per Share (basic)',
      'abbreviation': 'BEPS',
      'formula': 'netInc /sh'
    },
    'epsDil': {
      'label': 'Earnings per Share (diluted)',
      'abbreviation': 'DEPS',
      'formula': 'netInc /shDil'
    }
  },
  'assetSheet': {
    'cce': {
      'label': 'Cash & Cash Equivalents',
      'abbreviation': 'CCE'
    },
    'stInv': {
      'label': 'Short-term Investments'
    },
    'cceStInv': {
      'label': 'Cash, Cash Equivalents & Short-term Investments',
      'formula': 'cce +stInv'
    },
    'acntRcv': {
      'label': 'Account Receivable'
    },
    'ivty': {
      'label': 'Inventory',
      'formula': 'ivtyRaw +ivtyWip +ivtyFg +othIvty'
    },
    'othCrtAst': {
      'label': 'Other Current Assets',
      'formula': 'astForSl +crtDfrAst +crtDfrTaxAst +taxRcv +crtDscOp + mscCrtAst'
    },
    'totCrtAst': {
      'label': 'Total Current Assets',
      'formula': 'cce +stInv +acntRcv +ivty +othCrtAst'
    },
    'ppe': {
      'label': 'Property, Plant & Equipment',
      'abbreviation': 'PPE',
      'formula': 'plnt +ppty +eqpt +compEqpt +trnsEqpt +cip +ls +othPpe -acmDprcPpe'
    },
    '': {
        
    },
    'totNoCrtAst': {
      'label': 'Total Non-Current Assets',
      'formula': 'ppe +ltInv +itgbAst +gw'
    },
    'totAst': {
      'label': 'Total Assets',
      'formula': 'totCrtAst +totNoCrtAst'
    },

  },
  'liabilitySheet': {},
  'equitySheet': {},
  'cashFlowStatement': {
    'opCf': {
      'label': 'Operational Cash Flow'
    },
    'invCf': {
      'label': 'Investing Cash Flow'
    },
    'finCf': {
      'label': 'Financial Cash Flow'
    },
    'freeCf': {
      'label': 'Free Cash Flow'
    },
    'freeCfFirm': {
      'label': 'Free Cash Flow to Firm'
    },
  } 
}

class Fundamentals():

  def __init__(self, tickerObj, tickerId=''):
    self._ticker = tickerObj
    self._tickerId = tickerId
    self._financials = None
    self._market = None
    self._riskFree = None

  @property
  def market(self):
    return self._market

  @market.setter
  def market(self, parser, startDate=None):

    if startDate is None:
      df = parser()

    else:
      df = parser(startDate)

    if 'close' not in set(df.columns):
      raise ValueError('DataFrame does not include a "close" column')

    df.rename(columns={'close': 'market'}, inplace=True)
    self._riskFree = df['market'].resample('D').ffill()

    @property
    def riskFree(self):
        return self._riskFree

    @riskFree.setter
    def riskFree(self, parser, startDate=None, colNames={}):

      if startDate is None:
        df = parser()

      else:
        df = parser(startDate)

      if 'close' not in set(df.columns):
        raise ValueError('DataFrame does not include a "close" column')

      df.rename(columns={'close': 'riskFree'}, inplace=True)
      self._riskFree = df['riskFree'].resample('D').ffill()

# Load financial statement item names
def finItemRenameDict(src) -> dict:
  path = Path.cwd() / 'data' / 'finItems.csv'
  
  df = pd.read_csv(path, index_col=0)
  df.reset_index(inplace=True)
  
  mask = df['source'] == src
  df = df.loc[mask]
  
  res = {
    k: v for k, v in zip(df['sourceLabel'], df['itemValue'])
  }  
  return res

def getFinItems(sliceCols, **kwargs):
  path = Path.cwd() / 'data' / 'finItems.csv'
  
  df = pd.read_csv(path, index_col=0)
  df.reset_index(inplace=True)

  conditions = []
  for col, value in kwargs.items():

    if isinstance(value, str):
      value = [value]

    conditions.append((df[col].isin(value)))
      
  mask = reduce(and_, conditions)

  return df.loc[mask, sliceCols]

def getFinData(parser, dbName, tblName, tickerId='', ix=['id', 'date'], updateInterval=0):
  '''Loads OHLCV and financial data for a given ticker. 
  The data gets scrapped and stored in a database.    
  '''  
  if tickerId:
    dbPath = Path().cwd() / 'data' / dbName
    engine = sqla.create_engine(f'sqlite:///{dbPath}')
    insp = sqla.inspect(engine)

    if not insp.has_table(tblName):
      df = parser()
      df.reset_index(inplace=True)
      df['id'] = tickerId
      df.set_index(ix, inplace=True)

      # Upsert to db
      upsertDb(df, dbName, tblName)
      
    else:
      query = f'SELECT * FROM "{tblName}" WHERE id = "{tickerId}"'
      df = pd.read_sql(query, con=engine, index_col=ix,
          parse_dates={'date': {'format': '%Y-%m-%d'}})

      if df.empty:
        df = parser() # Parse OHLCV
        df.reset_index(inplace=True)
        df['id'] = tickerId
        df.set_index(ix, inplace=True)

        # Upsert to db
        upsertDb(df, dbName, tblName)

      else:
        lastDate = df.index.get_level_values('date').max()
        if relativedelta(dt.now(), lastDate).days > updateInterval:
          # Parse OHLCV 
          dfNew = parser() # Parse OHLCV
          dfNew.reset_index(inplace=True)
          dfNew['id'] = tickerId
          dfNew.set_index(ix, inplace=True)

          if dfNew is not None:
            upsertDb(dfNew, dbName, tblName)

            df = pd.read_sql(query, con=engine, index_col=ix,
              parse_dates={'date': {'format': '%Y-%m-%d'}})

  else:
    df = parser()
    # Remove None
    if df is not None:
      df.fillna(np.nan, inplace=True)

  return df

# Deprecated function
def dprFinData(parser, dbName, tickerId='', ix=['id', 'date'], updateInterval=0): 

  if tickerId:
    dbPath = Path().cwd() / 'data' / dbName
    engine = sqla.create_engine(f'sqlite:///{dbPath}')
    insp = sqla.inspect(engine)

    if not insp.has_table(tickerId):
      df = parser()

      if df is not None:
        df.reset_index().to_sql(tickerId, con=engine, index=False)

    else:
      query = f'SELECT * FROM "{tickerId}"'
      df = pd.read_sql(query, con=engine, parse_dates={'date': {'format': '%Y-%m-%d'}}, index_col=ix)

      lastDate = df.index.get_level_values('date').max()
      if relativedelta(dt.now(), lastDate).days > updateInterval:
        # Parse OHLCV 
        dfNew = parser() # Parse OHLCV

        if dfNew is not None:
          df = df.combine_first(dfNew)

          diffCols = dfNew.columns.difference(df.columns).tolist()
          if diffCols:
            df = df.join(dfNew[diffCols], how='outer')

          df.reset_index().to_sql(tickerId, con=engine, if_exists='replace', index=False)
  
  else:
    df = parser()

  # Remove None
  if df is not None:
    df.fillna(np.nan, inplace=True)

  return df

# Weighted average cost of capital
def wacc(df, quoteParser, mktParser, rskFreeParser, tickerId='', betaPeriod=1, dbtMaturity=5): # yahoo: ^TNX/'^TYX; fred: DSG10
  # t: Weighted average debt maturity
  
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
  return df

def creditRating(icr, cap):
  # ICR: Interest Coverage Ratio
  
  if np.isnan(icr):
    return np.nan
  
  elif np.isposinf(icr):
    return 0.004
  
  elif np.isneginf(icr):
    return 0.4
  
  if  cap == 'small':
    icrIntervals = (-1e5, 0.5, 0.8, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 6,
                    7.5, 9.5, 12.5, 1e5)
      
  elif cap == 'large':
    icrIntervals = (-1e5, 0.2, 0.65, 0.8, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 
                    3, 4.25, 5.5, 6.5, 8.5, 1e5)
  
  spreadList = (0.1512, 0.1134, 0.0865, 0.082, 0.0515, 0.0421, 0.0351, 
                0.024, 0.02, 0.0156, 0.0122, 0.0122, 0.0108, 0.0098, 0.0078,
                0.0063)
  
  if icr < icrIntervals[0]:
    spread = 0.2
      
  elif icr > icrIntervals[-1]:
    spread = 0.005
  
  else:
    hit = True
    i = 1
    spread = 0
    while hit:
        
      if icr >= icrIntervals[i-1] and icr < icrIntervals[i]:
        hit = False
        spread = spreadList[i-1]
      
      i += 1
  
  return spread

def financialRatios(df):

  # Return on equity
  if 'roe' not in set(df.columns):
    df['roe'] = (df['netInc'] / 
      df['totEqt'].rolling(2, min_periods=0).mean())
  
  # Return on tangible common equity
  if 'tgbEqt' in set(df.columns):
    df['rote'] = (df['netInc'] / 
      df['tgbEqt'].rolling(2, min_periods=0).mean())
  
  # Return on assets
  if 'roa' not in set(df.columns):
    df['roa'] = (
      df['netInc'] / 
      df['totAst'].rolling(2, min_periods=0).mean())
      
  # Return on net assets
  df['rona'] = df['netInc'] / (
    (df['ppe'] + df['wrkCap']).rolling(2, min_periods=0).mean())
  
  # Return on invested capital 
  df['roic'] = df['netInc'] / (
    df['totEqt'] + df['totDbt'])
  
  # Cash return on capital invested
  df['croci'] = df['freeCf'] / (
    df['totEqt'].rolling(2, min_periods=0).mean() + 
    df['totDbt'].rolling(2, min_periods=0).mean())
  
  # Economic profit
  df['ecPrft'] = df['croci'] - df['wacc']
  
  # Return on capital employed (ebit/(total assets - current liabilities))
  
  # Sustainable growth rate
  df['sgr'] = df['roe'] * (1 - df['dvd'] / df['netInc'])
  
  # Growth efficiency
  df['grwEfc'] = (df['rvn'].diff() / df['rvn'].abs().shift(1)) / df['sgr'] 
  
  # Asset turnover
  df['astTovr'] = (df['rvn'] / df['totAst'].rolling(2, min_periods=0).mean())
  
  # Fixed asset turnover
  df['fixAstTovr'] = df['rvn'] / df['ppe'].rolling(2, min_periods=0).mean()
  
  # Inventory turnover
  if not (df['ivty'] == 0).all():
    df['ivtyTovr'] = df['rvn'] / df['ivty'].rolling(2, min_periods=0).mean()
  
  # Receivables turnover        
  df['rcvTovr'] = (df['rvn'] / df['acntRcv'].rolling(2, min_periods=0).mean())
      
  # Current ratio
  if 'crtRatio' not in set(df.columns):
    df['crtRatio'] = (
      df['totCrtAst'].rolling(2, min_periods=0).mean() / 
      df['totCrtLbt'].rolling(2, min_periods=0).mean())
  
  # Quick ratio
  df['qckRatio'] = ((
    df['cceStInv'].rolling(2, min_periods=0).mean() + 
    0.75 * df['acntRcv'].rolling(2, min_periods=0).mean() +
    0.5 * df['ivty']) / 
    df['totCrtLbt'])
  
  # Cash ratio
  df['cshRatio'] = df['cceStInv'] / df['totCrtLbt']
  
  # Debt ratio
  if 'dbtRatio' not in set(df.columns):
    df['dbtRatio'] = (
      df['totDbt'].rolling(2, min_periods=0).mean() / 
      df['totAst'].rolling(2, min_periods=0).mean())
  
  # Debt to equity ratio
  df['dbtEqtRatio'] = (
    df['totDbt'].rolling(2, min_periods=0).mean() / 
    df['totEqt'].rolling(2, min_periods=0).mean())
  
  # Cash flow to debt ratio
  df['cfDbtRatio'] = (df['opCf'] / 
                      df['totDbt'].rolling(2, min_periods=0).mean())
  
  # Gross profit margin
  if 'gpm' not in df.columns:
    df['gpm'] = (
      (df['rvn'] - df['rvnEx']) / df['rvn'])
  
  # Net profit margin
  if 'npm' not in df.columns:
    df['npm'] = df['netInc'] / df['rvn']
  
  # Operating profit margin
  if 'opm' not in df.columns:
      
    if 'opInc' not in df.columns:
      df['opInc'] = df['ebit']
    
    df['opm'] = df['opInc'] / df['rvn']
  
  # Income quality
  df['incQy'] = df['opCf'] / df['netInc']
  
  # Cash conversion cycle
      
  # Days inventory outstanding
  if (df['ivty'] == 0).all():
    df['dio'] = 0
  else:
    df['dio'] = 365 * (df['ivty'].rolling(2, min_periods=0).mean() / df['rvnEx'])
          
  # Days sales outstanding
  df['dso'] = 365 * (df['acntRcv'].rolling(2, min_periods=0).mean() / df['rvn'])
  
  # Operating cycle
  df['opCycle'] = (df['dso'] + df['dio'])
  
  # Days payable outstanding
  df['dpo'] = 365 * (
    df['acntPbl'].rolling(2, min_periods=0).mean() 
    / df['opEx'])
  
  df['ccc'] = df['dio'] + df['dso'] - df['dpo']
      
  if 'shDil' not in set(df.columns):
    if 'sh' in set(df.columns):
      df['shDil'] = df['sh']
    else:
      df['shDil'] = np.nan
      
  if 'epsDil' not in set(df.columns):
    if 'eps' in set(df.columns):
      df['epsDil'] = df['eps']
    else:    
      df['epsDil'] = df['netInc'] / df['shDil']
      
  # Dividend ratios
  df['dpr'] = df['dvd'] / df['netInc'] # Dividend payout ratio

  return df

def priceMultiples(df):

  # Data frequency
  span = relativedelta(df.index.get_level_values('date').max(), 
    df.index.get_level_values('date').min()).years
  freq = int(np.ceil(len(df) / span)) if span != 0 else 1

  # Price ratios
  if 'shQuote' not in set(df.columns) and df['shDil'].isna().all:
    raise ValueError 

  if 'mktCap' not in set(df.columns):
    df['mktCap'] = df['shQuote'] * df['shDil']

  df['dvdYld'] = (df['dvd'] / df['shDil']) / df['shQuote']

  df['peRatio'] = df['shQuote'] / df['epsDil']
  
  eps = df['epsDil'].where(df['epsDil'] != 0, 1e-6)
  epsg = (eps.diff(freq) / eps.abs().shift(freq)) * 100
  
  df['pegRatio'] = df['peRatio'] / epsg.where(epsg != 0, 1e-6)
  
  df['pbRatio'] = df['mktCap'] / (df['totEqt'])
  
  df['ptbRatio'] = df['mktCap'] / (df['tgbEqt'])
  
  df['psRatio'] = df['mktCap'] / df['rvn']
      
  df['pfcfRatio'] = df['mktCap'] / df['freeCf']
  
  # Price to net current asset value
  if 'prfEqt' not in df.columns:
    df['prfEqt'] = 0
  
  df['pncavRatio'] = df['mktCap'] / (df['totCrtAst'] - df['totLbt'] - 
                                      df['prfEqt'].fillna(0))
  
  # Price to Net Net Working Capital
  df['pnnwcRatio'] = df['mktCap'] / (df['cceStInv'] + 
    0.75 * df['acntRcv'] + 
    0.5 * df['ivty'] - df['totLbt'])
  
  # Price to Graham Value
  df['pgvRatio'] = df['shQuote'] / (
    np.sqrt(22.5 * df['epsDil'] * df['totEqt'] / df['shDil']))
  
  if 'mktValDbt' in set(df.columns):
    df['ev'] = df['mktCap'] + df['mktValDbt'] - df['cceStInv']

  else:
    df['ev'] = df['mktCap'] + df['totDbt'] - df['cceStInv']
  
  df['evRvn'] = df['ev'] / df['rvn']
  
  df['evEbitda'] = df['ev'] / df['ebitda']
  
  df['capExCf'] = df['capEx'] / df['opCf']

  # Earning power valuation
  df = epv(df)
  df['pepvRatio'] = df['shQuote'] / df['epvFairVal']

  # Price-projected cash flow growth
  vals = df[['mktCap', 'cceStInv', 'totDbt', 'freeCf', 'wacc', 'ev']]
  df['prjCfGrw'] = vals.apply(cfGrowth, axis=1)

  return df

# Market cap-projected cash flow growth
def cfGrowth(row, n=10,  gL=0.03):
    
  # Optimizing function
  def fn(gS, mc, cce, dbt, fcf, wacc, ev, n,  gL):
          
    npv = (mc - cce + dbt)
    
    cf = fcf
    for i in range(1, n+1):
      cf += np.sign(cf) * cf * gS / (1 + wacc)**i
    
    # Terminal value
    if wacc > gL:
      tv = (cf * (1 + gL)) / (wacc - gL) / (1 + wacc)**n
        
    else:
      tv = np.abs(ev * (1 + gL)**n) / (1 + wacc)**n
    
    cf += tv
    
    return cf - npv
  
  if row.isnull().any():
    return np.nan
  
  else:
      
    try:
      args = (row['mktCap'], row['cceStInv'], row['totDbt'],
              row['freeCf'], row['wacc'], row['ev'], n, gL)
      
      res = root_scalar(
        fn,
        bracket=[-10, 10],
        args=args,
        method='bisect', 
      )
  
      return res.root
    
    except:
        return np.nan

# Dividend growth
def ggm(df, fcPer=20, ltGrowth=0.03):
    
  # Weighted average cost of capital
  if df['wacc'].isnull().all():
    df['ggmFairVal'] = np.nan
    return df
  
  else:
    wacc = df['wacc'].ewm(span=len(df['wacc'].dropna())).mean()
  
  # Check if company pays dividends        
  mask = ~(df['dvd'] == 0)
  if df[mask].empty or df['div'].isnull().all():
    d = df['freeCf'].copy()
      
  else:
    d = df['dvd'].abs().copy()
  
  dG = d.diff() / d.abs().shift(1)
  
  meanGrowth = dG.ewm(span=len(dG.dropna())).mean()
  
  h = fcPer / 2 # Half-life
  
  ev = d * ((1 + meanGrowth) + h * (ltGrowth - meanGrowth)) / (wacc - 
                                                                meanGrowth)
          
  ev += (df['cceStInv'] - df['totDbt'])
  
  # Fair value
  df['ggmFairVal'] = ev / df['shDil']
              
  return df

@jit(nopython=True)
def dcfMonteCarlo(nSim, params, r0, tg, tgMpl, rgDst, waccDst):
  '''
  Parameters:
  0: Revenue (rev)
  1: Revenue expense (revEx)
  2: Operating expense (opEx)
  3: Depreciation and amortization (da)
  4: Interest expense (intEx)
  5: Tax expense (taxEx)
  6: Capital expenditure (capEx)
  7: Change in working capital (chgWorkCap)
  '''

  sim = np.zeros(nSim)
  
  for i in range(nSim):
      
    tmp = np.zeros(params.shape)
    
    # Forecasted revenue
    tmp[:,0] = r0 * (1 + params[:,0] + rgDst[i]).cumprod() 
    
    for j in range(1, params.shape[1]):
      tmp[:,j] = tmp[:,0] * params[:,j]
    
    ebit = tmp[:,0] - tmp[:,1:3].sum(axis=1)
    taxRate = tmp[:,5] / (ebit - tmp[:,4])
    freeCf = ebit * (1 - taxRate) + tmp[:,3] - tmp[:,6] - tmp[:,7]
            
    # Discounted cash flows
    dcf = 0
    for k in range(len(freeCf)):
      dcf += freeCf[k] / (1 + waccDst[i])**(k+1) # waccDst[i]
        
    # Terminal value
    cf = (freeCf[-1] + tmp[-1,7] - tmp[-1,3]) # D&A cannot exceed CAPEX
    
    if waccDst[i] > tg: # Gordon growth model
      tv = cf * (1 + tg) / (waccDst[i] - tg) / (1 + waccDst[i])**len(freeCf) 

    
    else: # Multiples methods
      tv = tgMpl * (ebit[-1] + tmp[-1,3]) / (1 + waccDst[i])**len(freeCf)
        
    # Future value
    sim[i] = dcf + tv
      
  return sim

@jit(nopython=True)
def epvMonteCarlo(nSim, params, opmDst, waccDst):
  '''
  Parameters:
  0: Revenue (rev)
  1: Operating margin (opm)
  2: Tax rate (taxRate)
  3: Depreciation and amortization (da)
  4: Selling, general and administration expenses (sgaEx)
  5: Research and development expenses (rdaEx)
  6: Maintenance capital expenditure (mxCapEx)
  '''
  
  sim = np.zeros(nSim)
      
  for i in range(nSim):
      
    # Normalized earning (growth adjustments)
    opm = params[1] + opmDst[i]
    ne = params[0] * (opm + params[4] + params[5])
    
    # Adjusted earnings
    ae = ne * (1 - params[2]) + (params[3] - params[6]) * params[0]
    
    # Earnings power
    epv = ae / waccDst[i] 
    
    sim[i] = epv
      
  return sim


# Piotroski F-score
def fScore(df):
    
  ocf = df['opCf']
  
  if 'shDil' not in df.columns:
    if 'sh' in df.columns:
      df['shDil'] = df['sh']
        
    else:
      df['shDil'] = np.nan

  df['fScore'] = (
    np.heaviside(df['roe'], 0) +
    np.heaviside(df['roa'].diff(), 0) +
    np.heaviside(ocf, 0) +
    np.heaviside(ocf/df['totAst'] - df['roa'], 0) +
    np.heaviside(df['totDbt'].diff(), 0) +
    np.heaviside(df['qckRatio'].diff(), 0) +
    np.heaviside(-df['shDil'].diff(), 1) +
    np.heaviside(df['gpm'].diff(), 0) + 
    np.heaviside(df['astTovr'].diff(), 0)
  )

  return df

def zScore(df):
    
  ta = df['totAst'].rolling(2, min_periods=0).mean()
  tl = df['totLbt'].rolling(2, min_periods=0).mean()
  s = df['rvn']
  wc = df['wrkCap']
  re = df['rtnErn'].rolling(2, min_periods=0).mean()
  ebitda = df['ebitda']
  e = df['totEqt']
  
  df['zScore'] = (1.2 * wc/ta + 1.4 * re/ta + 3.3 * ebitda/ta + s/ta + 
                  0.6 * e/tl)

  return df

def mScore(df):

  # Days Sales in Receivables Index
  dsri = df['acntRcv'] / df['rvn']
  dsri /= dsri.shift(1)
      
  # Gross Margin Index            
  gmi = df['gpm'].shift(1) / df['gpm'] 
      
  # Asset Quality Index    
  aqi = (1 - (df['wrkCap'] + df['ppe'] + df['ltInv']) / 
          df['totAst'])
  
  aqi /= aqi.shift(1)
  
  # Sales Growth Index
  sgi = df['rvn'] / df['rvn'].shift(1)
  
  # Depreciation Index    
  depi = (df['da'] / 
          (df['ppe'] - 
              df['da']))
  
  depi /= depi.shift(-1)
  
  # Sales General and Administrative Expenses Index 
  if 'sgaEx' in df.columns:
    sga = df['sgaEx']
  
  elif 'gaEx' in df.columns:
    sga = df['gaEx']
  
  elif 'swEx' in df.columns:
    sga = df['gaEx']

  else:
    sga = df['othOpEx']
      
  sgai = sga / df['rvn']
  
  sgai /= sgai.shift(1)
      
  # Leverage Index
  if 'totLbt' not in df.columns:
    df['totLbt'] = df['totCrtAst'] + df['totNoCrtLbt'] + df['othLbt']
  
  li = df['totLbt'] / df['totAst']
  
  li /= li.shift(1)
  
  # Total Accruals to Total Assets
  ocf = df['opCf']
  tata = (df['opInc'] - ocf) / df['totAst']
  
  # Beneish M-score
  df['mScore'] = (
    0.92 * dsri.fillna(0) + 0.528 * gmi.fillna(0) + 
    0.404 * aqi.fillna(0) + 0.892 * sgi.fillna(0) + 
    0.115 * depi.fillna(0) - 0.172 * sgai.fillna(0) + 
    4.679 * tata.fillna(0) - 0.327 * li.fillna(0)
  )
      
  return df

def getFundamentals(finParser, quoteParser, mktParser, rskFreeParser, tickerId='', betaPeriod=5):

  def helper(checkDate=None):

    # Get financials
    df = getFinData(finParser, 'financials.db', 'stock', tickerId, 
      ix=['id', 'date', 'period'], updateInterval=100)

    if df is None:
      return None

    if checkDate is not None:
      lastDate = df.index.get_level_values('date').max()
      if lastDate.date == checkDate.date: # New data not available
        return None

      # Convert columns to numeric type
      df = df.apply(pd.to_numeric, errors='coerce')

      if 'shQuote' not in set(df.columns):
        if tickerId:
          quotes = getFinData(quoteParser, 'ohlcv.db', tickerId)
        else:
          quotes = quoteParser()
        
        if 'adjClose' in quotes.columns:
          quotes = quotes['adjClose']
        else:
          quotes = quotes['close']

        quotes = quotes.resample('D').ffill()
        quotes.rename('shQuote', inplace=True)
        df = df.reset_index().merge(quotes, on='date', how='left').set_index(['date', 'period'])

      df = wacc(df, quoteParser, mktParser, rskFreeParser, tickerId, betaPeriod)
      if 'q' in df.index.get_level_values('period'):
        mask = (slice(None), 'q')
        df.loc[mask, 'wacc'] /= 4

      cols = set(df.columns)
      for i in ['ivty', 'acntRcv', 'acntPbl']:
        if i not in cols:
          df[i] = 0

      dfs = []
      funcs = [financialRatios, priceMultiples, fScore, mScore, zScore]
      
      for p in df.index.get_level_values('period').unique():
        temp = df.xs(p, level='period', drop_level=False)
        for f in funcs:
          temp = f(temp)
        dfs.append(temp)

      if len(dfs) > 1:        
        df = dfs.pop(0)
        df = df.combine_first(dfs[0])
      else:
        df = dfs[0]

      # Fundamental items
      cols = getFinItems('itemValue', sheet=['fundamentals']).tolist()
      cols = set(cols).intersection(set(df.columns))
      cols.add('shQuote')

      if tickerId:
        df['id'] = tickerId
        df.set_index('id', append=True, inplace=True)
        df = df.swaplevel(0)
        df = df.swaplevel()

      return df[list(cols)]

  if tickerId:
    dbPath = Path().cwd() / 'data' / 'fundamentals.db'
    engine = sqla.create_engine(f'sqlite:///{dbPath}')
    insp = sqla.inspect(engine)

    if not insp.has_table('stock'):
      # Get fundamentals
      df = helper()
      if df is not None:
        upsertDb(df, 'fundamentals.db', 'stock')

    else:
      # Get fundamentals
      query = f'SELECT * FROM stock WHERE id = "{tickerId}"'
      df = pd.read_sql(query, con=engine, index_col=['date', 'period'],
          parse_dates={'date': {'format': '%Y-%m-%d'}})
      
      if df.empty:
        df = helper()
        if df is not None:
          upsertDb(df, 'fundamentals.db', 'stock')

        else:
          lastDate = df.index.get_level_values('date').max()

          if relativedelta(dt.now(), lastDate).days > 100:
            # Update fundamentals
            dfNew = helper(lastDate)
            if dfNew is not None:
              upsertDb(dfNew, 'fundamentals.db', 'stock')
              df = pd.read_sql(query, con=engine, index_col=['date', 'period'],
                parse_dates={'date': {'format': '%Y-%m-%d'}})
              
  else:
    # Get fundamentals
    df = helper()

  return df

def mergeTickers(asset='stock'):
    
  fuzzyXchg = {
    'barrons': ['XBUD', 'XMAU', 'XMEX', 'XMIL','XTAE', 'XWAR', 'PINX:GREY', 'LTS'],
    'marketwatch': ['XBOG', 'XBOM', 'XBRA', 'XBUD', 'XCAI', 'XCNQ', 'XTSX', 'XNSE', 'XTAE', 
      'XPAR', 'XPRA', 'XFRA', 'XHAM', 'XBER', 'XMAL', 'XMAU', 'XMIL', 'XOSL', 'XWAR',
      'XSAU', 'XSAT', 'XSTO:XSAT', 'XNGM:XSTO' 'XSWX', 'XADS', 'PINX:GREY'],
    'investing': ['PINX:GREY', 'XWBO', 'XBAH', 'XDHA', 'XBRU', 'XBUL', 
      'NEOE:XTSE/XTSX', 'XBOG', 'XCYS', 'XPRA', 'XCSE', 'CHIX', 'XPAR', 
      'XMUN', 'XBER', 'XETR', 'XFRA', 'XHAM', 'XSTU', 'XMUN', 'XDUS', 
      'XBUD', 'XICE', 'XBOM', 'XNSE', 'XDUB', 'XMIL', 'XKUW', 'XLUX',
      'XKLS', 'XMEX', 'XAMS', 'XNSA', 'XOSL', 'XKAR', 'XWAR', 'DSMD', 
      'XBSE', 'XSES', 'XJSE', 'XMAD', 'XCOL', 'XSTO', 'XNGM', 'XSWX',
      'XTUN', 'XDFM', 'XLON' ],
    'yahoo': ['PINX:GREY', 'XBOM', 'XBUE', 'XMIL', 'XWAR', 'XWBO']
  }

  dbPath = Path().cwd() / 'data' / 'ticker.db'
  engine = sqla.create_engine(f'sqlite:///{dbPath}')
  query = textwrap.dedent('''
      SELECT morningstarId, ms.ticker AS morningstarTicker, ms.tickerTrim AS tickerTrim, 
          ms.name AS morningstarName, ms.morningstarMic AS morningstarMic
      FROM morningstarStock AS ms
  ''')
  df = pd.read_sql(query, con=engine)
  
  # Merge tables
  for src in fuzzyXchg.keys():
    tbl = f'{src}{asset.capitalize()}'

    srcXchg = src
    if src == 'marketwatch':
      srcXchg = 'barrons'

    xchgSplit = textwrap.dedent(f'''WITH split(mic, xchg, str) AS (
        SELECT e.morningstarMic, "", e.{srcXchg}Exchange||"/" FROM exchange AS e 
            WHERE e.{srcXchg}Exchange IS NOT NULL
        UNION ALL SELECT
        mic,
        substr(str, 0, instr(str, "/")),
        substr(str, instr(str, "/")+1)
        FROM split WHERE str != ""
      ) 
    ''')
    query = textwrap.dedent(f'''
      {xchgSplit}
      SELECT {tbl}.ticker AS {src}Ticker, {tbl}.tickerTrim AS tickerTrim, {tbl}.name AS {src}Name, 
      split.mic AS morningstarMic
      FROM {tbl}
      JOIN split
        ON {tbl}.exchange = xchg
    ''')
    dfTemp = pd.read_sql(query, con=engine)
    df = df.merge(
      dfTemp, on=['tickerTrim', 'morningstarMic'], how='outer')
      
  # Create index
  df.set_index(['morningstarMic', 'tickerTrim'], inplace=True)
      
  ## Fuzzy match
  for src, xchg in fuzzyXchg.items():
    for x in tqdm(xchg):
      x = [i.split('/') for i in x.split(':')]
      # Match candidates
      maskRight = (
        df[f'{src}Ticker'].isna() & 
        ~df['morningstarTicker'].isna() &
        df.index.get_level_values('morningstarMic').isin(x[-1])
      )
      dfRight = df[maskRight]
      # String to match
      maskLeft = (
        df['morningstarTicker'].isna() & 
        ~df[f'{src}Ticker'].isna() &
        df.index.get_level_values('morningstarMic').isin(x[-1])
      )
      dfLeft = df[maskLeft]
      
      dfMatch = fuzzMatch(
        dfLeft[f'{src}Name'], dfRight['morningstarName'],   
      ).drop('ratio', axis=1)
      
      # Merge matches
      dfMatch = dfMatch.merge(dfLeft[[f'{src}Name', f'{src}Ticker']], on=f'{src}Name')
      dfRight.drop([f'{src}Name', f'{src}Ticker'], axis=1, inplace=True)
      dfRight.reset_index(inplace=True)
      dfRight = dfRight.merge(dfMatch, on='morningstarName')
      dfRight.set_index(['morningstarMic', 'tickerTrim'], inplace=True)

      mask = (
        df[f'{src}Name'].isin(dfMatch[f'{src}Name'].dropna()) &
        df.index.get_level_values('morningstarMic').isin(x[-1])
      )
      df = df[~mask]
      df = df.combine_first(dfRight)
  
  df.reset_index(inplace=True)
  df.to_sql(asset, con=engine, index=False, if_exists='replace')