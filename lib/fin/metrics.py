import numpy as np
import pandas as pd

from lib.fin.calculation import applier

def f_score(df: pd.DataFrame) -> pd.DataFrame:
    
  ocf = df['operating_cashflow']
  
  df['piotroski_f_score'] = (
    np.heaviside(df['return_on_equity'], 0) +
    np.heaviside(df['return_on_assets'].diff(), 0) +
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