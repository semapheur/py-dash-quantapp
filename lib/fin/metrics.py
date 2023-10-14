from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    np.heaviside(-applier(df['weighted_average_shares_outstanding_basic'], 'diff'), 1) +
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
  tata = (df['operating_income_loss'] - ocf) / df['average_assets']
  
  # Beneish M-score
  df['beneish_m_score'] = (-4.84 +
    0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 
    0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * li
  )
  return df

def beta(
  _id: str,
  fin: pd.DataFrame,
  quote: pd.Series | partial[pd.DataFrame], 
  market_fetcher: partial[pd.DataFrame],
  riskefree_fetcher: partial[pd.DataFrame], # yahoo: ^TNX/'^TYX; fred: DSG10
  period: int = 1,
) -> pd.DataFrame:
  from lib.ticker.fetch import get_ohlcv

  def calculate_beta(
    dates: pd.DatetimeIndex, 
    returns: pd.DataFrame,
    months: int
  ) -> pd.DataFrame:
    
    days = {
      3: 63.,
      12: 252.
    }

    beta = np.full(len(dates), np.nan)
    market_return = np.full(len(dates), np.nan)
    risk_free_rate = np.full(len(dates), np.nan)

    if returns.index.min() > dates.min():
      dates = dates[dates.values > returns.index.min()]
    
    for i in range(len(dates)):
      if i == 0:
        mask = (
          (returns.index >= min(dates[i], returns.index.min())) & 
          (returns.index <= dates[i])
        )
      else:  
        mask = (returns.index >= dates[i-1]) & (returns.index <= dates[i])
      
      temp: pd.DataFrame = returns.loc[mask]
      if temp.empty:
        continue

      x = sm.add_constant(temp['market_return'])
      model = sm.OLS(temp['equity_return'], x)
      ols_result = model.fit()
      beta[i] = ols_result.params.iloc[-1]
      
      market_return[i] = temp['market_return'].mean() * period
      risk_free_rate[i] = temp['risk_free_rate'].mean()

    result = pd.DataFrame(data={
      'date': dates, 
      'beta': beta,
      'market_return': market_return, 
      'risk_free_rate': risk_free_rate
    })
    result['months'] = months

    result.loc[:, 'market_return'] *= days[period]
    
    if months == 3:
      result.loc[:, 'risk_free_rate'] = result['risk_free_rate'] / 4  

    return result
  
  start_date = (fin.index.get_level_values('date').min() - 
    relativedelta(years=period))
  start_date = dt.strftime(start_date, '%Y-%m-%d')

  if isinstance(quote, partial):
    quote: pd.DataFrame = get_ohlcv(_id, 'stock', quote, cols={'date', 'close'})
    quote.rename(columns={'close': 'equity_return'}, inplace=True)
    quote.set_index('date', inplace=True)

  market = market_fetcher(start_date)
  market = market['close'].rename('market_return')

  riskfree = riskefree_fetcher(start_date)
  riskfree = riskfree['close'].rename('risk_free_rate')
  riskfree /= 100

  returns: pd.DataFrame = pd.concat([quote, market, riskfree], axis=1).ffill()

  cols = ['equity_return', 'market_return']
  returns.loc[:, cols] = returns[cols].pct_change() 
  #returns.loc[:, cols] = returns[cols].diff() / returns[cols].abs().shift(1)
  returns.dropna(inplace=True)
  
  betas: list[pd.DataFrame] = []
  for period in (3, 12):
    mask = (slice(None), slice(None), period)
    dates = (fin.loc[mask, :]
      .sort_index(level='date')
      .index.get_level_values('date')
    )
    betas.append(calculate_beta(dates, returns, period))

  beta = pd.concat(betas)
  fin = (fin
    .reset_index()
    .merge(beta, on=['date', 'months'], how='left')
    .set_index(['date', 'period', 'months'])
  )
  return fin

# Weighted average cost of capital
def weighted_average_cost_of_capital(
  fin: pd.DataFrame, 
  debt_maturity: int = 10
) -> pd.DataFrame:

  if 'beta' not in set(fin.columns):
    return fin

  fin.loc[:, 'capitalization_class'] = fin['market_capitalization'].apply(
    lambda x: 'small' if x < 2e9 else 'large'
  )
      
  fin.loc[:, 'yield_spread'] = fin.apply(
    lambda r: yield_spread(
      r['interest_coverage_ratio'],
      r['capitalization_class'])
    , axis=1)
  
  fin.loc[(slice(None), slice(None), 3), 'yield_spread'] /= 4
  
  fin['beta_levered'] = fin['beta'] * (1 + (1 - fin['tax_rate']) * 
    fin['debt'] / fin['equity'])
  
  # Cost of equity
  fin['equity_risk_premium'] = fin['beta_levered'] * (
    fin['market_return'] - fin['risk_free_rate'])
  fin['cost_equity'] = fin['risk_free_rate'] + fin['equity_risk_premium']

  # Cost of debt
  fin['cost_debt'] = fin['risk_free_rate'] + fin['yield_spread']
  
  # Market value of debt
  fin['market_value_debt'] = (fin['interest_expense'] / 
    fin['cost_debt']) * (
      1 - (1 / (1 + fin['cost_debt'])**debt_maturity)) + (
      fin['debt'] / (1 + fin['cost_debt'])**debt_maturity)
  
  fin['equity_to_capital'] = (fin['market_capitalization'] / 
    (fin['market_capitalization'] + fin['market_value_debt'])
  )

  fin['weighted_average_cost_of_capital'] = (
    fin['cost_equity'] * fin['equity_to_capital'] +
    fin['cost_debt'] * (1 - fin['tax_rate']) * (
      fin['market_value_debt'] / (
        fin['market_capitalization'] + 
        fin['market_value_debt']))
  )
  excl = ['market_return', 'capitalization_class']
  fin = fin[fin.columns.difference(excl)]
  return fin

def yield_spread(icr: float, cap: Literal['small', 'large']):
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

def discounted_cash_flow(
  df: pd.DataFrame, 
  fc_period: int = 20, 
  longterm_growth: float = 0.03
) -> pd.DataFrame:
    
  # Weighted average cost of capital
  if df['weight_average_cost_of_capital'].isnull().all():
    df['discounted_cashflow_value'] = np.nan
    return df
  
  else:
    wacc = df['weight_average_cost_of_capital'].ewm(
      span=len(df['weight_average_cost_of_capital'])).mean()
  
  # Free cash flow growth
  fcf = df['free_cash_flow_firm']
  fcf_roc = fcf.diff() / fcf.abs().shift(1)

  fcf_growth = fcf_roc.ewm(span=len(fcf_roc.dropna())).mean()
          
  dcf = np.zeros(len(df))
  x = np.arange(1, fc_period + 1)
  for i, cfg in enumerate(fcf_growth):
    growth_projection = (
      longterm_growth + (cfg - longterm_growth) / 
      (1 + np.exp(np.sign(cfg - longterm_growth) * (x - fc_period/2)))
    )
    
    cf = fcf[i]
    for j, g in zip(x, growth_projection):
      cf += np.abs(cf) * g
      present = cf / ((1 + wacc[i])**j)
      dcf[i] += present
        
    if wacc[i] > longterm_growth:
      terminal = np.abs(present) * (1 + longterm_growth) / (wacc[i] - longterm_growth)
    else:
      terminal = np.abs(df['ev'].iloc[i] * (1 + longterm_growth)**fc_period)
                    
    terminal /= (1 + wacc[i])**fc_period
    dcf[i] += terminal
      
  dcf += (df['liquid_assets'] - df['debt'])
  
  df['discounted_cashflow_value'] = (
    dcf / df['split_adjusted_weighted_average_shares_outstanding_basic'])
  #df['price_to_discounted_cashflow'] = (
  # df['share_price'] / df['discounted_cashflow_value'])
      
  return df

def earnings_power_value(df: pd.DataFrame) -> pd.DataFrame:
    
  # Weighted average cost of capital
  if df['weighted_average_cost_of_capital'].isnull().all():
    df['earnings_power_value'] = np.nan
    return df
  
  else:
    wacc = df['weighted_average_cost_of_capital'].ewm(
      span=len(df['weighted_average_cost_of_capital'])).mean()
  
  # Sustainable revenue
  rev = df['revenue'].dropna()
  if len(rev) == 0:
    df['earnings_power_value'] = np.nan
    return df

  sust_rev = rev.ewm(span=len(rev)).mean()
  
  # Tax rate
  tax = df['tax_rate']
  
  # Adjusted depreciation
  ad = 0.5 * tax * df['depreciation_depletion_amortization_accretion']
  
  # Maintenance capex
  rev_growth = rev.diff() / rev.abs().shift(1)
  
  if rev_growth.dropna().empty:
    maint_capex = df['depreciation_depletion_amortization_accretion'] / sust_rev
  
  else:
    capex = df['capital_expenditure'].ewm(span=len(df['capital_expenditure'])).mean()
    capex_margin = capex / sust_rev
    rev_growth = rev_growth.ewm(span=len(rev_growth.dropna())).mean()    
    maint_capex = capex_margin * (1 - rev_growth)
    # maint_capex = capex - (capex_margin * rev_growth)
      
  # Operating margins
  om = (df['pretax_income_loss'] + df['interest_expense']) / df['revenue'] 
  
  om_mean = om.ewm(span=len(om.dropna())).mean()
  
  # Adjusted earning
  adj_earn = sust_rev * om_mean * (1 - tax) + ad - (maint_capex * sust_rev)
      
  # Earnings power
  epv = adj_earn / wacc
    
  epv += df['liquid_assets'] - df['debt']
  
  # Fair value
  df['earnings_power_value'] = epv / df['shDil']
  
  return df