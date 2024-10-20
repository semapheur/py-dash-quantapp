from numba import jit
import numpy as np

# https://colab.research.google.com/drive/1XtCNkpbfSoiMXpypcJ3DOzXBZy4jRzn_?usp=sharing#scrollTo=feJ1s39mkFEw
# https://www.youtube.com/watch?v=30uh1YBrsQ0


@jit(nopython=True)
def discount_cashflow(
  start_year: int,
  start_revenue: float,
  years: int | float,
  revenue_growth: float,
  operating_margin: float,
  tax_rate: float,
  reinvestment_rate: float,
  risk_free_rate: float,
  yield_spread: float,
  equity_risk_premium: float,
  equity_value_weight: float,
  beta: float,
) -> np.ndarray[float]:
  years = int(years)

  revenue = (
    start_revenue
    * np.array(np.power(1 + revenue_growth, start_year)).repeat(years).cumprod()
  )
  fcff = revenue * operating_margin * (1 - tax_rate) * (1 - reinvestment_rate)

  cost_debt = (risk_free_rate + yield_spread) * (1 - tax_rate)
  cost_equity = risk_free_rate + beta * equity_risk_premium

  cost_capital = (
    equity_value_weight * cost_equity + (1 - equity_value_weight) * cost_debt
  )

  dcf = fcff / np.array(np.power(1 - cost_capital, start_year)).repeat(years).cumprod()

  return np.array([years, revenue[-1], dcf.sum()])


def terminal_value(
  start_revenue: float,
  revenue_growth: float,
  operating_margin: float,
  tax_rate: float,
  reinvestment_rate: float,
  risk_free_rate: float,
  yield_spread: float,
  equity_risk_premium: float,
  equity_value_weight: float,
  beta: float,
) -> float:
  revenue = start_revenue * (1 + revenue_growth)
  fcff = revenue * operating_margin * (1 - tax_rate) * (1 - reinvestment_rate)

  cost_debt = (risk_free_rate + yield_spread) * (1 - tax_rate)
  cost_equity = risk_free_rate + beta * equity_risk_premium

  cost_capital = (
    equity_value_weight * cost_equity + (1 - equity_value_weight) * cost_debt
  )

  return fcff / (cost_capital - revenue_growth)


@jit(nopython=True)
def dcf(
  current_revenue: float,
  revenue_growths: np.ndarray[float],
  operating_margins: np.ndarray[float],
  tax_rates: np.ndarray[float],
  reinvestment_rates: np.ndarray[float],
  risk_free_rates: np.ndarray[float],
  yield_spreads: np.ndarray[float],
  equity_risk_premiums: np.ndarray[float],
  equity_value_weights: np.ndarray[float],
  betas: np.ndarray[float],
):
  revenue = current_revenue * (1 + revenue_growths).cumprod()
  ebit = revenue * operating_margins
  nopat = ebit * (1 - tax_rates)
  fcff = nopat * (1 - reinvestment_rates)

  cost_debt = (risk_free_rates + yield_spreads) * (1 - tax_rates)
  cost_equity = risk_free_rates + betas * equity_risk_premiums

  cost_capital = (
    equity_value_weights * cost_equity + (1 - equity_value_weights) * cost_debt
  )

  dcf = fcff / (1 - cost_capital).cumprod()

  return dcf.sum()
