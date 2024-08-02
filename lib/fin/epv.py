import numpy as np
from numba import jit


@jit(nopython=True)
def earnings_power(
  max_revenue: float,
  revenue_level: float,
  operating_margin: float,
  tax_rate: float,
  capex_margin: float,
  maintenance_rate: float,
  ddaa_margin: float,
  risk_free_rate: float,
  yield_spread: float,
  equity_risk_premium: float,
  equity_value_weight: float,
  beta: float,
) -> float:
  revenue = max_revenue * revenue_level
  nopat = revenue * operating_margin * (1 - tax_rate)
  normalized_profit = nopat + (0.5 * tax_rate) * ddaa_margin
  maintenance_capex = revenue * capex_margin * maintenance_rate
  adjusted_earning = normalized_profit - maintenance_capex

  cost_debt = (risk_free_rate + yield_spread) * (1 - tax_rate)
  cost_equity = risk_free_rate + beta * equity_risk_premium

  cost_capital = (
    equity_value_weight * cost_equity + (1 - equity_value_weight) * cost_debt
  )

  epv = adjusted_earning / cost_capital

  return epv


@jit(nopython=True)
def epv_monte_carlo(nSim, params, opmDst, waccDst):
  """
  Parameters:
  0: Revenue (rev)
  1: Operating margin (opm)
  2: Tax rate (taxRate)
  3: Depreciation and amortization (da)
  4: Selling, general and administration expenses (sgaEx)
  5: Research and development expenses (rdaEx)
  6: Maintenance capital expenditure (mxCapEx)
  """

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
