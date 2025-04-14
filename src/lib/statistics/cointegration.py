import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


def rolling_cointegration(
  x: pd.Series, y: pd.Series, window: int = 120
) -> pd.DataFrame:
  betas = []
  pvals = []
  indices = []

  for i in range(len(x) - window + 1):
    x_win = x.iloc[i : i + window]
    y_win = y.iloc[i : i + window]

    # OLS regression to estimate beta
    model = sm.OLS(y_win, sm.add_constant(x_win)).fit()
    beta = model.params[1]

    # Engle-Granger test
    _, pval, _ = coint(y_win, x_win)

    betas.append(beta)
    pvals.append(pval)
    indices.append(x.index[i + window - 1])

  return pd.DataFrame({"beta": betas, "pval": pvals}, index=indices)
