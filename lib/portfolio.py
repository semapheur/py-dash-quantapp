from typing import Literal


import cvxpy as cp
import numpy as np
import pandas as pd
from pandera.typing import DataFrame


class Portfolio:
  def __init__(
    self,
    stock_prices: DataFrame,
    risk_free_return=0.03,
    period: Literal["a", "q", "m"] = "a",
  ):
    self.returns = stock_prices.pct_change().to_numpy()
    self.mean_returns: np.ndarray = self.returns.mean(axis=1)
    self.cov_matrix = np.cov(self.returns)
    self.down_cov_matrix = np.minimum(self.returns, 0)
    self.risk_free_return = risk_free_return / 252  # Daily return
    self.tickers = stock_prices.columns.tolist()
    self.num_assets = len(self.tickers)

    if period == "m":
      self.period = 5 * 4
    elif period == "q":
      self.period = 5 * 4 * 3
    elif period == "a":
      self.period = 252  # Annual trading days

    # Optimisation
    self._w = cp.Variable(self.num_assets)

  def set_risk_free_return(self, risk_free_return):
    self.risk_free_return = risk_free_return

  def expected_return(self, weights: np.ndarray) -> float:
    return self.mean_returns.dot(weights) * self.period

  def risk_variance(
    self, weights: np.ndarray, risk_ratio: Literal["sharpe", "sortino"]
  ) -> float:
    cov_matrix = self.cov_matrix if risk_ratio == "sharpe" else self.down_cov_matrix

    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(
      self.period
    )

  def risk_adjusted_return(self, expected_return: float, risk_variance: float) -> float:
    return (expected_return - self.risk_free_return) / risk_variance

  def performance(
    self, weights: np.ndarray[float], risk_ratio: Literal["sharpe", "sortino"]
  ) -> tuple[float, float, float]:
    expected_return = self.expected_return(weights)
    risk_variance = self.risk_variance(weights, risk_ratio)
    risk_adjusted_return = self.risk_adjusted_return(expected_return, risk_variance)

    return expected_return, risk_variance, risk_adjusted_return

  def optimize(
    self,
    problem: Literal["minrisk", "maxrar"],
    risk_ratio: Literal["sharpe", "sortino"],
  ):
    if risk_ratio == "sharpe":
      risk = cp.quad_form(self._w, self.cov_matrix)
    elif risk_ratio == "sortino":
      risk = cp.quad_form(self._w, self.down_cov_matrix)

    # Minimize risk
    if problem == "minrisk":
      constraints = [cp.sum(self._w) == 1, self._w >= 0]

    # Maximize risk
    elif problem == "maxrar":
      k = cp.Variable()

      constraints = [
        cp.sum((self.mean_returns - self.risk_free_return).T @ self._w) == 1,
        cp.sum(self._w) == k,
        k >= 0,
        self._w <= k,
        # (self._w/k) >= 0
      ]

    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    if problem == "maxrar":
      w = self._w.value / k.value
    else:
      w = self._w.value

    perf = np.array(self.performance(w, risk_ratio))
    cols = ["Return", "Risk", "Risk-adjusted return"] + self.tickers
    data = np.concatenate((perf, w))

    result = pd.DataFrame([data], columns=cols)

    return result

  def efficient_frontier(
    self, num_samples: int, risk_ratio: Literal["sharpe", "sortino"]
  ):
    gamma = cp.Parameter(nonneg=True)
    returns = self.mean_returns.T @ self._w
    if risk_ratio == "sharpe":
      risk = cp.quad_form(self._w, self.cov_matrix)
    elif risk_ratio == "sortino":
      risk = cp.quad_form(self._w, self.down_cov_matrix)

    constraints = [cp.sum(self._w) == 1, self._w >= 0]
    prob = cp.Problem(cp.Maximize(returns - gamma * risk), constraints)

    data = np.zeros((self.num_assets + 3, num_samples))

    gammaVals = np.linspace(0, 10, num=num_samples)
    for i in range(num_samples):
      gamma.value = gammaVals[i]
      prob.solve()

      data[0, i] = returns.value * self.period
      data[1, i] = np.sqrt(risk.value * self.period)
      data[2, i] = gamma.value

      for j in range(len(self._w.value)):
        data[j + 3, i] = self._w.value[j]

    cols = ["Return", "Risk", "Risk-adjusted return"] + self.tickers
    result = pd.DataFrame(data.T, columns=cols)

    return result

  def monte_carlo(self, num_sims: int, risk_ratio: Literal["sharpe", "sortino"]):
    def rand_weights(n):
      k = np.random.rand(n)
      return k / np.sum(k)

    data = np.zeros((self.num_assets + 3, num_sims))

    for s in range(num_sims):
      w = rand_weights(self.num_assets)
      perf = self.performance(w, risk_ratio)

      data[0, s] = perf[0]  # Return
      data[1, s] = perf[1]  # Risk
      data[2, s] = perf[2]  # Return-to-risk

      for j in range(len(w)):
        data[j + 3, s] = w[j]

    cols = ["Return", "Risk", "Risk-adjusted return"] + self.tickers
    result = pd.DataFrame(data.T, columns=cols)

    return result
