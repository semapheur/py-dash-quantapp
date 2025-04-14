import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit
def calcululate_variance_ratios(
  time_series: NDArray[np.float64], min_lag_power: int, max_lag_power: int
) -> tuple[NDArray[np.float64], np.ndarray]:
  max_results = max_lag_power - min_lag_power
  tau_values = np.zeros(max_results)
  valid_lags = np.zeros(max_results, dtype=np.int64)
  valid_count = 0

  for i in range(min_lag_power, max_lag_power):
    lag = 2**i

    if lag >= len(time_series):
      continue

    diff = time_series[lag:] - time_series[:-lag]
    if len(diff) < 2:
      continue

    std = np.std(diff)
    if std == 0:
      continue

    tau_values[valid_count] = np.sqrt(std)
    valid_lags[valid_count] = lag
    valid_count += 1

  return tau_values[:valid_count], valid_lags[:valid_count]


def hurst_variance(time_series: NDArray[np.float64]) -> float:
  time_series = time_series[~np.isnan(time_series)]

  min_lag_power = 4
  max_lag_power = int(np.log2(len(time_series) / 2))
  if max_lag_power <= min_lag_power:
    return np.nan

  tau, lags = calcululate_variance_ratios(time_series, min_lag_power, max_lag_power)

  if len(tau) < 2:
    return np.nan

  log_lags = np.log10(lags)
  log_tau = np.log10(tau)
  model = np.polyfit(log_lags, log_tau, 1)

  hurst = model[0] * 2.0

  return np.clip(hurst, 0.0, 1.0)


@nb.njit(parallel=True)
def calculate_rescale_range(
  time_series: np.ndarray, lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  rs_values = np.zeros(len(lags))
  actual_lags = np.zeros(len(lags), dtype=np.int64)

  for lag_idx in nb.prange(len(lags)):
    lag = lags[lag_idx]
    n_segments = int(len(time_series) / lag)

    if n_segments == 0:
      rs_values[lag_idx] = np.nan
      continue

    rs_sum = 0.0
    count = 0

    for i in range(n_segments):
      segment = time_series[i * lag : (i + 1) * lag]

      if len(segment) < 2:
        continue

      std = np.std(segment)
      if std == 0:
        continue

      mean = np.mean(segment)
      z = np.cumsum(segment - mean)
      r = np.max(z) - np.min(z)
      rs = r / std

      rs_sum += rs
      count += 1

    if count > 0:
      rs_values[lag_idx] = rs_sum / count
      actual_lags[lag_idx] = lag
    else:
      rs_values[lag_idx] = np.nan

  return rs_values, actual_lags


@nb.jit
def valid_rs_values(
  rs_values: NDArray[np.float64], actual_lags: np.ndarray
) -> tuple[NDArray[np.float64], np.ndarray]:
  mask = ~np.isnan(rs_values)

  return rs_values[mask], actual_lags[mask]


def hurst_rescaled_range(time_series: NDArray[np.float64]) -> float:
  time_series = time_series[~np.isnan(time_series)]

  max_lag_power = int(np.log2(len(time_series) / 2))
  min_lag_power = 4
  if max_lag_power <= min_lag_power:
    return np.nan

  lags = np.array([2**i for i in range(min_lag_power, max_lag_power)])

  rs_values, actual_lags = calculate_rescale_range(time_series, lags)
  valid_rs, valid_lags = valid_rs_values(rs_values, actual_lags)

  if len(valid_rs) < 2:
    return np.nan

  log_lags = np.log(valid_lags)
  log_rs = np.log(rs_values)

  hurst = np.polyfit(log_lags, log_rs, 1)[0]

  return hurst
