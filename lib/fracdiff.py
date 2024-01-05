import numpy as np
from scipy.fft import fft, ifft
from scipy.special import binom

def _frac_diff(x: np.ndarray, order: float, step: float) -> np.ndarray:

  n = len(x)
  max_j = min(n - 1, int(np.ceil(order)))

  j_values = np.arange(max_j + 1)
  binom_coef = np.array([binom(order, j) for j in j_values])

  diff_sum = np.zeros_like(x, dtype=float)
  for k in range(max_j, n):
    diff_sum[k] = np.sum(((-1)**j_values) * binom_coef * x[k - j_values])

  result = diff_sum / (step ** order)
  return result

def frac_diff(x: np.ndarray, d: float) -> np.ndarray:
  n = len(x)

  weights = np.zeros(n)
  weights[0] = -d

  for k in range(2, n):
    weights[k-1] = weights[k-2] * (k - 1 - d) / k

  result = np.copy(x)

  for i in range(n):
    dat = x[:i]
    w = weights[:i]
    result[i] = x[i] + np.dot(w, dat[::-1])

  return result

def fast_frac_diff(x: np.ndarray, d: float) -> np.ndarray:
  
  def next_pow2(n):
    return (n - 1).bit_length()
  
  n = len(x)
  fft_len = 2 ** next_pow2(2 * n - 1)
  prod_ids = np.arange(1, n)
  frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
  dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
  return np.real(dx[0:n])