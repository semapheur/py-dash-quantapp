import numpy as np
from scipy.fft import fft, ifft

def frac_diff(x, d: float):
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

def fast_frac_diff(x, d: float):
  
  def next_pow2(n):
    return (n - 1).bit_length()
  
  n = len(x)
  fft_len = 2 ** next_pow2(2 * n - 1)
  prod_ids = np.arange(1, n)
  frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
  dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
  return np.real(dx[0:n])