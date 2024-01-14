import numpy as np
from scipy.special import gamma, loggamma


class GGD:
  # Generalized Gaussian distribution

  @staticmethod
  def pdf(x: np.ndarray, loc: float, scale: float, shape: float) -> np.ndarray:
    return (shape / (2 * scale * gamma(1 / shape))) * np.exp(
      -((np.abs(x - loc) / scale) ** shape)
    )

  @staticmethod
  def logpdf(x: np.ndarray, loc: float, scale: float, shape: float) -> np.ndarray:
    return (
      np.log(shape / (2 * scale))
      - loggamma(1 / shape)
      - (np.abs(x - loc) / scale) ** shape
    )

  @staticmethod
  def rand(n: int, loc: float, scale: float, shape: float):
    inv_shape = np.reciprocal(shape)
    z = scale * np.random.gamma(inv_shape, scale=1, size=n)
    return np.where(z < 0.5, loc - z, loc + z)

  # @staticmethod
  # def quantile(q: float, loc: float, scale: float, shape: float):
  #  inv_shape = np.reciprocal(shape)
  #  r = 2 * q - 1
  #  z = scale * quantile(Gamma(inv_shape, 1), np.abs(r))**inv_shape
  #  return loc + copysign(z, r)
