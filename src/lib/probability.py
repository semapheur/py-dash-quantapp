import numpy as np
import openturns as ot
from scipy.special import gamma, loggamma
import scipy.stats as stt


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


def nearest_postive_definite_matrix(a: np.ndarray) -> np.ndarray:
  b = (a + a.T) / 2
  eigval, eigvec = np.linalg.eig(b)
  eigval[eigval < 0] = 0

  return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def make_distribution(name: str, params: list[float]):
  def check_params(
    length: int,
    params: list[float],
  ):
    if (n := len(params)) == length:
      return

    raise ValueError(
      f"The {name} distribution takes {length} parameters, however {n} were given!"
    )

  match name.lower():
    case "normal":
      check_params(2, params)
      dist = ot.Normal(*params)
    case "skewnormal":
      check_params(3, params)
      dist = ot.Distribution(
        ot.SciPyDistribution(stt.skewnorm(params[0], loc=params[1], scale=params[2]))
      )
    case "triangular":
      check_params(3, params)
      dist = ot.Triangular(*params)
    case "uniform":
      check_params(2, params)
      dist = ot.Uniform(*params)

  return dist


def plot_pdf(
  distribution: ot.Distribution, num_points=1000, quantile_range=(0.001, 0.999)
):
  lower_bound = distribution.computeQuantile(quantile_range[0])[0]
  upper_bound = distribution.computeQuantile(quantile_range[1])[0]

  x = np.linspace(lower_bound, upper_bound, num_points)
  y = np.array([distribution.computePDF(i) for i in x])

  return x, y
