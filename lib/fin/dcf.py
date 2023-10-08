import numpy as np
import openturns as ot
import scipy.stats as stt

# https://colab.research.google.com/drive/1XtCNkpbfSoiMXpypcJ3DOzXBZy4jRzn_?usp=sharing#scrollTo=feJ1s39mkFEw
# https://www.youtube.com/watch?v=30uh1YBrsQ0

def make_distribution(name: str, params: list[float]):

  def check_params(length: int, params: list[float],):
    if (n := len(params)) == length:
      return
    
    raise ValueError(
      f'The {name} distribution takes {length} parameters, however {n} were given!')

  match name.lower():
    case 'normal':
      check_params(2, params)
      dist = ot.Normal(*params)
    case 'skewnormal':
      check_params(3, params)
      dist = ot.Distribution(
        ot.SciPyDistribution(stt.skewnorm(
          params[0], loc=params[1], scale=params[2])))
    case 'triangular':
      check_params(3, params)
      dist = ot.Triangular(*params)
    case 'uniform':
      check_params(2, params)
      dist = ot.Uniform(*params)

  return dist

def dcf(
  present_revenue: float,
  revenue_growths: np.ndarray[float],
  operating_margins: np.ndarray[float]
):

  revenue = present_revenue * (1 + np.array([revenue_growths])).cumprod()
  operating_income = revenue * operating_margins
  #pretax_income

  return