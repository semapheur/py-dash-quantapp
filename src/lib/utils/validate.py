import math


def validate_currency(code: str) -> bool:
  import pycountry

  if pycountry.currencies.get(alpha_3=code) is None:
    return False

  return True


def normalize_nan(value: float) -> float:
  if math.isnan(value):
    return None

  return value
