from datetime import datetime as dt, date as Date, time
from dateutil.relativedelta import relativedelta
import json
import math
import re
from pathlib import Path
from typing import cast, Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series


def int_parser(value):
  try:
    return int(value)
  except (ValueError, TypeError):
    return value


def load_json(path: str | Path) -> dict:
  with open(path, "r") as f:
    return json.load(f)


def update_json(path: str | Path, data: dict):
  if isinstance(path, str):
    path = Path(path)

  if path.suffix != ".json":
    path = path.with_suffix(".json")

  try:
    with open(path, "r") as f:
      file_data = json.load(f)

  except (FileNotFoundError, json.JSONDecodeError):
    file_data = {}

  file_data.update(data)

  with open(path, "w") as f:
    json.dump(file_data, f)


def minify_json(path: str | Path, new_name: Optional[str] = None):
  if isinstance(path, str):
    path = Path(path)

  with open(path, "r") as f:
    data = json.load(f)

  if not new_name:
    new_path = path.with_name(f"{path.stem}_mini.json")
  else:
    new_path = path.with_name(new_name).with_suffix(".json")

  with open(new_path, "w") as f:
    json.dump(data, f, separators=(",", ":"))


def replace_all(text: str, replacements: dict[str, str]) -> str:
  for k, v in replacements.items():
    text = text.replace(k, v)
  return text


def camel_split(txt: str) -> list[str]:
  pattern = r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))"
  return re.findall(pattern, txt)


def camel_abbreviate(txt: str, chars: int = 2):
  words = camel_split(txt)
  words[0] = words[0].lower()
  words = [word[:chars] for word in words]
  return "".join(words)


def snake_abbreviate(txt: str, chars: int = 2):
  words = camel_split(txt)
  words = [word[:chars].lower() for word in words]
  return "_".join(words)


# Rename DataFrame columns
class renamer:
  def __init__(self):
    self.d = dict()

  def __call__(self, x: str):
    if x not in self.d:
      self.d[x] = 0
      return x
    else:
      self.d[x] += 1
      return "%s_%d" % (x, self.d[x])


def replace_dict_keys(pattern: str, replacement: str, data: dict) -> dict:
  new_data = {}
  for k, v in data.items():
    new_key = re.sub(pattern, "", k)
    new_data[new_key] = v

  return new_data


def insert_characters(string: str, inserts: dict[str, list[int]]):
  result = string
  offset = 0

  for char in inserts.keys():
    for pos in inserts[char]:
      result = result[: pos + offset] + char + result[pos + offset :]
      offset += len(char)

  return result


def combine_duplicate_columns(df: DataFrame) -> DataFrame:
  duplicated = df.columns.duplicated()

  if not duplicated.any():
    return df

  df_duplicated = combine_duplicate_columns(cast(DataFrame, df.loc[:, duplicated]))
  df = cast(DataFrame, df.loc[:, ~duplicated])

  for col in df_duplicated.columns:
    df.loc[:, col] = df[col].combine_first(df_duplicated[col])

  return df


def handle_date(date: dt | Date) -> dt:
  if isinstance(date, Date):
    date = dt.combine(date, time())

  return date


def slice_df_by_date(
  df: pd.DataFrame,
  start_date: Optional[dt | Date] = None,
  end_date: Optional[dt | Date] = None,
) -> pd.DataFrame:
  if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("Data frame must have a datetime index!")

  if start_date is not None:
    df = df.loc[df.index >= pd.Timestamp(start_date)]

  if end_date is not None:
    df = df.loc[df.index <= pd.Timestamp(end_date)]

  return df


def month_difference(date1: dt | Date, date2: dt | Date) -> int:
  delta = relativedelta(max(date1, date2), min(date1, date2))
  return delta.years * 12 + delta.months + round(delta.days / 30)


def df_time_difference(
  dates: pd.DatetimeIndex, periods: int = 30, freq: str = "D"
) -> np.ndarray[float]:
  return np.round(
    np.diff(dates.to_numpy(), prepend=[np.datetime64("nat", "D")])
    / np.timedelta64(periods, freq)
  )


def df_business_days(dates: pd.DatetimeIndex, fill: float = np.nan) -> Series[int]:
  dates_ = dates.to_numpy().astype("datetime64[D]")

  values = np.concatenate((np.array([fill]), np.busday_count(dates_[:-1], dates_[1:])))
  return pd.Series(values)


Quarter: TypeAlias = Literal["Q1", "Q2", "Q3", "Q4"]


def fiscal_quarter(date: dt, fiscal_month: int, fiscal_day: int) -> Quarter:
  condition = date.month < fiscal_month or (
    date.month == fiscal_month and date.day <= fiscal_day
  )
  fiscal_year = date.year - 1 if condition else date.year
  fiscal_start = dt(fiscal_year, fiscal_month, fiscal_day)
  months = month_difference(date, fiscal_start)

  return cast(Quarter, f"Q{math.ceil(months/3)}")


def fiscal_quarter_monthly(month: int, fiscal_end_month: Optional[int] = None) -> int:
  if fiscal_end_month is not None:
    month = 12 - ((fiscal_end_month - month) % 12)

  return ((month - 1) // 3) + 1


def month_end(year: int, month: int, unleap=True) -> int:
  import calendar

  if unleap and month == 2 and calendar.isleap(year):
    return 28

  return calendar.monthrange(year, month)[1]


def validate_currency(code: str) -> bool:
  import pycountry

  if pycountry.currencies.get(alpha_3=code) is None:
    return False

  return True


def get_constructor_args(class_obj) -> set[str]:
  import inspect

  # Get the constructor signature
  signature = inspect.signature(class_obj.__init__)
  return set(signature.parameters.keys())


def remove_words(
  strings: list[str], blacklist: list[str], escape: bool = False
) -> list[str]:
  pattern = r"(?!^)\b(?:{})\b".format(
    "|".join(map(re.escape, blacklist) if escape else blacklist)
  )
  return [
    re.sub(r"\s+", " ", re.sub(pattern, "", s, flags=re.I).strip()) for s in strings
  ]


def split_multiline(row: Series[str]):
  split_cells = []
  max_lines = 0
  for cell in row:
    if pd.isna(cell):
      split_cell = [None]
    else:
      split_cell = cell.split("\n")
    split_cells.append(split_cell)
    max_lines = max(max_lines, len(split_cell))

  result = np.full((max_lines, len(row)), None, dtype=object)

  for i, cell in enumerate(split_cells):
    result[-len(cell) :, i] = cell

  return pd.DataFrame(result, columns=row.index)
