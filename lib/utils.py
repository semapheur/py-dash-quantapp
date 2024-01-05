from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import math
import re
from pathlib import Path
from typing import cast, Literal, Optional, TypeAlias

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.const import HEADERS


def int_parser(value):
  try:
    return int(value)
  except (ValueError, TypeError):
    return value


def load_json(path: str | Path) -> dict:
  with open(path, 'r') as f:
    return json.load(f)


def update_json(path: str | Path, data: dict):
  if isinstance(path, str):
    path = Path(path)

  if path.suffix != '.json':
    path = path.with_suffix('.json')

  try:
    with open(path, 'r') as f:
      file_data = json.load(f)

  except (FileNotFoundError, json.JSONDecodeError):
    file_data = {}

  file_data.update(data)

  with open(path, 'w') as f:
    json.dump(file_data, f)


def minify_json(path: str | Path, new_name: Optional[str] = None):
  if isinstance(path, str):
    path = Path(path)

  with open(path, 'r') as f:
    data = json.load(f)

  if not new_name:
    new_path = path.with_name(f'{path.stem}_mini.json')
  else:
    new_path = path.with_name(new_name).with_suffix('.json')

  with open(new_path, 'w') as f:
    json.dump(data, f, separators=(',', ':'))


def replace_all(text, dic):
  for i, j in dic.items():
    text = text.replace(i, j)
  return text


def camel_split(txt: str) -> list[str]:
  pattern = r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))'
  return re.findall(pattern, txt)


def camel_abbreviate(txt: str, chars: int = 2):
  words = camel_split(txt)
  words[0] = words[0].lower()
  words = [word[:chars] for word in words]
  return ''.join(words)


def snake_abbreviate(txt: str, chars: int = 2):
  words = camel_split(txt)
  words = [word[:chars].lower() for word in words]
  return '_'.join(words)


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
      return '%s_%d' % (x, self.d[x])


def insert_characters(string: str, inserts: dict[str, list[int]]):
  result = string
  offset = 0

  for char in inserts.keys():
    for pos in inserts[char]:
      result = result[: pos + offset] + char + result[pos + offset :]
      offset += len(char)

  return result


def combine_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
  duplicated = df.columns.duplicated()

  if not duplicated.any():
    return df

  df_duplicated = combine_duplicate_columns(df.loc[:, duplicated])
  df = df.loc[:, ~duplicated]

  for col in df_duplicated.columns:
    df.loc[:, col] = df[col].combine_first(df_duplicated[col])

  return df


def month_difference(date1: dt, date2: dt) -> int:
  delta = relativedelta(max(date1, date2), min(date1, date2))
  return delta.years * 12 + delta.months + round(delta.days / 30)


def df_time_difference(dates: pd.DatetimeIndex, periods: int = 30, freq: str = 'D'):
  return np.round(
    np.diff(dates.to_numpy(dtype=np.datetime64)) / np.timedelta64(periods, freq)
  )


def df_business_days(dates: pd.DatetimeIndex, fill: float = np.nan) -> pd.Series:
  dates_ = dates.to_numpy().astype('datetime64[D]')

  values = np.concatenate((np.array([fill]), np.busday_count(dates_[:-1], dates_[1:])))
  return pd.Series(values)


Quarter: TypeAlias = Literal['Q1', 'Q2', 'Q3', 'Q4']


def fiscal_quarter(date: dt, fiscal_month: int, fiscal_day: int) -> Quarter:
  condition = date.month < fiscal_month or (
    date.month == fiscal_month and date.day <= fiscal_day
  )
  fiscal_year = date.year - 1 if condition else date.year
  fiscal_start = dt(fiscal_year, fiscal_month, fiscal_day)
  months = month_difference(date, fiscal_start)

  return cast(Quarter, f'Q{math.ceil(months/3)}')


def download_file(url: str, file_path: str | Path):
  with open(file_path, 'wb') as file:
    with httpx.stream('GET', url=url, headers=HEADERS) as response:
      total = int(response.headers.get('content-length', 0))
      if not total:
        print(response.headers)
        raise Exception('Download failed!')

      with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit='B') as progress:
        bytes_downloaded = response.num_bytes_downloaded
        for chunk in response.iter_bytes():
          file.write(chunk)
          progress.update(response.num_bytes_downloaded - bytes_downloaded)
          bytes_downloaded = response.num_bytes_downloaded
