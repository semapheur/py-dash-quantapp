from datetime import datetime as dt
from typing import cast

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series
import polars as pl


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


def combine_duplicate_columns(df: DataFrame) -> DataFrame:
  duplicated = df.columns.duplicated()

  if not duplicated.any():
    return df

  df_duplicated = combine_duplicate_columns(cast(DataFrame, df.loc[:, duplicated]))
  df = cast(DataFrame, df.loc[:, ~duplicated])

  for col in df_duplicated.columns:
    df.loc[:, col] = df[col].combine_first(df_duplicated[col])

  return df


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


def slice_polars_by_date(
  df: pl.DataFrame,
  date_column: str,
  start_date: dt | None = None,
  end_date: dt | None = None,
) -> pl.DataFrame:
  if df[date_column].dtype not in (pl.Datetime, pl.Date):
    raise ValueError("Dataframe must have a datetime column!")

  lf = df.lazy()
  if start_date is not None:
    lf = lf.filter(pl.col(date_column) >= pl.lit(start_date))

  if end_date is not None:
    lf = lf.filter(pl.col(date_column) <= pl.lit(end_date))

  return lf.collect()


def slice_pandas_by_date(
  df: pd.DataFrame,
  start_date: dt | None = None,
  end_date: dt | None = None,
) -> pd.DataFrame:
  if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("Dataframe must have a datetime index!")

  if start_date is not None:
    df = df.loc[df.index >= pd.Timestamp(start_date)]

  if end_date is not None:
    df = df.loc[df.index <= pd.Timestamp(end_date)]

  return df


def df_time_difference(
  dates: pd.DatetimeIndex, periods: int = 30, freq: str = "D"
) -> np.ndarray:
  return np.round(
    np.diff(dates.to_numpy(), prepend=[np.datetime64("nat", "D")])
    / np.timedelta64(periods, freq)
  )


def df_business_days(dates: pd.DatetimeIndex, fill: float = np.nan) -> Series[int]:
  dates_ = dates.to_numpy().astype("datetime64[D]")

  values = np.concatenate((np.array([fill]), np.busday_count(dates_[:-1], dates_[1:])))
  return pd.Series(values)
