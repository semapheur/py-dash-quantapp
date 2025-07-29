from datetime import datetime as dt
from typing import cast, Literal

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series
import polars as pl

type Frequency = Literal["D", "H", "M", "S"]


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


def lower_and_coalesce_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
  columns = lf.collect_schema().names()
  name_map: dict[str, list[str]] = {}

  for column in columns:
    key = column.lower()
    name_map.setdefault(key, []).append(column)

  combined_exprs: list[pl.Expr] = []

  for key, columns in name_map.items():
    if len(columns) == 1:
      combined_exprs.append(pl.col(columns[0]).alias(key))

    else:
      exprs = [pl.col(c) for c in columns]
      combined_exprs.append(pl.coalesce(exprs).alias(key))

  return lf.select(combined_exprs)


def rename_and_coalesce_columns(
  lf: pl.LazyFrame,
  rename_map: dict[str, list[str]],
) -> pl.LazyFrame:
  column_cache = set(lf.collect_schema().names())

  exprs: list[pl.Expr] = []
  processed_cols: set[str] = set()

  for new_name, old_names in rename_map.items():
    found_names = list(column_cache.intersection(old_names))

    if not found_names:
      continue

    if len(found_names) == 1:
      exprs.append(pl.col(found_names[0]).alias(new_name))
    else:
      exprs.append(pl.coalesce(pl.col(found_names)).alias(new_name))

    processed_cols.update(found_names)

  for col in column_cache.difference(processed_cols):
    exprs.append(pl.col(col))

  return lf.select(exprs)


def combine_duplicate_columns_pandas(df: DataFrame) -> DataFrame:
  duplicated = df.columns.duplicated()

  if not duplicated.any():
    return df

  df_duplicated = combine_duplicate_columns_pandas(
    cast(DataFrame, df.loc[:, duplicated])
  )
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
  dates_column: str, periods: int = 30, freq: Frequency = "D"
) -> pl.Expr:
  if freq == "D":
    period_duration = pl.duration(days=periods)
  elif freq == "H":
    period_duration = pl.duration(hours=periods)
  elif freq == "M":
    period_duration = pl.duration(minutes=periods)
  elif freq == "S":
    period_duration = pl.duration(seconds=periods)
  else:
    raise ValueError(f"Unsupported frequency: {freq}")

  return (
    pl.col(dates_column)
    .diff(period_duration)
    .dt.total_nanoseconds()
    .truediv(period_duration.dt.total_nanoseconds())
    .round(0)
  )


def df_time_difference_pandas(
  dates: pd.DatetimeIndex, periods: int = 30, freq: str = "D"
) -> np.ndarray:
  return cast(
    np.ndarray,
    np.round(
      np.diff(dates.to_numpy(), prepend=[np.datetime64("nat", "D")])
      / np.timedelta64(periods, freq)
    ),
  )


def df_business_days(dates: pd.DatetimeIndex, fill: float = np.nan) -> Series[int]:
  dates_ = dates.to_numpy().astype("datetime64[D]")

  values = np.concatenate((np.array([fill]), np.busday_count(dates_[:-1], dates_[1:])))
  return pd.Series(values)


def fiscal_quarter_monthly_polars(
  date_col: pl.Expr, fiscal_end_month: int | None = None
) -> pl.Expr:
  month = date_col.dt.month()

  if fiscal_end_month is not None:
    adjusted_month = 12 - ((fiscal_end_month - month) % 12)
  else:
    adjusted_month = month

  return ((adjusted_month - 1) // 3) + 1
