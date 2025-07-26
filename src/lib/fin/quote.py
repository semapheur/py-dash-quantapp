from datetime import date as Date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Annotated, Any, Coroutine, Literal

import pandas as pd
from pandera.typing import DataFrame
import polars as pl

from lib.db.lite import (
  fetch_sqlite,
  read_sqlite,
  upsert_sqlite,
  polars_from_sqlite,
  polars_to_sqlite_upsert,
)
from lib.fin.models import Quote
from lib.utils.dataframe import slice_pandas_by_date, slice_polars_by_date
from lib.utils.time import date_to_int

type Security = Literal["stock", "forex", "index"]
type QuoteColumn = Literal["open", "high", "low", "close", "volume"]

OHLCV_COLUMNS: list[QuoteColumn] = ["open", "high", "low", "close", "volume"]


async def load_ohlcv(
  table: str,
  security: Security,
  ohlcv_fetcher: partial[
    Coroutine[Any, Any, Annotated[pl.DataFrame, DataFrame[Quote]]]
  ],
  delta: int = 1,
  start_date: dt | Date | None = None,
  end_date: dt | Date | None = None,
  cols: list[QuoteColumn] = OHLCV_COLUMNS,
) -> Annotated[pl.DataFrame, DataFrame[Quote]]:
  db_path = f"{security}_quote.db"

  ohlcv = _load_existing_ohlcv(db_path, table, start_date, end_date, cols)

  if ohlcv is None or ohlcv.is_empty():
    return await _fetch_and_store_ohlcv(
      ohlcv_fetcher, table, security, start_date, end_date, cols
    )

  if _needs_update(ohlcv, end_date, delta, db_path, table):
    return await _update_and_merge_ohlcv(
      ohlcv, ohlcv_fetcher, table, security, start_date, end_date, cols
    )

  return cast(
    DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
  )


def _upsert_ohlcv(
  table: str,
  security: Security,
  df: pl.DataFrame,
) -> None:
  df = df.with_columns(
    pl.col("date").dt.strftime("%Y%m%d").cast(pl.Int32).alias("date")
  )
  polars_to_sqlite_upsert(df, f"{security}_quote.db", table, ("date",))


def _load_existing_ohlcv(
  db_path: str,
  table: str,
  start_date: dt | Date | None,
  end_date: dt | Date | None,
  cols: list[QuoteColumn],
) -> DataFrame[Quote] | None:
  col_text = "date, " + ", ".join(cols)
  query = f"SELECT {col_text} FROM '{table}'"
  params = {}

  conditions = []
  if start_date is not None:
    params["start_date"] = date_to_int(start_date)
    conditions.append("date >= :start_date")

  if end_date is not None:
    params["end_date"] = date_to_int(end_date)
    conditions.append("date <= :end_date")

  if conditions:
    query += " WHERE " + " AND ".join(conditions)

  date_transform = [
    pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date")
  ]

  return polars_from_sqlite(
    db_path, query, params=params, column_transform=date_transform
  )


async def _fetch_and_store_ohlcv(
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  table: str,
  security: Security,
  start_date: dt | Date | None,
  end_date: dt | Date | None,
  cols: list[QuoteColumn],
) -> DataFrame[Quote]:
  """Fetch all OHLCV data when no existing data is found."""
  try:
    ohlcv = await ohlcv_fetcher()
  except Exception as e:
    raise Exception(f"Failed to fetch OHLCV: {e}")

  if ohlcv is None or ohlcv.is_empty():
    raise Exception(f"No OHLCV data returned for {security}")

  _upsert_ohlcv(table, security, ohlcv.clone())

  # Select only requested columns
  ohlcv = ohlcv.select(["date"] + cols)

  return cast(
    DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
  )


def _needs_update(
  ohlcv: DataFrame[Quote],
  end_date: dt | Date | None,
  delta: int,
  db_path: str,
  table: str,
) -> bool:
  """Determine if existing data needs to be updated with newer data."""
  last_date = ohlcv["date"].max()

  # Check if we need data beyond what we have
  if end_date is not None:
    if last_date == end_date:
      return False

    # last_date < end_date
    query = f"""SELECT EXISTS (
      SELECT 1
      FROM '{table}'
      WHERE date > :last_date
    )"""
    return bool(fetch_sqlite(db_path, query, {"last_date": last_date})[0][0])

  today = dt.now().date() if isinstance(last_date, Date) else dt.now()
  if delta == 1 and today.weekday() >= 5:
    delta = 2

  if end_date is None and relativedelta(today, last_date).days > delta:
    return True

  return False


async def _update_and_merge_ohlcv(
  ohlcv: DataFrame[Quote],
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  table: str,
  security: Security,
  start_date: dt | Date | None,
  end_date: dt | Date | None,
  cols: list[QuoteColumn],
) -> DataFrame[Quote]:
  last_date = ohlcv["date"].max()

  try:
    new_ohlcv = await ohlcv_fetcher(start_date=last_date)
  except Exception as e:
    raise Exception(f"Failed to fetch updated OHLCV for {security}: {e}")

  if new_ohlcv is None or new_ohlcv.is_empty():
    return cast(
      DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
    )

  _upsert_ohlcv(table, security, new_ohlcv.clone())

  # Merge and deduplicate data
  merged_data = pl.concat([ohlcv, ohlcv]).unique(subset=["date"]).sort("date")

  # Select only requested columns
  merged_data = merged_data.select(["date"] + cols)

  return cast(
    DataFrame[Quote], slice_polars_by_date(merged_data, "date", start_date, end_date)
  )


def _upsert_ohlcv_pandas(
  table: str,
  security: Security,
  df: DataFrame[Quote],
) -> None:
  df.index = cast(pd.DatetimeIndex, df.index).strftime("%Y%m%d").astype(int)
  upsert_sqlite(df, f"{security}_quote.db", table)


async def load_ohlcv_pandas(
  table: str,
  security: Security,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: dt | Date | None = None,
  end_date: dt | Date | None = None,
  cols: list[QuoteColumn] | None = None,
) -> DataFrame[Quote]:
  cols_ = cols or ["open", "high", "low", "close", "volume"]
  col_text = "CAST(date AS TEXT) AS date, " + ", ".join(cols_)

  query = f"SELECT {col_text} FROM '{table}'"

  params = {}
  if start_date is not None:
    params["start_date"] = int(dt.strftime(start_date, "%Y%m%d"))
    query += " WHERE date >= :start_date"

  if end_date is not None:
    params["end_date"] = int(dt.strftime(end_date, "%Y%m%d"))
    query += f"{' AND' if 'WHERE' in query else ' WHERE'} date <= :end_date"

  ohlcv = read_sqlite(
    f"{security}_quote.db",
    query,
    params=params,
    index_col="date",
    date_parser={"date": {"format": "%Y%m%d"}},
  )

  if ohlcv is None or ohlcv.empty:
    try:
      ohlcv = await ohlcv_fetcher()
    except Exception as e:
      raise Exception(f"Failed to fetch OHLCV: {e}")

    _upsert_ohlcv_pandas(table, security, ohlcv.copy())
    if cols is not None:
      ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, cols])

    return cast(DataFrame[Quote], slice_pandas_by_date(ohlcv, start_date, end_date))

  if not isinstance(ohlcv.index, pd.DatetimeIndex):
    try:
      ohlcv.index = pd.to_datetime(ohlcv.index, format="%Y%m%d")
    except Exception as e:
      raise ValueError(f"Failed to parse index as datetime: {e}")

  last_date = ohlcv.index.max()
  today = dt.now()
  if delta == 1 and today.weekday() >= 5:
    delta = 2

  if relativedelta(today, last_date).days <= delta:
    return cast(DataFrame[Quote], slice_pandas_by_date(ohlcv, start_date, end_date))

  try:
    new_ohlcv = await ohlcv_fetcher(last_date)
  except Exception as e:
    raise Exception(f"Failed to fetch OHLCV: {e}")

  if new_ohlcv is None or new_ohlcv.empty:
    return slice_pandas_by_date(ohlcv, start_date, end_date)

  _upsert_ohlcv_pandas(table, security, new_ohlcv.copy())

  ohlcv = cast(
    DataFrame[Quote],
    pd.concat([ohlcv, new_ohlcv], axis=0).sort_index().drop_duplicates(),
  )
  if cols is not None:
    ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, cols])

  return cast(DataFrame[Quote], slice_pandas_by_date(ohlcv, start_date, end_date))


def load_ohlcv_batch_pandas(
  tables: list[str],
  security: Security,
  start_date: dt | Date | None = None,
  end_date: dt | Date | None = None,
  cols: list[QuoteColumn] | None = None,
) -> pd.DataFrame | None:
  cols_ = cols or ["open", "high", "low", "close", "volume"]
  col_text = "CAST(date AS TEXT) AS date, " + ", ".join(cols_)

  where_clause = ""
  params = {}
  if start_date is not None:
    params["start_date"] = int(dt.strftime(start_date, "%Y%m%d"))
    where_clause += " WHERE date >= :start_date"

  if end_date is not None:
    params["end_date"] = int(dt.strftime(end_date, "%Y%m%d"))
    where_clause += (
      f"{' AND' if 'WHERE' in where_clause else ' WHERE'} date <= :end_date"
    )

  dfs: list[pd.DataFrame] = []
  for table in tables:
    query = f"SELECT {col_text} FROM '{table}' {where_clause}"

    df = read_sqlite(
      f"{security}_quote.db",
      query,
      params=params,
      index_col="date",
      date_parser={"date": {"format": "%Y%m%d"}},
    )

    if df is None or df.empty:
      continue

    if cols is not None and len(cols) == 1:
      df.rename(columns={cols[0]: table}, inplace=True)
    else:
      df.columns = pd.MultiIndex.from_product([[table], df.columns])
    dfs.append(df)

  if len(dfs) == 0:
    return None

  return pd.concat(dfs, axis=1).sort_index()
