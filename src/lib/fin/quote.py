from datetime import date as Date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Any, Coroutine, Literal

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

type Security = Literal["stock", "forex", "index"]
type QuoteColumn = Literal["open", "high", "low", "close", "volume"]


def upsert_ohlcv(
  table: str,
  security: Security,
  df: pl.DataFrame,
) -> None:
  df = df.with_columns(
    pl.col("date").dt.strftime("%Y%m%d").cast(pl.Int32).alias("date")
  )
  polars_to_sqlite_upsert(df, f"{security}_quote.db", table, ("date",))


def upsert_ohlcv_pandas(
  table: str,
  security: Security,
  df: DataFrame[Quote],
) -> None:
  df.index = cast(pd.DatetimeIndex, df.index).strftime("%Y%m%d").astype(int)
  upsert_sqlite(df, f"{security}_quote.db", table)


async def load_ohlcv(
  table: str,
  security: Security,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: dt | Date | None = None,
  end_date: dt | Date | None = None,
  cols: list[QuoteColumn] | None = None,
) -> DataFrame[Quote]:
  cols_ = cols or ["open", "high", "low", "close", "volume"]
  col_text = "date, " + ", ".join(cols_)

  query = f"SELECT {col_text} FROM '{table}'"

  params = {}
  if start_date is not None:
    params["start_date"] = int(dt.strftime(start_date, "%Y%m%d"))
    query += " WHERE date >= :start_date"

  if end_date is not None:
    params["end_date"] = int(dt.strftime(end_date, "%Y%m%d"))
    query += f"{' AND' if 'WHERE' in query else ' WHERE'} date <= :end_date"

  date_cast = [
    pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date")
  ]
  ohlcv = polars_from_sqlite(
    f"{security}_quote.db", query, params=params, column_transform=date_cast
  )

  if ohlcv is None or ohlcv.is_empty():
    try:
      ohlcv = await ohlcv_fetcher()
    except Exception as e:
      raise Exception(f"Failed to fetch OHLCV: {e}")

    upsert_ohlcv(table, security, ohlcv.clone())
    if cols is not None:
      ohlcv = ohlcv.select(["date"] + cols)

    return cast(
      DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
    )

  last_date = ohlcv["date"].max()
  update = False
  if end_date is not None and last_date < end_date:
    query = f"""SELECT EXISTS (
      SELECT 1
      FROM '{table}'
      WHERE date > :last_date
    )"""
    update = fetch_sqlite(f"{security}_quote.db", query, {"last_date": last_date})[0][0]
    print(f"update: {update}")

  today = dt.now()
  if delta == 1 and today.weekday() >= 5:
    delta = 2

  if end_date is None and relativedelta(today, last_date).days <= delta:
    update = True

  if not update:
    return cast(
      DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
    )

  try:
    new_ohlcv = await ohlcv_fetcher(start_date=last_date)
  except Exception as e:
    raise Exception(f"Failed to fetch OHLCV: {e}")

  if new_ohlcv is None or new_ohlcv.is_empty():
    return slice_polars_by_date(ohlcv, "date", start_date, end_date)

  print(f"last date: {last_date}")
  print(f"end date: {end_date}")
  print(new_ohlcv.shape)
  upsert_ohlcv(table, security, new_ohlcv.clone())

  ohlcv = pl.concat([ohlcv, new_ohlcv]).unique().sort("date")

  if cols is not None:
    ohlcv = ohlcv[["date"] + cols]

  return cast(
    DataFrame[Quote], slice_polars_by_date(ohlcv, "date", start_date, end_date)
  )


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

    upsert_ohlcv_pandas(table, security, ohlcv.copy())
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

  upsert_ohlcv_pandas(table, security, new_ohlcv.copy())

  ohlcv = cast(
    DataFrame[Quote],
    pd.concat([ohlcv, new_ohlcv], axis=0).sort_index().drop_duplicates(),
  )
  if cols is not None:
    ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, cols])

  return cast(DataFrame[Quote], slice_pandas_by_date(ohlcv, start_date, end_date))


def load_ohlcv_batch(
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
