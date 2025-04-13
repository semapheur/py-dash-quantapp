from datetime import date as Date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Any, Coroutine, Literal

import pandas as pd
from pandera.typing import DataFrame

from lib.db.lite import read_sqlite, upsert_sqlite
from lib.fin.models import Quote
from lib.utils.dataframe import slice_df_by_date


def upsert_ohlcv(
  table: str,
  security: Literal["stock", "forex", "index"],
  df: DataFrame[Quote],
) -> None:
  df.index = cast(pd.DatetimeIndex, df.index).strftime("%Y%m%d").astype(int)
  upsert_sqlite(df, f"{security}_quote.db", table)


async def load_ohlcv(
  table: str,
  security: Literal["stock", "forex", "index"],
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: dt | Date | None = None,
  end_date: dt | Date | None = None,
  cols: list[Literal["open", "high", "low", "close", "volume"]] | None = None,
) -> DataFrame[Quote] | None:
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
      print(f"Failed to fetch OHLCV: {e}")
      return None

    upsert_ohlcv(table, security, ohlcv.copy())
    if cols is not None:
      ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, cols])

    return cast(DataFrame[Quote], slice_df_by_date(ohlcv, start_date, end_date))

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
    return cast(DataFrame[Quote], slice_df_by_date(ohlcv, start_date, end_date))

  try:
    new_ohlcv = await ohlcv_fetcher(last_date)
  except Exception as e:
    print(f"Failed to fetch OHLCV: {e}")
    return slice_df_by_date(ohlcv, start_date, end_date)

  if new_ohlcv is None or new_ohlcv.empty:
    return slice_df_by_date(ohlcv, start_date, end_date)

  upsert_ohlcv(table, security, new_ohlcv.copy())
  if cols is not None:
    new_ohlcv = cast(DataFrame[Quote], new_ohlcv.loc[:, cols])

  ohlcv = cast(
    DataFrame[Quote],
    pd.concat([ohlcv, new_ohlcv], axis=0).sort_index().drop_duplicates(),
  )

  return cast(DataFrame[Quote], slice_df_by_date(ohlcv, start_date, end_date))
