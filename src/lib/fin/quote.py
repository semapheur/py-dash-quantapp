from datetime import date as Date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Any, Coroutine, Literal, Optional

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
  df.index = df.index.dt.strftime("%Y%m%d").astype(int)
  upsert_sqlite(df, f"{security}_quote.db", table)


async def load_ohlcv(
  table: str,
  security: Literal["stock", "forex", "index"],
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: Optional[dt | Date] = None,
  end_date: Optional[dt | Date] = None,
  cols: Optional[list[Literal["open", "high", "low", "close", "volume"]]] = None,
) -> DataFrame[Quote]:
  col_text = "CAST(date AS TEXT), open, high, low, close, volume"
  if cols is not None:
    col_text = "CAST(date AS TEXT), " + ", ".join(cols)

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

  if ohlcv is None:
    ohlcv = await ohlcv_fetcher()
    ohlcv.index = ohlcv.index.dt.strftime("%Y%m%d").astype(int)
    upsert_ohlcv(table, security, ohlcv)
    if cols is not None:
      ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, list(cols)])

    return cast(DataFrame[Quote], slice_df_by_date(ohlcv, start_date, end_date))

  if (delta is None) or (end_date is not None):
    return cast(DataFrame[Quote], ohlcv)

  last_date: dt = ohlcv.index.max()
  if not isinstance(last_date, dt):
    print(f"Last date: {last_date}")

  if relativedelta(dt.now(), last_date).days <= delta:
    return cast(DataFrame[Quote], ohlcv)

  new_ohlcv = await ohlcv_fetcher(last_date)

  if new_ohlcv is None:
    return ohlcv

  upsert_ohlcv(table, security, new_ohlcv)
  ohlcv = cast(
    DataFrame[Quote], pd.concat([ohlcv, new_ohlcv], axis=0).drop_duplicates()
  )

  return cast(DataFrame[Quote], ohlcv)
