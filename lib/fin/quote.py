from datetime import date as Date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Any, Coroutine, Literal, Optional

import pandas as pd
from pandera.typing import DataFrame
from sqlalchemy.types import Date as SQLDate

from lib.db.lite import read_sqlite, upsert_sqlite
from lib.fin.models import Quote
from lib.utils import slice_df_by_date


async def load_ohlcv(
  table: str,
  security: Literal["stock", "forex", "index"],
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: Optional[dt | Date] = None,
  end_date: Optional[dt | Date] = None,
  cols: Optional[list[Literal["open", "high", "low", "close", "volume"]]] = None,
) -> DataFrame[Quote]:
  col_text = "*"
  if cols is not None:
    col_text = "date, " + ", ".join(cols)

  query = f"SELECT {col_text} FROM '{table}'"

  if start_date is not None:
    query += f" WHERE DATE(date) >= DATE('{start_date:%Y-%m-%d}')"

  if end_date is not None:
    query += f"{' AND' if 'WHERE' in query else ' WHERE'} DATE(date) <= DATE('{end_date:%Y-%m-%d}')"

  ohlcv = read_sqlite(
    f"{security}_quote.db",
    query,
    index_col="date",
    date_parser={"date": {"format": "%Y-%m-%d"}},
  )

  if ohlcv is None:
    ohlcv = await ohlcv_fetcher()
    upsert_sqlite(ohlcv, f"{security}_quote.db", table, {"date": SQLDate})
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

  upsert_sqlite(new_ohlcv, f"{security}_quote.db", table, {"date": SQLDate})
  ohlcv = cast(
    DataFrame[Quote], pd.concat([ohlcv, new_ohlcv], axis=0).drop_duplicates()
  )

  # ohlcv = read_sqlite(
  #  f"{security}_quote.db",
  #  query,
  #  index_col="date",
  #  date_parser={"date": {"format": "%Y-%m-%d"}},
  # )

  return cast(DataFrame[Quote], ohlcv)
