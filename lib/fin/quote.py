from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Any, Coroutine, Literal, Optional

from pandera.typing import DataFrame

from lib.db.lite import read_sqlite, upsert_sqlite
from lib.fin.models import Quote


async def get_ohlcv(
  _id: str,
  security: str,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  delta: int = 1,
  start_date: Optional[dt] = None,
  cols: Optional[set[Literal['open', 'high', 'low', 'close', 'volume']]] = None,
) -> DataFrame[Quote]:
  col_text = '*'
  if cols is not None:
    cols_ = list({'date'}.union(cols))
    col_text = ', '.join(cols_)

  query = f'SELECT {col_text} FROM "{_id}"'

  if start_date is not None:
    query += f' WHERE DATE(date) >= DATE({start_date.strftime("%Y-%m-%d")})'

  ohlcv = read_sqlite(
    f'{security}_quote.db',
    query,
    index_col='date',
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )

  if ohlcv is None:
    ohlcv = await ohlcv_fetcher()
    upsert_sqlite(ohlcv, f'{security}_quote.db', _id)
    if cols is not None:
      ohlcv = cast(DataFrame[Quote], ohlcv.loc[:, cols_])

    return cast(DataFrame[Quote], ohlcv)

  last_date: dt = ohlcv.index.max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return cast(DataFrame[Quote], ohlcv)

  new_ohlcv = await ohlcv_fetcher(last_date)

  if new_ohlcv is None:
    return ohlcv

  upsert_sqlite(ohlcv, f'{security}_quote.db', _id)
  ohlcv = read_sqlite(
    f'{security}_quote.db',
    query,
    index_col='date',
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )

  return cast(DataFrame[Quote], ohlcv)
