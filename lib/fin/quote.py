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
  cols: Optional[
    set[Literal['open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
  ] = None,
) -> DataFrame[Quote]:
  col_text = '*'
  if cols is not None:
    col_text = ', '.join({'date'}.union(cols))

  query = f'SELECT {col_text} FROM "{_id}"'

  if date := ohlcv_fetcher.args:
    query += f' WHERE DATE(date) >= DATE({date[0]})'

  ohlcv = read_sqlite(
    f'{security}_quote.db',
    query,
    index_col='date',
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )

  if ohlcv is None:
    ohlcv = await ohlcv_fetcher()
    upsert_sqlite(ohlcv, f'{security}_quote.db', _id)

    return ohlcv[list(cols)]

  last_date: dt = ohlcv.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return cast(DataFrame[Quote], ohlcv)

  new_ohlcv = await ohlcv_fetcher(last_date.strftime('%Y-%m-%d'))

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