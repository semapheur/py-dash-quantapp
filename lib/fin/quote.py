from dateutil.relativedelta import relativedelta
from functools import partial
from typing import Optional

from pandera.typing import DataFrame


def get_ohlcv(
  _id: str,
  security: str,
  ohlcv_fetcher: partial[pd.DataFrame],
  delta: int = 1,
  cols: Optional[set[str]] = None,
) -> pd.DataFrame:
  col_text = '*'
  if cols is not None:
    col_text = ', '.join(cols.union({'date'}))

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
    ohlcv = ohlcv_fetcher()
    upsert_sqlite(ohlcv, f'{security}_quote.db', _id)

    return ohlcv[list(cols)]

  last_date: dt = ohlcv.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return ohlcv

  new_ohlcv = ohlcv_fetcher(last_date.strftime('%Y-%m-%d'))

  if new_ohlcv is None:
    return ohlcv

  upsert_sqlite(ohlcv, f'{security}_quote.db', _id)
  ohlcv = read_sqlite(
    f'{security}_quote.db',
    query,
    index_col='date',
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )

  return ohlcv
