from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
from typing import cast, Awaitable, Optional

from ordered_set import OrderedSet
import pandas as pd

from lib.db.lite import read_sqlite
from lib.fin.calculation import calculate_items, trailing_twelve_months
from lib.fin.metrics import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.yahoo.ticker import Ticker
from lib.ticker.fetch import get_ohlcv


def load_schema(query: Optional[str] = None) -> dict[str, dict]:
  if query is None:
    query = """
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    """

  df = read_sqlite('taxonomy.db', query)
  df.loc[:, 'calculation'] = df['calculation'].apply(lambda x: json.loads(x))
  schema = {k: v for k, v in zip(df['item'], df['calculation'])}
  return schema


async def calculate_fundamentals(
  _id: str,
  fin_table: pd.DataFrame,
  ohlcv_fetcher: partial[Awaitable[pd.DataFrame]],
  beta_period: int = 5,
  update: bool = False,
) -> pd.DataFrame:
  price = get_ohlcv(_id, 'stock', ohlcv_fetcher, cols={'date', 'close'})
  price.rename(columns={'close': 'share_price'}, inplace=True)
  price = price.resample('D').ffill()

  fin_table = trailing_twelve_months(fin_table)
  if update and (3 in cast(pd.MultiIndex, fin_table.index).levels[2]):
    fin_table = pd.concat(
      (
        fin_table.loc[(slice(None), slice(None), 3), :].tail(2),
        fin_table.loc[(slice(None), slice(None), 12), :],
      ),
      axis=0,
    )

  fin_table.reset_index(inplace=True)
  fin_table = (
    fin_table.reset_index()
    .merge(price, how='left', on='date')
    .set_index(['date', 'period', 'months'])
    .drop('index', axis=1)
  )
  schema = load_schema()
  fin_table = calculate_items(fin_table, schema)

  start_date: dt = fin_table.index.get_level_values('date').min() - relativedelta(
    years=beta_period
  )

  market_close = await Ticker('^GSPC').ohlcv(start_date)

  partial(Ticker('^GSPC').ohlcv)
  riskfree_fetcher = partial(Ticker('^TNX').ohlcv)
  price.rename(columns={'share_price': 'equity_return'}, inplace=True)
  fin_table = beta(_id, fin_table, price, market_fetcher, riskfree_fetcher)

  fin_table = weighted_average_cost_of_capital(fin_table)
  fin_table = f_score(fin_table)
  fin_table = m_score(fin_table)

  if 'altman_z_score' not in set(fin_table.columns):
    fin_table = z_score(fin_table)

  return fin_table


def handle_ttm(df: pd.DataFrame) -> pd.DataFrame:
  if 'TTM' not in df.index.get_level_values('period').unique():
    return df

  df.sort_index(level='date', inplace=True)
  mask = (slice(None), slice('TTM'), 12)
  drop_index = df.loc[mask, :].tail(-2).index[0]
  ttm_index = df.loc[mask, :].tail(1).index[0]

  df.drop(drop_index, inplace=True)
  df.rename(index={ttm_index: (dt(1900, 1, 1))}, inplace=True)

  return df


def load_ttm(df: pd.DataFrame) -> pd.DataFrame:
  ttm_date = df.index.get_level_values('date').max()

  renamer = {(dt(1900, 1, 1), 'TTM', 12): (ttm_date, 'TTM', 12)}
  df.rename(index=renamer, inplace=True)
  return df


async def get_fundamentals(
  _id: str,
  financials_fetcher: partial[Awaitable[pd.DataFrame]],
  ohlcv_fetcher: partial[pd.DataFrame],
  cols: Optional[set[str]] = None,
  delta: int = 120,
) -> pd.DataFrame:
  col_text = '*'
  index_col = OrderedSet(('date', 'period', 'months'))
  if cols is not None:
    col_text = ', '.join(cols.union(index_col))

  query = f'SELECT {col_text} FROM "{_id}"'
  df = read_sqlite(
    'fundamentals.db', query, index_col=list(index_col), parse_dates=True
  )

  if df is None:
    df = await financials_fetcher()
    df = calculate_fundamentals(_id, df, ohlcv_fetcher)
    upsert_sqlite(handle_ttm(df), 'fundamentals.db', _id)

  last_date: dt = df.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return df

  _df = await financials_fetcher(last_date.strftime('%Y-%m-%d'))
  if _df is None:
    return df

  df.sort_index(level='date', inplace=True)
  props = {3: (None, 8), 12: ('FY', 1)}
  for m in _df.index.get_level_values('months').unique():
    mask = (slice(None), slice(props[m][0]), m)
    _df = pd.concat((df.loc[mask, :].tail(props[m][1]), _df), axis=0)

  _df = calculate_fundamentals(_df, ohlcv_fetcher)
  _df: pd.DataFrame = _df.loc[_df.index.difference(df.index), :]
  upsert_sqlite(handle_ttm(_df), 'fundamentals.db', _id)
  df = pd.concat((df, _df), axis=0)

  # df = read_sqlite('fundamentals.db', query,
  #  index_col=index_col,
  #  parse_dates=True
  # )
  return df
