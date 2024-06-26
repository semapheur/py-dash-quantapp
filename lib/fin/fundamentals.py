from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
from typing import cast, Any, Coroutine, Optional
from sqlalchemy.types import Date

from ordered_set import OrderedSet
from numpy import nan
import pandas as pd
from pandera.typing import DataFrame, Index, Series

from lib.db.lite import read_sqlite, upsert_sqlite
from lib.fin.calculation import calculate_items, trailing_twelve_months
from lib.fin.metrics import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.fin.models import Quote, CloseQuote
from lib.fin.statement import load_financials, stock_splits
from lib.fin.quote import get_ohlcv
from lib.fin.taxonomy import TaxonomyCalculation
from lib.yahoo.ticker import Ticker


def load_schema(query: Optional[str] = None) -> dict[str, TaxonomyCalculation]:
  if query is None:
    query = """
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    """

  df = read_sqlite('taxonomy.db', query)
  if df is None:
    raise ValueError('Could not load taxonomy!')

  df.loc[:, 'calculation'] = df['calculation'].apply(lambda x: json.loads(x))
  schema = {k: TaxonomyCalculation(**v) for k, v in zip(df['item'], df['calculation'])}
  return schema


def stock_split_adjust(financials: DataFrame, ratios: Series | None) -> DataFrame:
  price_cols = {
    'share_price_close',
    'share_price_open',
    'share_price_high',
    'share_price_low',
    'share_price_average',
  }
  price_cols_ = list(price_cols.intersection(financials.columns))

  for i, col in enumerate(price_cols_):
    financials[f'{col}_adjusted'] = financials[col]
    price_cols_[i] = f'{col}_adjusted'

  if ratios is None:
    return financials

  for date, ratio in ratios.items():
    mask = financials.index.get_level_values('date') < date
    financials.loc[mask, price_cols_] /= ratio

  return financials


def merge_share_price(financials: DataFrame, price: DataFrame[CloseQuote]) -> DataFrame:
  price_records = [
    {
      'share_price_close': nan,
      'share_price_open': nan,
      'share_price_high': nan,
      'share_price_low': nan,
      'share_price_average': nan,
    }
    for _ in range(len(financials))
  ]

  for i, ix in enumerate(cast(pd.MultiIndex, financials.index)):
    end_date = cast(dt, ix[0])
    start_date = end_date - relativedelta(months=cast(int, ix[2]))

    price_ = (
      cast(Series[float], price.loc[start_date:end_date, 'close']).copy().sort_index()
    )
    if price_.empty:
      continue

    price_records[i]['share_price_close'] = price_.iloc[-1]
    price_records[i]['share_price_open'] = price_.iloc[0]
    price_records[i]['share_price_high'] = price_.max()
    price_records[i]['share_price_low'] = price_.min()
    price_records[i]['share_price_average'] = price_.mean()

  price_data = pd.DataFrame.from_records(price_records, index=financials.index)

  return cast(DataFrame, pd.concat((financials, price_data), axis=1))


async def calculate_fundamentals(
  id: str,
  currency: str,
  financials: DataFrame,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  beta_years: Optional[int] = None,
  update: bool = False,
) -> DataFrame:
  start_date: dt = cast(pd.MultiIndex, financials.index).levels[
    0
  ].min() - relativedelta(years=beta_years or 1)
  price = cast(
    DataFrame[CloseQuote],
    await get_ohlcv(
      f'{id}_{currency}', 'stock', ohlcv_fetcher, start_date=start_date, cols=['close']
    ),
  )

  financials = trailing_twelve_months(financials)
  if update and (3 in cast(pd.MultiIndex, financials.index).levels[2]):
    financials = cast(
      DataFrame,
      pd.concat(
        (
          financials.loc[(slice(None), slice(None), 3), :].tail(2),
          financials.loc[(slice(None), slice(None), 12), :],
        ),
        axis=0,
      ),
    )

  financials = merge_share_price(financials, price)
  split_ratios = stock_splits(id)
  financials = stock_split_adjust(financials, split_ratios)
  schema = load_schema()
  financials = calculate_items(financials, schema)
  financials.to_csv('test.csv')

  market_fetcher = partial(Ticker('^GSPC').ohlcv)
  market_close = cast(
    DataFrame[CloseQuote],
    await get_ohlcv(
      'GSPC', 'index', market_fetcher, start_date=start_date, cols=['close']
    ),
  )
  riskfree_fetcher = partial(Ticker('^TNX').ohlcv)
  riskfree_rate = cast(
    DataFrame[CloseQuote],
    await get_ohlcv(
      'TNX', 'index', riskfree_fetcher, start_date=start_date, cols=['close']
    ),
  )

  financials = beta(
    financials,
    price['close'].rename('equity_return').pct_change(),
    market_close['close'].rename('market_return').pct_change(),
    riskfree_rate['close'].rename('riskfree_rate') / 100,
  )

  financials = weighted_average_cost_of_capital(financials)
  financials = f_score(financials)
  financials = m_score(financials)

  if 'altman_z_score' not in set(financials.columns):
    financials = z_score(financials)

  return financials


def handle_ttm(df: DataFrame) -> DataFrame:
  if 'TTM' not in cast(pd.MultiIndex, df.index).levels[1].unique():
    return df

  df.sort_index(level='date', inplace=True)
  mask = (slice(None), slice('TTM'), 12)
  drop_index = df.loc[mask, :].tail(-2).index[0]
  ttm_index = df.loc[mask, :].tail(1).index[0]

  df.drop(drop_index, inplace=True)
  df.rename(index={ttm_index: (dt(1900, 1, 1))}, inplace=True)

  return df


def load_ttm(df: pd.DataFrame) -> pd.DataFrame:
  ttm_date = cast(pd.MultiIndex, df.index).levels[0].max()

  renamer = {(dt(1900, 1, 1), 'TTM', 12): (ttm_date, 'TTM', 12)}
  df.rename(index=renamer, inplace=True)
  return df


def load_fundamentals(
  id: str, currency: str, cols: Optional[set[str]] = None
) -> DataFrame | None:
  col_text = '*'
  index_col = OrderedSet(('date', 'period', 'months'))
  if cols is not None:
    col_text = ', '.join(cols.union(index_col))

  table = f'{id}_{currency}'
  query = f'SELECT {col_text} FROM "{table}"'
  df = read_sqlite(
    'fundamentals.db',
    query,
    index_col=list(index_col),
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )
  return df


async def update_fundamentals(
  id: str,
  currency: str,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  beta_years: Optional[None] = None,
  cols: Optional[set[str]] = None,
) -> pd.DataFrame:
  table = f'{id}_{currency}'

  financials = await load_financials(id, currency)
  if financials is None:
    raise ValueError(f'Statements have not been seeded for {id}')

  fundamentals = load_fundamentals(id, currency, cols)

  if fundamentals is None:
    fundamentals = await calculate_fundamentals(
      id, currency, financials, ohlcv_fetcher, beta_years
    )

    upsert_sqlite(fundamentals, 'fundamentals.db', table, {'date': Date})
    return fundamentals

  last_financials: dt = cast(pd.MultiIndex, fundamentals.index).levels[0].max()
  last_fundamentals: dt = cast(pd.MultiIndex, fundamentals.index).levels[0].max()
  if last_fundamentals >= last_financials:
    return fundamentals

  financials = cast(
    DataFrame,
    financials.loc[financials.index.get_level_values('date') > last_fundamentals],
  )

  fundamentals.sort_index(level='date', inplace=True)
  props = {3: (None, 8), 12: ('FY', 1)}
  months = cast(Index[int], cast(pd.MultiIndex, financials.index).levels[2].unique())
  for m in months:
    mask = (slice(None), slice(props[m][0]), m)
    fundamentals_ = cast(
      DataFrame,
      pd.concat((fundamentals.loc[mask, :].tail(props[m][1]), financials), axis=0),
    )

  fundamentals_ = await calculate_fundamentals(
    id, currency, fundamentals_, ohlcv_fetcher, beta_years=beta_years, update=True
  )
  fundamentals_ = cast(
    DataFrame, fundamentals_.loc[fundamentals_.index.difference(fundamentals.index), :]
  )
  upsert_sqlite(fundamentals_, 'fundamentals.db', table)
  return cast(DataFrame, pd.concat((fundamentals, fundamentals_), axis=0))
