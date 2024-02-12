from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
from typing import cast, Any, Coroutine, Optional

from ordered_set import OrderedSet
import pandas as pd
from pandera.typing import DataFrame

from lib.db.lite import read_sqlite, upsert_sqlite
from lib.fin.calculation import calculate_items, trailing_twelve_months
from lib.fin.metrics import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.fin.models import Quote
from lib.fin.statement import load_financials
from lib.fin.quote import get_ohlcv
from lib.yahoo.ticker import Ticker


def load_schema(query: Optional[str] = None) -> dict[str, dict]:
  if query is None:
    query = """
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    """

  df = read_sqlite('taxonomy.db', query)
  if df is None:
    raise ValueError('Could not load taxonomy!')

  df.loc[:, 'calculation'] = df['calculation'].apply(lambda x: json.loads(x))
  schema = {k: v for k, v in zip(df['item'], df['calculation'])}
  return schema


async def calculate_fundamentals(
  id: str,
  fin_table: DataFrame,
  ohlcv_fetcher: partial[Coroutine[Any, Any, DataFrame[Quote]]],
  beta_period: int = 5,
  update: bool = False,
) -> DataFrame:
  price = await get_ohlcv(id, 'stock', ohlcv_fetcher, cols={'close'})
  price = cast(DataFrame[Quote], price.resample('D').ffill())

  fin_table = trailing_twelve_months(fin_table)
  if update and (3 in cast(pd.MultiIndex, fin_table.index).levels[2]):
    fin_table = cast(
      DataFrame,
      pd.concat(
        (
          fin_table.loc[(slice(None), slice(None), 3), :].tail(2),
          fin_table.loc[(slice(None), slice(None), 12), :],
        ),
        axis=0,
      ),
    )

  fin_table.reset_index(inplace=True)
  fin_table = cast(
    DataFrame,
    fin_table.reset_index()
    .merge(price.rename(columns={'close': 'share_price'}), how='left', on='date')
    .set_index(['date', 'period', 'months'])
    .drop('index', axis=1),
  )
  schema = load_schema()
  fin_table = calculate_items(fin_table, schema)

  start_date: dt = fin_table.index.get_level_values('date').min() - relativedelta(
    years=beta_period
  )
  market_close = await Ticker('^GSPC').ohlcv(start_date)
  riskfree_rate = await Ticker('^TNX').ohlcv(start_date)

  fin_table = beta(
    fin_table,
    price['close'].rename('equity_return').pct_change(),
    market_close['close'].rename('market_return').pct_change(),
    riskfree_rate['close'].rename('riskfree_rate') / 100,
  )

  fin_table = weighted_average_cost_of_capital(fin_table)
  fin_table = f_score(fin_table)
  fin_table = m_score(fin_table)

  if 'altman_z_score' not in set(fin_table.columns):
    fin_table = z_score(fin_table)

  return fin_table


def handle_ttm(df: DataFrame) -> DataFrame:
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
  cols: Optional[set[str]] = None,
) -> pd.DataFrame:
  table = f'{id}_{currency}'

  financials = await load_financials(id, currency)
  if financials is None:
    raise ValueError(f'Statements have not been seeded for {id}')

  fundamentals = load_fundamentals(id, currency, cols)

  if fundamentals is None:
    fundamentals = await calculate_fundamentals(id, financials, ohlcv_fetcher)
    upsert_sqlite(handle_ttm(fundamentals), 'fundamentals.db', table)
    return fundamentals

  last_financials: dt = fundamentals.index.get_level_values('date').max()
  last_fundamentals: dt = fundamentals.index.get_level_values('date').max()
  if last_fundamentals >= last_financials:
    return fundamentals

  financials = cast(
    DataFrame,
    financials.loc[financials.index.get_level_values('date') > last_fundamentals],
  )

  fundamentals.sort_index(level='date', inplace=True)
  props = {3: (None, 8), 12: ('FY', 1)}
  for m in financials.index.get_level_values('months').unique():
    mask = (slice(None), slice(props[m][0]), m)
    fundamentals_ = cast(
      DataFrame,
      pd.concat((fundamentals.loc[mask, :].tail(props[m][1]), financials), axis=0),
    )

  fundamentals_ = await calculate_fundamentals(id, fundamentals_, ohlcv_fetcher)
  fundamentals_ = cast(
    DataFrame, fundamentals_.loc[fundamentals_.index.difference(fundamentals.index), :]
  )
  upsert_sqlite(handle_ttm(fundamentals_), 'fundamentals.db', table)
  return cast(DataFrame, pd.concat((fundamentals, fundamentals_), axis=0))
