from datetime import datetime as dt, date as Date, timedelta
from enum import Enum
from functools import partial
import json
import sqlite3
from typing import cast, Optional

import pandas as pd
from pandera.typing import DataFrame, Series

from lib.db.lite import read_sqlite, sqlite_path
from lib.fin.calculation import stock_split_adjust
from lib.fin.models import (
  FinStatement,
  FinStatementFrame,
  Instant,
  Interval,
  StockSplit,
  FiscalPeriod,
  FinData,
)
from lib.fin.quote import get_ohlcv
from lib.utils import combine_duplicate_columns, df_time_difference
from lib.yahoo.ticker import Ticker


class ScopeEnum(Enum):
  quarterly = 3
  annual = 12


stock_split_items = {
  'StockSplitRatio',
  'StockholdersEquityNoteStockSplitConversionRatio',
  'StockholdersEquityNoteStockSplitConversionRatio1',
  'ShareholdersEquityNoteStockSplitConverstaionRatioAuthorizedShares',
}


def df_to_statements(df: DataFrame[FinStatementFrame]) -> list[FinStatement]:
  return [
    FinStatement(
      url=row['url'],
      scope=row['scope'],
      date=row['date'],
      period=row['period'],
      fiscal_end=row['fiscal_end'],
      currency=row['currency'],
      data=row['data'],
    )
    for row in df.to_dict('records')
  ]


async def statement_to_df(
  financials: FinStatement, currency: Optional[str] = None
) -> DataFrame:
  def parse_date(period: Instant | Interval) -> Date:
    if isinstance(period, Interval):
      return period.end_date

    return period.instant

  async def get_exchange_rate(
    currency: str, unit: str, period: Instant | Interval
  ) -> float:
    ticker = f'{unit}{currency}'

    if isinstance(period, Instant):
      start_date = period.instant
      end_date = start_date + timedelta(days=2)
    elif isinstance(period, Interval):
      start_date = period.start_date
      end_date = period.end_date

    exchange_fetcher = partial(Ticker(ticker + '=X').ohlcv)
    rate = await get_ohlcv(
      ticker, 'exchange', exchange_fetcher, None, start_date, end_date, {'close'}
    )
    return rate['close'].mean()

  fin_date = financials.date
  fin_scope = financials.scope
  fin_period = financials.period
  currencies = financials.currency

  df_data: dict[tuple[Date, FiscalPeriod, int], dict[str, int | float]] = {}

  rate = 1.0

  for item, entries in financials.data.items():
    for entry in entries:
      date = parse_date(entry['period'])
      if date != fin_date:
        continue

      if isinstance(entry['period'], Interval):
        months = entry['period'].months
      else:
        months = ScopeEnum[fin_scope].value

      period = fin_period
      if fin_period == 'FY' and months < 12:
        period = 'Q4'

      if value := entry.get('value'):
        if (currency is not None) and (unit := entry.get('unit', '')) in currencies:
          rate = await get_exchange_rate(currency, unit, entry['period'])

        df_data.setdefault((fin_date, period, months), {})[item] = value * rate

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[item] = value

      if (members := entry.get('members')) is None:
        continue

      for member, m_entry in members.items():
        if (m_value := m_entry.get('value')) is None:
          continue

        if (currency is not None) and (unit := m_entry.get('unit', '')) in currencies:
          rate = await get_exchange_rate(currency, unit, entry['period'])

        dim = '.' + d if (d := m_entry.get('dim')) else ''
        key = f'{item}{dim}.{member}'
        df_data.setdefault((fin_date, period, months), {})[key] = m_value * rate

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or m_entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period', 'months']
  return cast(DataFrame, df)


async def load_financials(id_: str, currency: Optional[str] = None) -> DataFrame | None:
  df_scrap = load_statements(id_)
  if df_scrap is None:
    return None

  financials = df_to_statements(df_scrap)

  dfs: list[DataFrame] = []

  for f in financials:
    dfs.append(await statement_to_df(f, currency))

  df = pd.concat(dfs, join='outer')
  df.sort_index(level=0, ascending=True, inplace=True)
  df = df.loc[df.index.get_level_values('months').isin((12, 9, 6, 3)), :]

  if {'Q1', 'Q2', 'Q3', 'Q4'}.issubset(cast(pd.MultiIndex, df.index).levels[1]):
    df = fix_financials(df)

  ratios = stock_splits(id_)
  if ratios is not None:
    df = stock_split_adjust(df, ratios)

  return cast(DataFrame, df)


def fix_financials(df: pd.DataFrame) -> pd.DataFrame:
  query = """
    SELECT json_each.value AS gaap, item FROM items 
    JOIN json_each(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  """
  items = read_sqlite('taxonomy.db', query)
  if items is None:
    raise ValueError('Taxonomy could not be loaded!')

  df = df[list(set(df.columns).intersection(set(items['gaap'])))]

  rename = {k: v for k, v in zip(items['gaap'], items['item'])}
  df.rename(columns=rename, inplace=True)
  df = combine_duplicate_columns(df)

  query = 'SELECT item FROM items WHERE aggregate = "sum"'
  sum_items = read_sqlite('taxonomy.db', query)
  if sum_items is None:
    raise ValueError('Taxonomy could not be loaded!')

  diff_items = list(set(sum_items['item']).intersection(set(df.columns)))

  conditions = (('Q1', 3), ('Q2', 6), ('Q3', 9), ('FY', 12))

  period = df.index.get_level_values('period')
  months = df.index.get_level_values('months')
  for i in range(1, len(conditions)):
    mask = (period == conditions[i - 1][0]) & (months == conditions[i - 1][1]) | (
      period == conditions[i][0]
    ) & (months == conditions[i][1])
    df_ = df.loc[mask, diff_items]
    df_.sort_index(level='date', inplace=True)

    df_.loc[:, 'month_diff'] = df_time_difference(
      cast(pd.DatetimeIndex, df_.index.get_level_values('date')), 30, 'D'
    )

    df_.loc[:, diff_items] = df_[diff_items].diff()
    df_ = df_.loc[df_['month_diff'] == 3, diff_items]

    df_ = df_.loc[(slice(None), conditions[i][0], conditions[i][1]), :]
    df_.reset_index(level='months', inplace=True)
    df_.loc[:, 'months'] = 3
    df_.set_index('months', append=True, inplace=True)

    if conditions[i][0] == 'FY':
      df_.reset_index(level='period', inplace=True)
      df_.loc[:, 'period'] = 'Q4'
      df_.set_index('period', append=True, inplace=True)
      df_ = df_.reorder_levels(['date', 'period', 'months'])

    df = df.combine_first(df_)

  mask = (months == 3) & (period.isin({'Q1', 'Q2', 'Q3', 'Q4'})) | (months == 12) & (
    period == 'FY'
  )

  # df.reset_index(level='months', inplace=True)
  # df.loc[df['months'] == 12,'months'] = 'a'
  # df.loc[df['months'] == 3,'months'] = 'q'
  # df.rename(columns={'months': 'scope'}, inplace=True)
  # df.set_index('scope', append=True, inplace=True)

  return df.loc[mask, :]


def get_stock_splits(fin_data: FinData) -> list[StockSplit]:
  data: list[StockSplit] = []

  split_item = stock_split_items.intersection(fin_data.keys())

  if not split_item:
    return data

  splits = fin_data[split_item.pop()]

  for entry in splits:
    value = cast(float, entry.get('value'))

    data.append(
      StockSplit(
        date=cast(Interval, entry['period']).start_date,
        stock_split_ratio=value,
      )
    )
  return data


def stock_splits(id_: str) -> Series[float]:
  where_text = ' AND '.join(
    [f'json_extract(data, "$.{item}") IS NOT NULL' for item in stock_split_items]
  )

  query = f'SELECT data FROM "{id_}" WHERE {where_text}'
  df_parse = cast(
    DataFrame[str], read_sqlite('financials.db', query, dtype={'data': str})
  )
  if df_parse is None:
    return None

  fin_data = cast(list[FinData], df_parse['data'].apply(json.loads).to_list())

  df_data: list[StockSplit] = []
  for data in fin_data:
    df_data.extend(get_stock_splits(data))

  df = pd.DataFrame(df_data)
  df.drop_duplicates(inplace=True)
  df.set_index('date', inplace=True)

  return cast(Series[float], df['stock_split_ratio'])


def load_statements(
  id_: str, date: Optional[str] = None
) -> DataFrame[FinStatementFrame] | None:
  query = f'SELECT * FROM "{id_}" ORDER BY date ASC'
  if date:
    query += f' WHERE date >= {dt.strptime(date, "%Y-%m-%d")}'

  df = read_sqlite(
    'financials.db',
    query,
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )
  return df


def upsert_statements(
  db_name: str,
  table: str,
  statements: list[FinStatement],
):
  db_path = sqlite_path(db_name)

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  cur.execute(
    f"""CREATE TABLE IF NOT EXISTS "{table}"(
    url TEXT PRIMARY KEY,
    scope TEXT,
    date TEXT,
    period TEXT,
    fiscal_end TEXT,
    currency TEXT,
    data TEXT
  )"""
  )

  query = f"""INSERT INTO 
    "{table}" VALUES (:url, :scope, :date, :period, :fiscal_end, :currency, :data)
    ON CONFLICT (url) DO UPDATE SET  
      data=json_patch(data, excluded.data),
      currency=(
        SELECT json_group_array(value)
        FROM (
          SELECT json_each.value
          FROM json_each(currency)
          WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded.currency))
        )
      )
  """
  cur.executemany(query, [s.model_dump() for s in statements])

  con.commit()
  con.close()


def select_statements(db_name: str, table: str) -> list[FinStatement]:
  db_path = sqlite_path(db_name)

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()
    cur.row_factory = lambda _, row: FinStatement(**row)

    financials: list[FinStatement] = cur.execute(f'SELECT * FROM "{table}"').fetchall()

  return financials
