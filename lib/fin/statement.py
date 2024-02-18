from datetime import date as Date, timedelta
from enum import Enum
from functools import partial
import json
import sqlite3
from typing import cast, Optional

from asyncstdlib.functools import cache
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


@cache
async def fetch_exchange_rate(
  ticker: str, start_date: Date, end_date: Date, extract_date: Optional[Date] = None
) -> float:
  from numpy import nan

  exchange_fetcher = partial(Ticker(ticker + '=X').ohlcv)
  rate = await get_ohlcv(
    ticker, 'forex', exchange_fetcher, None, start_date, end_date, ['close']
  )
  if rate.empty:
    return nan

  if extract_date is None:
    return rate['close'].mean()

  return rate.resample('D').ffill().at[pd.to_datetime(extract_date), 'close']


async def statement_to_df(
  financials: FinStatement, currency: Optional[str] = None, multiple=False
) -> DataFrame:
  def parse_date(period: Instant | Interval) -> Date:
    if isinstance(period, Interval):
      return period.end_date

    return period.instant

  async def exchange_rate(
    currency: str, unit: str, period: Instant | Interval
  ) -> float:
    ticker = f'{unit}{currency}'.upper()

    extract_date: None | Date = None
    if isinstance(period, Instant):
      extract_date = period.instant
      start_date = extract_date - timedelta(days=7)
      end_date = extract_date + timedelta(days=7)
    elif isinstance(period, Interval):
      start_date = period.start_date
      end_date = period.end_date

    rate = await fetch_exchange_rate(ticker, start_date, end_date, extract_date)
    return rate

  fin_date = pd.to_datetime(financials.date)
  fin_scope = financials.scope
  fin_period = financials.period
  currencies = financials.currency.difference({currency})

  df_data: dict[tuple[Date, FiscalPeriod, int], dict[str, int | float]] = {}

  for item, entries in financials.data.items():
    for entry in entries:
      rate = 1.0
      date = pd.to_datetime(parse_date(entry['period']))

      if (date > fin_date) or ((not multiple) and date != fin_date):
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
          rate = await exchange_rate(currency, unit, entry['period'])

        df_data.setdefault((date, period, months), {})[item] = value * rate

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or entry.get('unit') == 'shares'
        ):
          df_data.setdefault((date, 'Q4', 3), {})[item] = value * rate

      if (members := entry.get('members')) is None:
        continue

      for member, m_entry in members.items():
        if (m_value := m_entry.get('value')) is None:
          continue

        if (currency is not None) and (unit := m_entry.get('unit', '')) in currencies:
          rate = await exchange_rate(currency, unit, entry['period'])

        dim = '.' + d if (d := m_entry.get('dim')) else ''
        key = f'{item}{dim}.{member}'
        df_data.setdefault((date, period, months), {})[key] = m_value * rate

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or m_entry.get('unit') == 'shares'
        ):
          df_data.setdefault((date, 'Q4', 3), {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period', 'months']
  return cast(DataFrame, df)


async def load_financials(id: str, currency: Optional[str] = None) -> DataFrame | None:
  df_statements = load_statements(id)
  if df_statements is None:
    return None

  df_statements.sort_values('date', inplace=True)
  statements = df_to_statements(df_statements)

  # for s in statements:
  #  df = cast(DataFrame, df.combine_first(await statement_to_df(s, currency)))

  dfs = [await statement_to_df(statements.pop(0), currency, True)] + [
    await statement_to_df(s, currency) for s in statements
  ]
  df = pd.concat(dfs, join='outer')

  df.sort_index(level=0, ascending=True, inplace=True)
  df = cast(
    DataFrame, df.loc[df.index.get_level_values('months').isin((12, 9, 6, 3)), :]
  )

  df = fix_financials(df)

  ratios = stock_splits(id)
  if ratios is not None:
    df = stock_split_adjust(df, ratios)

  return df


def fix_financials(df: DataFrame) -> DataFrame:
  def check_combos(ix: pd.MultiIndex, conditions: set[tuple[str, int]]) -> bool:
    return conditions.issubset(set(ix.droplevel('date')))

  query = """
    SELECT json_each.value AS gaap, item FROM items 
    JOIN json_each(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  """
  items = read_sqlite('taxonomy.db', query)
  if items is None:
    raise ValueError('Taxonomy could not be loaded!')

  df.to_csv('test_.csv')

  df = cast(
    DataFrame, df.loc[:, list(set(df.columns).intersection(set(items['gaap'])))]
  )

  rename = {k: v for k, v in zip(items['gaap'], items['item'])}
  df.rename(columns=rename, inplace=True)
  df = combine_duplicate_columns(df)

  query = 'SELECT item FROM items WHERE aggregate = "sum"'
  sum_items = read_sqlite('taxonomy.db', query)
  if sum_items is None:
    raise ValueError('Taxonomy could not be loaded!')

  diff_items = list(set(sum_items['item']).intersection(set(df.columns)))

  conditions = (('Q1', 3), ('Q2', 6), ('Q3', 9), ('FY', 12))
  mp = df.index.droplevel('date')
  for i in range(1, len(conditions)):
    mask = (mp == conditions[i - 1]) | (mp == conditions[i])
    # mask = (period == conditions[i - 1][0]) & (months == conditions[i - 1][1]) | (
    #  period == conditions[i][0]
    # ) & (months == conditions[i][1])
    df_ = df.loc[mask, diff_items].copy()
    if not check_combos(
      cast(pd.MultiIndex, df_.index), {conditions[i - 1], conditions[i]}
    ):
      continue

    df_.sort_index(level='date', inplace=True)

    df_['month_difference'] = df_time_difference(
      cast(pd.DatetimeIndex, df_.index.get_level_values('date')), 30, 'D'
    )

    df_.loc[:, diff_items] = df_[diff_items].diff()
    df_ = df_.loc[df_['month_difference'] == 3, diff_items]

    df_ = df_.loc[(slice(None), conditions[i][0], conditions[i][1]), :]
    df_.reset_index(level='months', inplace=True)
    df_.loc[:, 'months'] = 3
    df_.set_index('months', append=True, inplace=True)

    if conditions[i][0] == 'FY':
      df_.reset_index(level='period', inplace=True)
      df_.loc[:, 'period'] = 'Q4'
      df_.set_index('period', append=True, inplace=True)
      df_ = df_.reorder_levels(['date', 'period', 'months'])

    df = cast(DataFrame, df.combine_first(df_))

  period = df.index.get_level_values('period')
  months = df.index.get_level_values('months')

  mask = ((months == 3) & period.isin({'Q1', 'Q2', 'Q3', 'Q4'})) | (
    (months == 12) & (period == 'FY')
  )

  return cast(DataFrame, df.loc[mask, :].dropna(how='all'))


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


def stock_splits(id: str) -> Series[float]:
  where_text = ' AND '.join(
    [f'json_extract(data, "$.{item}") IS NOT NULL' for item in stock_split_items]
  )

  query = f'SELECT data FROM "{id}" WHERE {where_text}'
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
  id: str, date: Optional[Date] = None
) -> DataFrame[FinStatementFrame] | None:
  query = f'SELECT * FROM "{id}" ORDER BY date ASC'
  if date:
    query += f' WHERE DATE(date) >= DATE("{date:%Y-%m-%d}")'

  df = read_sqlite(
    'financials.db',
    query,
    date_parser={'date': {'format': '%Y-%m-%d'}},
  )
  return df


def load_statements_json(id: str, date: Optional[Date] = None):
  df = load_statements(id, date)
  if df is None:
    return None

  statements = df_to_statements(df)

  return [s.to_dict() for s in statements]


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
    date DATE,
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
