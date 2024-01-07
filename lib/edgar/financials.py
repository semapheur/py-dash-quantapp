from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import sqlite3
from typing import cast, Optional

import pandas as pd
from pandera.typing import DataFrame, Series
from lib.db.lite import read_sqlite, sqlite_name
from lib.const import DB_DIR

from lib.fin.calculation import stock_split_adjust
from lib.edgar.company import Company
from lib.edgar.models import (
  Interval,
  RawFinancials,
  RawFinancialsFrame,
  StockSplit,
  FinData,
)
from lib.edgar.parse import parse_statements, statement_to_df
from lib.utils import combine_duplicate_columns, df_time_difference


async def scrap_financials(cik: int, id_: str):
  company = Company(cik)
  filings = await company.xbrl_urls()

  financials = await parse_statements(filings.tolist())
  upsert_financials('financials_scrap.db', id_, financials)


async def load_financials(
  cik: int, id_: str, delta=120, date: Optional[str] = None
) -> list[RawFinancials]:
  def df_to_financials(df: DataFrame[RawFinancialsFrame]) -> list[RawFinancials]:
    return [
      RawFinancials(
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

  company = Company(cik)

  # Load financials
  query = f'SELECT * FROM "{id_}" ORDER BY date ASC'
  if date:
    query += f' WHERE date >= {dt.strptime(date, "%Y-%m-%d")}'

  df = cast(
    DataFrame[RawFinancialsFrame],
    read_sqlite(
      'financials_scrap.db', query, date_parser={'date': {'format': '%Y-%m-%d'}}
    ),
  )

  if not df.empty:
    last_date = df['date'].max()

    if relativedelta(dt.now(), last_date).days < delta:
      return df_to_financials(df)

    new_filings = await company.xbrl_urls(last_date)

    if not new_filings:
      return df_to_financials(df)

    old_filings = set(df['id'])
    filings_diff = set(new_filings.index).difference(old_filings)

    if not filings_diff:
      return df_to_financials(df)
  else:
    new_filings = await company.xbrl_urls()

  new_fin = await parse_statements(new_filings.tolist())
  if new_fin:
    upsert_financials('financials_scrap.db', id_, new_fin)

  return [*new_fin, *df_to_financials(df)]


async def financials_table(cik: int, id_: str) -> pd.DataFrame:
  financials = await load_financials(cik, id_)

  dfs = []

  for f in financials:
    dfs.append(statement_to_df(f))

  df = pd.concat(dfs, join='outer')
  df.sort_index(level=0, ascending=True, inplace=True)
  df = fix_financials_table(df)

  ratios = stock_splits(id_)
  if ratios is not None:
    df = stock_split_adjust(df, ratios)

  return df


def get_stock_splits(fin_data: FinData) -> list[StockSplit]:
  data: list[StockSplit] = []

  name = 'StockholdersEquityNoteStockSplitConversionRatio1'
  splits = fin_data.get(name)
  if splits is None:
    return data

  for entry in splits:
    value = cast(float, entry.get('value'))

    data.append(
      StockSplit(
        date=cast(Interval, entry['period']).start_date,
        stock_split_ratio=value,
      )
    )
  return data


def fix_financials_table(df: pd.DataFrame) -> pd.DataFrame:
  query = """
    SELECT json_each.value AS gaap, item FROM items 
    JOIN JSON_EACH(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  """
  items = read_sqlite('taxonomy.db', query)

  df = df[list(set(df.columns).intersection(set(items['gaap'])))]

  rename = {k: v for k, v in zip(items['gaap'], items['item'])}
  df.rename(columns=rename, inplace=True)
  df = combine_duplicate_columns(df)

  query = 'SELECT item FROM items WHERE period = "duration"'
  items = read_sqlite('taxonomy.db', query)
  diff_items = list(set(items['item']).intersection(set(df.columns)))

  conditions = (('Q1', 3), ('Q2', 6), ('Q3', 9), ('FY', 12))

  period = df.index.get_level_values('period')
  months = df.index.get_level_values('months')
  for i in range(1, len(conditions)):
    mask = (period == conditions[i - 1][0]) & (months == conditions[i - 1][1]) | (
      period == conditions[i][0]
    ) & (months == conditions[i][1])
    _df = df.loc[mask, diff_items]
    _df.sort_index(level='date', inplace=True)

    _df.loc[:, 'month_diff'] = df_time_difference(
      cast(pd.DatetimeIndex, _df.index.get_level_values('date')), 30, 'D'
    ).array

    _df.loc[:, diff_items] = _df[diff_items].diff()
    _df = _df.loc[_df['month_diff'] == 3, diff_items]

    _df = _df.loc[(slice(None), conditions[i][0], conditions[i][1]), :]
    _df.reset_index(level='months', inplace=True)
    _df.loc[:, 'months'] = 3
    _df.set_index('months', append=True, inplace=True)

    if conditions[i][0] == 'FY':
      _df.reset_index(level='period', inplace=True)
      _df.loc[:, 'period'] = 'Q4'
      _df.set_index('period', append=True, inplace=True)
      _df = _df.reorder_levels(['date', 'period', 'months'])

    df = df.combine_first(_df)

  mask = (months == 3) & (period.isin({'Q1', 'Q2', 'Q3', 'Q4'})) | (months == 12) & (
    period == 'FY'
  )

  # df.reset_index(level='months', inplace=True)
  # df.loc[df['months'] == 12,'months'] = 'a'
  # df.loc[df['months'] == 3,'months'] = 'q'
  # df.rename(columns={'months': 'scope'}, inplace=True)
  # df.set_index('scope', append=True, inplace=True)

  df = df.loc[mask, :]

  return df.copy()


def stock_splits(id_: str) -> Series[float]:
  field = 'StockholdersEquityNoteStockSplitConversionRatio1'
  query = f'SELECT data FROM {id_} WHERE json_extract(data, "$.{field}") IS NOT NULL'
  df_parse = cast(
    DataFrame[str], read_sqlite('financials_scrap', query, dtype={'data': str})
  )
  fin_data = cast(list[FinData], df_parse['data'].apply(json.loads).to_list())

  df_data: list[StockSplit] = []
  for data in fin_data:
    df_data.extend(get_stock_splits(data))

  df = pd.DataFrame(df_data)
  df.drop_duplicates(inplace=True)
  df.set_index('date', inplace=True)

  return cast(Series[float], df['stock_split_ratio'])


def upsert_financials(
  db_name: str,
  table: str,
  financials: list[RawFinancials],
):
  db_path = DB_DIR / sqlite_name(db_name)

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  cur.execute(
    f"""CREATE TABLE IF NOT EXISTS "{table}"(
    url TEXT PRIMARY KEY,
    scope TEXT,
    date TEXT,
    period TEXT,
    fiscal_end TEXT,
    currency JSON,
    data JSON
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
  cur.executemany(query, [f.model_dump() for f in financials])

  con.commit()
  con.close()


def select_financials(db_name: str, table: str) -> list[RawFinancials]:
  db_path = DB_DIR / sqlite_name(db_name)

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()
    cur.row_factory = lambda _, row: RawFinancials(**row)

    financials: list[RawFinancials] = cur.execute(f'SELECT * FROM "{table}"').fetchall()

  return financials
