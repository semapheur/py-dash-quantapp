from contextlib import closing
from datetime import date as Date
from dateutil.relativedelta import relativedelta
from enum import Enum
from functools import partial
import json
import sqlite3
from typing import cast, Optional

from asyncstdlib.functools import cache
import pandas as pd
from pandera.typing import DataFrame, Series, Index

from lib.db.lite import read_sqlite, sqlite_path
from lib.fin.models import (
  FinStatement,
  FinStatementFrame,
  Instant,
  Duration,
  StockSplit,
  FiscalPeriod,
  FinData,
)
from lib.fin.quote import load_ohlcv
from lib.fin.taxonomy import load_taxonomy_items
from lib.utils.dataframe import combine_duplicate_columns, df_time_difference
from lib.utils.time import fiscal_quarter_monthly
from lib.yahoo.ticker import Ticker


class ScopeEnum(Enum):
  quarterly = 3
  annual = 12


stock_split_items = {
  "StockSplitRatio",
  "StockholdersEquityNoteStockSplitConversionRatio",
  "StockholdersEquityNoteStockSplitConversionRatio1",
  "ShareholdersEquityNoteStockSplitConverstaionRatioAuthorizedShares",
}

statements_columns = {
  "date": "INTEGER",
  "fiscal_period": "TEXT",
  "url": "TEXT",
  "fiscal_end": "TEXT",
  "currency": "TEXT",
  "periods": "TEXT",
  "units": "TEXT",
  "synonyms": "TEXT",
  "data": "TEXT",
}
statements_table_text = (
  ",".join(f"{k} {v}" for k, v in statements_columns.items())
  + ",UNIQUE (date, fiscal_period)"
)
statements_columns_text = ",".join(statements_columns.keys())
statements_values_text = ",".join(f":{k}" for k in statements_columns.keys())


def df_to_statements(df: DataFrame[FinStatementFrame]) -> list[FinStatement]:
  return [
    FinStatement(
      url=row["url"],
      date=row["date"],
      fiscal_period=row["fiscal_period"],
      fiscal_end=row["fiscal_end"],
      currency=row["currency"],
      periods=row["periods"],
      units=row["units"],
      synonyms=row["synonyms"],
      data=row["data"],
    )
    for row in df.to_dict("records")
  ]


@cache
async def fetch_exchange_rate(
  ticker: str, start_date: Date, end_date: Date, extract_date: Optional[Date] = None
) -> float:
  from numpy import nan

  exchange_fetcher = partial(Ticker(ticker + "=X").ohlcv, start_date, end_date)
  rate = await load_ohlcv(
    ticker, "forex", exchange_fetcher, None, start_date, end_date, ["close"]
  )

  if rate.empty:
    return nan

  if extract_date is None:
    return rate["close"].mean()

  return rate.resample("D").ffill().at[pd.to_datetime(extract_date), "close"]


async def statement_to_df(
  financials: FinStatement,
  currency: Optional[str] = None,
  multiple=False,
) -> DataFrame:
  def get_date(period: Instant | Duration) -> Date:
    return period.end_date if isinstance(period, Duration) else period.instant

  async def exchange_rate(
    currency: str, unit: str, period: Instant | Duration
  ) -> float:
    ticker = f"{unit}{currency}".upper()

    extract_date = period.instant if isinstance(period, Instant) else None
    start_date = (
      extract_date - relativedelta(days=7) if extract_date else period.start_date
    )
    end_date = extract_date + relativedelta(days=7) if extract_date else period.end_date

    return await fetch_exchange_rate(ticker, start_date, end_date, extract_date)

  if isinstance(currency, str):
    currency = currency.lower()

  fin_date = pd.to_datetime(financials.date)
  fiscal_end_month = int(financials.fiscal_end.split("-")[0])
  fin_period = financials.fiscal_period
  fin_scope = "annual" if fin_period == "FY" else "quarterly"
  currencies = financials.currency.difference({currency})

  df_data: dict[tuple[Date, FiscalPeriod, int, int], dict[str, int | float]] = {}

  for item, entries in financials.data.items():
    for entry in entries:
      period_type = entry["period"][0]
      period_index = entry["period"][1:]
      period: Duration | Instant = financials.periods[period_type][period_index]
      date = pd.to_datetime(get_date(period))

      months = relativedelta(fin_date, date).months

      if (
        (date > fin_date)
        or (months % ScopeEnum[fin_scope].value != 0)
        or ((not multiple) and date != fin_date)
      ):
        continue

      if isinstance(period, Duration):
        months = period.months
      else:
        months = ScopeEnum[fin_scope].value

      if months > 12:
        continue

      quarter = fiscal_quarter_monthly(date.month, fiscal_end_month)
      period = cast(FiscalPeriod, f"Q{quarter}" if months < 12 else "FY")
      if fin_period == "FY" and months < 12:
        period = "Q4"

      value = entry.get("value")
      unit_index = entry.get("unit")
      unit = financials.units[unit_index] if unit_index is not None else ""

      if value and (currency is not None) and unit in currencies:
        value *= await exchange_rate(currency, unit, period)

      index = (date, period, months, fiscal_end_month)
      df_data.setdefault(index, {})[item] = value

      if fin_period == "FY" and (isinstance(period, Instant) or unit == "shares"):
        index = (date, "Q4", 3, fiscal_end_month)
        df_data.setdefault(index, {})[item] = value

      members = entry.get("members")
      if members is None:
        continue

      for member, m_entry in members.items():
        m_value = m_entry.get("value")
        if m_value is None:
          continue

        unit_index = m_entry.get("unit")
        m_unit = financials.units[unit_index] if unit_index is not None else ""
        if (currency is not None) and m_unit in currencies:
          m_value *= await exchange_rate(currency, m_unit, period)

        dim = "." + d if (d := m_entry.get("dim")) else ""
        key = f"{item}{dim}.{member}"
        index = (date, period, months, fiscal_end_month)
        df_data.setdefault(index, {})[key] = m_value

        if fin_period == "FY" and (isinstance(period, Instant) or m_unit == "shares"):
          index = (date, "Q4", 3, fiscal_end_month)
          df_data.setdefault(index, {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient="index")
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ["date", "period", "months", "fiscal_end_month"]
  return cast(DataFrame, df)


async def load_statements(id: str, currency: Optional[str] = None) -> DataFrame | None:
  statements = load_raw_statements(id)

  if statements is None:
    return None

  dfs = [await statement_to_df(statements.pop(0), currency, True)] + [
    await statement_to_df(s, currency) for s in statements
  ]
  df = pd.concat(dfs, join="outer")

  df.sort_index(level=0, ascending=True, inplace=True)
  df = cast(
    DataFrame, df.loc[df.index.get_level_values("months").isin((12, 9, 6, 3)), :]
  )
  df = fix_statements(df)

  return df


def fix_statements(statements: DataFrame) -> DataFrame:
  def check_combos(ix: pd.MultiIndex, conditions: set[tuple[str, int]]) -> bool:
    return conditions.issubset(set(ix.droplevel("date")))

  def quarterize(df: DataFrame) -> DataFrame:
    conditions = (("Q1", 3), ("Q2", 6), ("Q3", 9), ("FY", 12))
    period_months: Index = df.index.droplevel(["date", "fiscal_end_month"])

    for i in range(1, len(conditions)):
      mask = period_months.isin((conditions[i - 1], conditions[i]))
      df_ = df.loc[mask, diff_items].copy()
      if not check_combos(
        cast(pd.MultiIndex, df_.index), {conditions[i - 1], conditions[i]}
      ):
        continue

      df_.sort_index(level="date", inplace=True)

      df_["month_difference"] = df_time_difference(
        cast(pd.DatetimeIndex, df_.index.get_level_values("date")), 30, "D"
      )
      df_[diff_items] = df_[diff_items].diff()
      df_ = df_.loc[df_["month_difference"] == 3, diff_items]
      df_ = df_.loc[(slice(None), conditions[i][0], conditions[i][1], slice(None)), :]
      df_.reset_index(level="months", inplace=True)
      df_["months"] = 3
      df_.set_index("months", append=True, inplace=True)

      if conditions[i][0] == "FY":
        df_.reset_index(level="period", inplace=True)
        df_.loc[:, "period"] = "Q4"
        df_.set_index("period", append=True, inplace=True)
        df_ = df_.reorder_levels(["date", "period", "months"])

      df = cast(DataFrame, df.combine_first(df_))

    return cast(DataFrame, df.dropna(how="all"))

  def load_sum_items() -> set[str]:
    query = "SELECT item FROM items WHERE aggregate = 'sum'"
    sum_items = read_sqlite("taxonomy.db", query)
    if sum_items is None:
      raise ValueError("Taxonomy could not be loaded!")
    return set(sum_items["item"].tolist())

  quarter_set = {"Q1", "Q2", "Q3", "Q4"}
  items = load_taxonomy_items()

  statements.columns = statements.columns.str.lower()
  statements = cast(
    DataFrame,
    statements.loc[:, list(set(statements.columns).intersection(set(items["gaap"])))],
  )

  rename = dict(zip(items["item"], items["gaap"]))
  statements.rename(columns=rename, inplace=True)
  statements = combine_duplicate_columns(statements)

  sum_items = load_sum_items()
  diff_items = list(sum_items.intersection(set(statements.columns)))

  fiscal_ends = cast(pd.MultiIndex, statements.index).levels[3]
  for fiscal_end in fiscal_ends:
    mask = statements.index.get_level_values("fiscal_end_month") == fiscal_end
    statements = cast(
      DataFrame, statements.combine_first(quarterize(statements.loc[mask, :]))
    )

  statements.reset_index("fiscal_end_month", drop=True, inplace=True)
  statements.sort_index(level="date", inplace=True)
  period = statements.index.get_level_values("period")
  months = statements.index.get_level_values("months")

  valid_mask = ((months == 3) & period.isin(quarter_set)) | (
    (months == 12) & (period == "FY")
  )
  statements = statements.loc[valid_mask, :].dropna(how="all")

  if len(fiscal_ends) == 1:
    return cast(DataFrame, statements)

  levels = ["date", "period", "months"]
  fiscal_end = statements.index.droplevel(["period", "months"])[-1]

  fy_mask = (statements.index.get_level_values("date") < fiscal_end[0]) & (
    statements.index.get_level_values("period") == "FY"
  )
  statements = statements.loc[~fy_mask, :]

  dates = statements.index.get_level_values("date")
  statements.reset_index(level="period", inplace=True)
  for i in statements.loc[dates < fiscal_end[0], "period"].index:
    statements.at[i, "period"] = f"Q{fiscal_quarter_monthly(i[0].month, fiscal_end[1])}"

  q4_mask = (statements.index.get_level_values("date") < fiscal_end[0]) & (
    statements["period"] == "Q4"
  )

  fy = statements.loc[q4_mask, :].copy()
  fy.reset_index(level="months", inplace=True)
  fy["period"] = "FY"
  fy["months"] = 12
  fy.set_index(["period", "months"], append=True, inplace=True)
  fy = fy.reorder_levels(levels)

  statements.set_index("period", append=True, inplace=True)
  statements = cast(
    DataFrame,
    statements.reorder_levels(levels),
  )

  statements = cast(DataFrame, pd.concat((statements, fy), axis=0))
  statements.sort_index(level=["date", "period"], inplace=True)

  rolling_mask = (statements.index.get_level_values("date") < fiscal_end[0]) & (
    statements.index.get_level_values("period") != "FY"
  )

  window_size = 4
  windows = statements.loc[rolling_mask, diff_items].rolling(4)
  for i in range(len(statements) - window_size + 1):
    window = windows.get_window(i)
    if list(window.index.get_level_values("period")) != ["Q1", "Q2", "Q3", "Q4"]:
      continue

    ix = (window.index.get_level_values("date").max(), "FY", 12)
    statements.loc[ix, diff_items] = window.sum()

  return cast(DataFrame, statements)


def get_stock_splits(fin_data: FinData) -> list[StockSplit]:
  data: list[StockSplit] = []

  split_item = stock_split_items.intersection(fin_data.keys())

  if not split_item:
    return data

  splits = fin_data[split_item.pop()]

  for entry in splits:
    value = cast(float, entry.get("value"))

    data.append(
      StockSplit(
        date=cast(Duration, entry["period"]).start_date,
        stock_split_ratio=value,
      )
    )
  return data


def stock_splits(id: str) -> Series[float]:
  where_text = " AND ".join(
    [f'json_extract(data, "$.{item}") IS NOT NULL' for item in stock_split_items]
  )

  query = f'SELECT data FROM "{id}" WHERE {where_text}'
  df_parse = cast(
    DataFrame[str], read_sqlite("statements.db", query, dtype={"data": str})
  )
  if df_parse is None:
    return None

  fin_data = cast(list[FinData], df_parse["data"].apply(json.loads).to_list())

  df_data: list[StockSplit] = []
  for data in fin_data:
    df_data.extend(get_stock_splits(data))

  df = pd.DataFrame(df_data)
  df.drop_duplicates(inplace=True)
  df.set_index("date", inplace=True)

  return cast(Series[float], df["stock_split_ratio"])


def load_raw_statements(
  id: str, date: Date | None = None, order: bool = True
) -> list[FinStatement]:
  db_path = sqlite_path("statements.db")

  query = f"SELECT * FROM '{id}'"
  params = []
  if date:
    query += " WHERE date >= ?"
    params.append(int(date.strftime("%Y%m%d")))

  if order:
    query += " ORDER BY date ASC"

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()
    cur.row_factory = lambda _, row: FinStatement(**row)

    statements: list[FinStatement] = cur.execute(query, params).fetchall()

  return statements


def load_raw_statement(id: str, date: int, period: FiscalPeriod) -> FinStatement | None:
  db_path = sqlite_path("statements.db")

  query = f"""
    SELECT * FROM '{id}' 
    WHERE date = ? AND fiscal_period = ?
  """
  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    table_exists = cur.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (id,)
    ).fetchone()
    if table_exists is None:
      return None

    cur.row_factory = sqlite3.Row
    statement = cur.execute(query, (date, period)).fetchone()

    if statement is None:
      return None

    return FinStatement(**dict(statement))


def store_updated_statements(
  db_name: str,
  table: str,
  statements: list[FinStatement],
):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}'({statements_table_text})")

    query = f"""
      UPDATE "{table}" SET
        url = :url,
        fiscal_end = :fiscal_end,
        currency = :currency,
        data = :data
      WHERE date = :date AND fiscal_period = :fiscal_period
    """
    cur.executemany(query, [s.model_dump_json() for s in statements])
    con.commit()


def insert_statements(
  db_name: str,
  table: str,
  statements: list[FinStatement],
):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}'({statements_columns})")

    query = f"""
      INSERT INTO "{table}" VALUES ({statements_values_text})
    """
    cur.executemany(query, [s.dump_json_values() for s in statements])
    con.commit()


def upsert_merged_statements(
  db_name: str,
  table: str,
  statements: list[FinStatement],
):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}'({statements_table_text})")

    for statement in statements:
      date = int(statement.date.strftime("%Y%m%d"))
      fiscal_period = statement.fiscal_period

      old_statement = load_raw_statement(table, date, fiscal_period)
      if old_statement is not None:
        old_statement.merge(statement)
      else:
        old_statement = statement

      query = f"""
        INSERT INTO "{table}"
        VALUES ({statements_values_text})
        ON CONFLICT(date, fiscal_period) DO UPDATE SET
          url = excluded.url,
          currency = excluded.currency,
          periods = excluded.periods,
          units = excluded.units,
          synonyms = excluded.synonyms,
          data = excluded.data
      """
      statement_json = old_statement.dump_json_values()
      cur.execute(
        query,
        {
          "date": date,
          "fiscal_period": fiscal_period,
          "fiscal_end": statement_json["fiscal_end"],
          "url": statement_json["url"],
          "currency": statement_json["currency"],
          "periods": statement_json["periods"],
          "units": statement_json["units"],
          "synonyms": statement_json["synonyms"],
          "data": statement_json["data"],
        },
      )
      con.commit()


def upsert_statements(
  db_name: str,
  table: str,
  statements: list[FinStatement],
):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}'({statements_columns})")

    query = f"""INSERT INTO 
      "{table}" VALUES ({statements_values_text})
      ON CONFLICT (date, fiscal_period) DO UPDATE SET
        data=json_patch(data, excluded.data),
        url=(
          SELECT json_group_array(value)
          FROM (
            SELECT json_each.value
            FROM json_each(url)
            WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded.url))
          )
        ),
        currency=(
          SELECT json_group_array(value)
          FROM (
            SELECT json_each.value
            FROM json_each(currency)
            WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded.currency))
          )
        )
    """
    cur.executemany(query, [s.dump_json_values() for s in statements])
    con.commit()


def statement_urls(
  db_name: str, id: str, url_pattern: str | None = None
) -> DataFrame | None:
  query = f"""
    SELECT date, json_each.value AS url FROM '{id}' 
    JOIN json_each(url) ON 1=1
  """

  if url_pattern:
    query += f" WHERE json_each.value LIKE '{url_pattern}'"

  df = read_sqlite(
    db_name,
    query,
    date_parser={"date": {"format": "%Y%m%d"}},
  )
  return df
