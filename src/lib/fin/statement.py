import asyncio
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
import polars as pl

from lib.db.lite import read_sqlite, sqlite_path
from lib.fin.models import (
  FinStatement,
  FinStatementFrame,
  Instant,
  Duration,
  StockSplit,
  FiscalPeriod,
  FinData,
  get_date,
)
from lib.fin.quote import load_ohlcv
from lib.fin.taxonomy import load_taxonomy_items
from lib.utils.dataframe import (
  combine_duplicate_columns,
  df_time_difference,
  fiscal_quarter_monthly_polars,
)
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
  "sources": "TEXT",
  "fiscal_end": "TEXT",
  "currencies": "TEXT",
  "periods": "TEXT",
  "units": "TEXT",
  "dimensions": "TEXT",
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
      sources=row["sources"],
      date=row["date"],
      fiscal_period=row["fiscal_period"],
      fiscal_end=row["fiscal_end"],
      currencies=row["currencies"],
      periods=row["periods"],
      units=row["units"],
      dimensions=row["dimensions"],
      synonyms=row["synonyms"],
      data=row["data"],
    )
    for row in df.to_dict("records")
  ]


@cache
async def fetch_exchange_rate(
  ticker: str, start_date: Date, end_date: Date, extract_date: Date | None = None
) -> float:
  from numpy import nan

  exchange_fetcher = partial(Ticker(ticker + "=X").ohlcv, period="max")
  rate = await load_ohlcv(
    ticker, "forex", exchange_fetcher, None, start_date, end_date, ["close"]
  )

  if rate.is_empty():
    return nan

  if extract_date is None:
    return rate["close"].mean()

  rate_daily = (
    rate.group_by_dynamic("date", every="1d", period="1d")
    .agg([pl.col("close").last()])
    .sort("date")
    .with_columns(pl.col("close").fill_null(strategy="forward"))
  )

  filtered = rate_daily.filter(pl.col("date") == pl.lit(extract_date).cast(pl.Date))
  if filtered.is_empty():
    return nan

  return filtered["close"].item()


@cache
async def fetch_exchange_rate_pandas(
  ticker: str, start_date: Date, end_date: Date, extract_date: Date | None = None
) -> float:
  from numpy import nan

  exchange_fetcher = partial(Ticker(ticker + "=X").ohlcv, period="max")
  rate = await load_ohlcv(
    ticker, "forex", exchange_fetcher, None, start_date, end_date, ["close"]
  )

  if rate.empty():
    return nan

  if extract_date is None:
    return rate["close"].mean()

  return rate.resample("D").ffill().at[pd.to_datetime(extract_date), "close"]


async def exchange_rate(currency: str, unit: str, period: Instant | Duration) -> float:
  ticker = f"{unit}{currency}".upper()

  extract_date = period.instant if isinstance(period, Instant) else None
  start_date = (
    extract_date - relativedelta(days=7) if extract_date else period.start_date
  )
  end_date = extract_date + relativedelta(days=7) if extract_date else period.end_date

  return await fetch_exchange_rate(ticker, start_date, end_date, extract_date)


async def statement_to_lf(
  financials: FinStatement,
  currency: str | None = None,
  multiple=False,
) -> pl.LazyFrame:
  def _put(row: dict, key: str, value: float | None, unit: str):
    if value is None:
      return

    if currency is not None and unit in currencies_other:
      row[key] = None
      rate_slots.append((row, key))
      rate_tasks.append(asyncio.create_task(exchange_rate(currency, unit, period)))
    else:
      row[key] = value

  if isinstance(currency, str):
    currency = currency.lower()

  fin_date = financials.date + relativedelta(days=1)
  fiscal_end_month = int(financials.fiscal_end.split("-")[0])
  fin_scope = "annual" if financials.fiscal_period == "FY" else "quarterly"
  currencies_other = financials.currencies.difference({currency})

  rows: list[dict] = []
  rate_tasks: list[asyncio.Task] = []
  rate_slots: list[tuple[dict, str]] = []

  scope_months = ScopeEnum[fin_scope].value

  for item, records in financials.data.items():
    for period, record in records.items():
      date = get_date(period)
      months_delta = relativedelta(fin_date, date).months

      if (
        (date > fin_date)
        or (months_delta % scope_months != 0)
        or ((not multiple) and date != fin_date)
      ):
        continue

      months_len = period.months if isinstance(period, Duration) else scope_months
      if months_len > 12:
        continue

      qtr = fiscal_quarter_monthly(date.month, fiscal_end_month)
      fin_period = cast(FiscalPeriod, f"Q{qtr}" if months_len < 12 else "FY")
      if fin_period == "FY" and months_len < 12:
        fin_period = "Q4"

      base_row = {
        "date": date,
        "period": fin_period,
        "months": months_len,
        "fiscal_end_month": fiscal_end_month,
      }
      row = base_row.copy()
      _put(row, item, record.value, record.unit)
      rows.append(row)

      if fin_period == "FY" and (
        isinstance(period, Instant) or record.unit == "shares"
      ):
        row_q4 = base_row.copy()
        row_q4["period"] = "Q4"
        _put(row_q4, item, record.value, record.unit)
        rows.append(row_q4)

      if record.members:
        for member, m in record.members.items():
          key = f"{item}{('.' + m.dim) if m.dim else ''}.{member}"
          _put(row, key, m.value, m.unit)
          if fin_period == "FY" and (isinstance(period, Instant) or m.unit == "shares"):
            row_q4m = base_row.copy()
            row_q4m["period"] = "Q4"
            _put(row_q4m, key, m.value, m.unit)
            rows.append(row_q4m)

  if rate_tasks:
    rates = await asyncio.gather(*rate_tasks)
    for (row, key), rate in zip(rate_slots, rates):
      row[key] = (row.get(key, 1.0) or 1.0) * rate

  lf = pl.LazyFrame(rows)

  lf = lf.with_columns(
    pl.col("date").cast(pl.Date),
    pl.col("period").cast(pl.Utf8),
    pl.col("months").cast(pl.Int8),
    pl.col("fiscal_end_month").cast(pl.Int8),
  )

  lf = lf.unique(subset=["date", "period", "months", "fiscal_end_month"])

  return lf


async def statement_to_df(
  financials: FinStatement,
  currency: str | None = None,
  multiple=False,
) -> DataFrame:
  if isinstance(currency, str):
    currency = currency.lower()

  fin_date = pd.to_datetime(financials.date)
  fiscal_end_month = int(financials.fiscal_end.split("-")[0])
  fin_period = financials.fiscal_period
  fin_scope = "annual" if fin_period == "FY" else "quarterly"
  currencies = financials.currencies.difference({currency})

  df_data: dict[tuple[Date, FiscalPeriod, int, int], dict[str, int | float]] = {}

  for item, records in financials.data.items():
    for period, record in records.items():
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
      fin_period = cast(FiscalPeriod, f"Q{quarter}" if months < 12 else "FY")
      if fin_period == "FY" and months < 12:
        fin_period = "Q4"

      value = record.value
      unit = record.unit

      if value and (currency is not None) and unit in currencies:
        value *= await exchange_rate(currency, unit, period)

      index = (date, fin_period, months, fiscal_end_month)
      df_data.setdefault(index, {})[item] = value

      if fin_period == "FY" and (isinstance(period, Instant) or unit == "shares"):
        index = (date, "Q4", 3, fiscal_end_month)
        df_data.setdefault(index, {})[item] = value

      members = record.members
      if members is None:
        continue

      for member, m_entry in members.items():
        m_value = m_entry.value
        if m_value is None:
          continue

        m_unit = m_entry.unit
        if (currency is not None) and m_unit in currencies:
          m_value *= await exchange_rate(currency, m_unit, period)

        dim = "." + d if (d := m_entry.dim) else ""
        key = f"{item}{dim}.{member}"
        index = (date, fin_period, months, fiscal_end_month)
        df_data.setdefault(index, {})[key] = m_value

        if fin_period == "FY" and (isinstance(period, Instant) or m_unit == "shares"):
          index = (date, "Q4", 3, fiscal_end_month)
          df_data.setdefault(index, {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient="index")
  df.index = pd.MultiIndex.from_tuples(
    list(df.index), names=["date", "period", "months", "fiscal_end_month"]
  )
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


def load_sum_items() -> set[str]:
  query = "SELECT item FROM items WHERE aggregate = 'sum'"
  sum_items = read_sqlite("taxonomy.db", query)
  if sum_items is None:
    raise ValueError("Taxonomy could not be loaded!")
  return set(sum_items["item"].tolist())


def fix_statements(statements: pl.LazyFrame) -> pl.DataFrame:
  def check_combos(lf: pl.LazyFrame, conditions: set[tuple[str, int]]) -> bool:
    unique_combos = (
      lf.select(
        [
          pl.col("period"),
          pl.col("months"),
        ]
      )
      .unique()
      .collect()
      .rows()
    )
    return conditions.issubset(set(unique_combos))

  def quarterize(lf: pl.LazyFrame) -> pl.LazyFrame:
    conditions = (("Q1", 3), ("Q2", 6), ("Q3", 9), ("FY", 12))

    result_lf = lf
    lf_columns = set(lf.collect_schema().names())
    for i in range(1, len(conditions)):
      mask_lf = lf.filter(
        (pl.col("period").is_in([conditions[i - 1][0], conditions[i][0]]))
        & (pl.col("months").is_in([conditions[i - 1][1], conditions[i][1]]))
      )
      if not check_combos(mask_lf, {conditions[i - 1], conditions[i]}):
        continue

      lf_sorted = mask_lf.sort("date")
      lf_with_diff = lf_sorted.with_columns(
        df_time_difference("date", 30, "D").alias("month_difference")
      )
      lf_filtered = lf_with_diff.filter(
        (pl.col("month_difference") == 3)
        & (pl.col("period") == conditions[i][0])
        & (pl.col("months") == conditions[i][1])
      )
      diff_cols = lf_columns.intersection(diff_items)
      diff_exprs = [pl.col(col).diff().alias(col) for col in diff_cols]
      nondiff_cols = lf_columns.difference(diff_items)
      non_diff_exprs = [pl.col(col) for col in nondiff_cols]

      lf_diffed = lf_filtered.select(diff_exprs + non_diff_exprs)
      lf_processed = lf_diffed.with_columns(pl.lit(3).alias("months"))

      if conditions[i][0] == "FY":
        lf_processed = lf_processed.with_columns(pl.lit("Q4").alias("period"))

      result_lf = result_lf.join(
        lf_processed, on=["date", "period", "months"], how="outer", coalesce=True
      )

    meta_cols = {"date", "period", "months", "fiscal_end_month"}
    data_cols = set(result_lf.collect_schema().names()).difference(meta_cols)
    if data_cols:
      result_lf = result_lf.filter(
        pl.any_horizontal([pl.col(col).is_not_null() for col in data_cols])
      )

    return result_lf

  quarter_set = {"Q1", "Q2", "Q3", "Q4"}
  items = load_taxonomy_items()

  statements = statements.with_columns(
    [pl.col(c).alias(c.lower()) for c in statements.collect_schema().names()]
  )

  available_gaap_cols = set(statements.collect_schema().names()).intersection(items)
  statements = statements.select(available_gaap_cols)

  rename_mapping = dict(zip(items["items"], items["gaap"]))
  rename_exprs = [
    pl.col(old_name).alias(new_name) if old_name in rename_mapping else pl.col(old_name)
    for old_name in statements.columns
    for new_name in [rename_mapping.get(old_name, old_name)]
  ]
  statements = statements.with_columns(rename_exprs)

  sum_items = load_sum_items()
  diff_items = list(sum_items.intersection(set(statements.collect_schema().names())))

  fiscal_ends = (
    statements.select("fiscal_end_month")
    .unique()
    .collect()["fiscal_end_month"]
    .to_list()
  )

  result_frames = []
  for fiscal_end in fiscal_ends:
    fiscal_mask = pl.col("fiscal_end_month") == fiscal_end
    fiscal_data = statements.filter(fiscal_mask)
    quarterized = quarterize(fiscal_data)
    result_frames.append(quarterized)

  if result_frames:
    statements = pl.concat(result_frames)

  statements = statements.drop("fiscal_end_month").sort("date")

  valid_mask = ((pl.col("months") == 3) & pl.col("period").is_in(quarter_set)) | (
    (pl.col("months") == 12) & (pl.col("period") == "FY")
  )
  statements = statements.filter(valid_mask)

  meta_cols = {"date", "period", "months"}
  data_cols = set(set(statements.collect_schema().names())).difference(meta_cols)
  statements = statements.filter(
    pl.any_horizontal([pl.col(c).is_not_null() for c in data_cols])
  )
  if len(fiscal_ends) == 1:
    return statements.collect()

  last_fiscal_end = statements.select("date").max().collect()["date"][0]

  fy_mask = (pl.col("date") < last_fiscal_end) & (pl.col("period") == "FY")
  statements = statements.filter(~fy_mask)

  statements = statements.with_columns(
    [
      pl.when(pl.col("date") < last_fiscal_end)
      .then(
        pl.format("Q{}", fiscal_quarter_monthly_polars(pl.col("date"), fiscal_ends[-1]))
      )
      .otherwise(pl.col("period"))
      .alias("period")
    ]
  )

  q4_mask = (pl.col("date") <= last_fiscal_end) & (pl.col("period") == "Q4")
  fy_records = statements.filter(q4_mask).with_columns(
    [pl.lit("FY").alias("period"), pl.lit(12).alias("months")]
  )

  statements = pl.concat([statements, fy_records])
  statements = statements.sort(["date", "period"])

  rolling_mask = (pl.col("date") < last_fiscal_end) & (pl.col("period") != "FY")
  rolling_data = statements.filter(rolling_mask).sort("date")

  rolling_data = rolling_data.with_columns(
    [
      *[
        pl.col(c).rolling_sum(window_size=4, min_samples=4).alias(f"__{c}_fy")
        for c in diff_items
      ],
      pl.col("period").shift(1).alias("__p1"),
      pl.col("period").shift(2).alias("__p2"),
      pl.col("period").shift(3).alias("__p3"),
    ]
  )

  full_year_condition = (
    (pl.col("period") == "Q4")
    & (pl.col("__p1") == "Q3")
    & (pl.col("__p2") == "Q2")
    & (pl.col("__p3") == "Q1")
  )

  rolling_data = rolling_data.with_columns(full_year_condition.alias("__is_full_year"))

  rolling_cols = set(rolling_data.collect_schema().names())
  non_diff_cols = rolling_cols.difference(meta_cols.union(diff_items))
  fy_rows = rolling_data.filter(pl.col("__is_full_year")).select(
    pl.col("date"),
    pl.lit("FY").alias("period"),
    pl.lit(12).alias("months"),
    *[pl.col(c) for c in non_diff_cols if not c.startswith("__")],
    *[pl.col(f"__{c}_fy").alias(c) for c in diff_items],
  )

  statements = pl.concat([statements, fy_rows]).sort(["date", "period"])

  return statements.collect()


def fix_statements_pandas(statements: DataFrame) -> DataFrame:
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

  def sum_if_complete_year(window):
    periods = list(window.index.get_level_values("period"))
    if len(periods) == 4 and set(periods) == quarter_set:
      return window.sum()

    return None

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

  quarterly_data = statements.loc[rolling_mask, diff_items].sort_index()
  rolling_sums = quarterly_data.rolling(window=4).apply(sum_if_complete_year, raw=False)

  for idx, values in rolling_sums.dropna().iterrows():
    date = idx[0]
    fy_idx = (date, "FY", 12)
    statements.loc[fy_idx, diff_items] = values

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

    cur.row_factory = lambda cursor, row: sqlite3.Row(cursor, tuple(row))
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
        sources = :sources,
        fiscal_end = :fiscal_end,
        currencies = :currencies,
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
        statement = old_statement

      query = f"""
        INSERT INTO "{table}"
        VALUES ({statements_values_text})
        ON CONFLICT(date, fiscal_period) DO UPDATE SET
          sources = excluded.sources,
          currencies = excluded.currencies,
          periods = excluded.periods,
          units = excluded.units,
          dimensions = excluded.dimensions,
          synonyms = excluded.synonyms,
          data = excluded.data
      """
      cur.execute(
        query,
        statement.model_dump(),
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
    cur.executemany(query, [s.model_dump() for s in statements])
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
