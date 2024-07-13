from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
from typing import cast, Optional
from sqlalchemy.types import Date

from ordered_set import OrderedSet
from numpy import inf, nan
import pandas as pd
from pandera.typing import DataFrame, Index, Series

from lib.db.lite import read_sqlite, upsert_sqlite, get_table_columns
from lib.fin.calculation import calculate_items, trailing_twelve_months
from lib.fin.metrics import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.fin.models import CloseQuote, SharePrice
from lib.fin.statement import load_statements, stock_splits
from lib.fin.quote import get_ohlcv
from lib.fin.taxonomy import TaxonomyCalculation
from lib.morningstar.ticker import Stock
from lib.yahoo.ticker import Ticker


def load_schema(query: Optional[str] = None) -> dict[str, TaxonomyCalculation]:
  if query is None:
    query = """
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    """

  df = read_sqlite("taxonomy.db", query)
  if df is None:
    raise ValueError("Could not load taxonomy!")

  df.loc[:, "calculation"] = df["calculation"].apply(lambda x: json.loads(x))
  schema = {k: TaxonomyCalculation(**v) for k, v in zip(df["item"], df["calculation"])}
  return schema


def stock_split_adjust(financials: DataFrame, ratios: Series | None) -> DataFrame:
  price_cols = {
    "share_price_close",
    "share_price_open",
    "share_price_high",
    "share_price_low",
    "share_price_average",
  }
  price_cols_ = list(price_cols.intersection(financials.columns))

  for i, col in enumerate(price_cols_):
    financials[f"{col}_adjusted"] = financials[col]
    price_cols_[i] = f"{col}_adjusted"

  if ratios is None:
    return financials

  for date, ratio in ratios.items():
    mask = financials.index.get_level_values("date") < date
    financials.loc[mask, price_cols_] /= ratio

  return financials


def merge_share_price(financials: DataFrame, price: DataFrame[CloseQuote]) -> DataFrame:
  def weighted_share_price(price_slice: DataFrame) -> DataFrame:
    wasob = "weighted_average_shares_outstanding_basic"

    for id in price_slice.columns:
      price_slice.loc[:, id] *= (
        financials.loc[ix, f"{wasob}.{id}"] / financials.loc[ix, wasob]
      )
    price_slice["close"] = price_slice.sum(axis=1)
    return price_slice

  price_records = [
    SharePrice(
      share_price_close=nan,
      share_price_open=nan,
      share_price_high=nan,
      share_price_low=nan,
      share_price_average=nan,
    )
  ] * len(financials)

  if len(price.columns) == 1:
    price.rename(columns={price.columns[0]: "close"}, inplace=True)

  price.sort_index(inplace=True)
  for i, ix in enumerate(cast(pd.MultiIndex, financials.index)):
    end_date = cast(dt, ix[0])
    start_date = end_date - relativedelta(months=cast(int, ix[2]))

    price_ = cast(DataFrame, price.loc[start_date:end_date].copy().sort_index())
    if price_.empty:
      continue

    if len(price.columns) > 1:
      price_ = weighted_share_price(price_)

    price_records[i] = SharePrice(
      share_price_close=price_["close"].iloc[-1],
      share_price_open=price_["close"].iloc[0],
      share_price_high=price_["close"].max(),
      share_price_low=price_["close"].min(),
      share_price_average=price_["close"].mean(),
    )

  price_data = pd.DataFrame.from_records(price_records, index=financials.index)

  return cast(DataFrame, pd.concat((financials, price_data), axis=1))


async def calculate_fundamentals(
  ticker_ids: list[str],
  currency: str,
  financials: DataFrame,
  beta_years: Optional[int] = None,
  update: bool = False,
) -> DataFrame:
  start_date: dt = cast(pd.MultiIndex, financials.index).levels[
    0
  ].min() - relativedelta(years=beta_years or 1)

  prices: list[DataFrame[CloseQuote]] = []
  for id in ticker_ids:
    ohlcv_fetcher = partial(Stock(id, currency).ohlcv)

    price = cast(
      DataFrame[CloseQuote],
      await get_ohlcv(
        f"{id}_{currency}",
        "stock",
        ohlcv_fetcher,
        start_date=start_date,
        cols=["close"],
      ),
    )
    price.rename(columns={"close": id}, inplace=True)
    prices.append(price)

  price_df = cast(DataFrame, pd.concat(prices, axis=1))

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

  wasob = "weighted_average_shares_outstanding_basic"
  if len(ticker_ids) > 1 and wasob not in financials.columns:
    share_cols = [f"{wasob}.{id}" for id in ticker_ids]
    financials[wasob] = financials.loc[:, share_cols].sum(axis=1)

  financials = merge_share_price(financials, price_df)
  split_ratios = stock_splits(id)
  financials = stock_split_adjust(financials, split_ratios)
  schema = load_schema()
  financials = calculate_items(financials, schema)
  # financials.to_csv(f"{ticker_ids[0]}_financials.csv")

  market_fetcher = partial(Ticker("^GSPC").ohlcv)
  market_close = cast(
    DataFrame[CloseQuote],
    await get_ohlcv(
      "GSPC", "index", market_fetcher, start_date=start_date, cols=["close"]
    ),
  )
  riskfree_fetcher = partial(Ticker("^TNX").ohlcv)
  riskfree_rate = cast(
    DataFrame[CloseQuote],
    await get_ohlcv(
      "TNX", "index", riskfree_fetcher, start_date=start_date, cols=["close"]
    ),
  )

  price_df["close"] = price_df.mean(axis=1)

  financials = beta(
    financials,
    price_df["close"].rename("equity_return").pct_change(),
    market_close["close"].rename("market_return").pct_change(),
    riskfree_rate["close"].rename("riskfree_rate") / 100,
  )

  financials = weighted_average_cost_of_capital(financials)
  financials = f_score(financials)
  financials = m_score(financials)

  if "altman_z_score" not in set(financials.columns):
    financials = z_score(financials)

  return financials


def handle_ttm(df: DataFrame) -> DataFrame:
  if "TTM" not in cast(pd.MultiIndex, df.index).levels[1].unique():
    return df

  df.sort_index(level="date", inplace=True)
  mask = (slice(None), slice("TTM"), 12)
  drop_index = df.loc[mask, :].tail(-2).index[0]
  ttm_index = df.loc[mask, :].tail(1).index[0]

  df.drop(drop_index, inplace=True)
  df.rename(index={ttm_index: (dt(1900, 1, 1))}, inplace=True)

  return df


def load_ttm(df: pd.DataFrame) -> pd.DataFrame:
  ttm_date = cast(pd.MultiIndex, df.index).levels[0].max()

  renamer = {(dt(1900, 1, 1), "TTM", 12): (ttm_date, "TTM", 12)}
  df.rename(index=renamer, inplace=True)
  return df


def load_financials(
  id: str, currency: str, columns: Optional[set[str]] = None
) -> DataFrame | None:
  col_text = "*"
  index_col = OrderedSet(("date", "period", "months"))
  table = f"{id}_{currency}"
  if columns is not None:
    table_columns = get_table_columns("financials.db", [table])
    select_columns = set(table_columns[table]).intersection(columns).union(index_col)
    col_text = ", ".join(select_columns)

  query = f"SELECT {col_text} FROM '{table}'"
  df = read_sqlite(
    "financials.db",
    query,
    index_col=list(index_col),
    date_parser={"date": {"format": "%Y-%m-%d"}},
  )
  return df


def load_ratios(id: str, columns: Optional[set[str]] = None) -> DataFrame | None:
  col_text = "*"
  index_col = OrderedSet(("date", "period", "months"))
  if columns is not None:
    table_columns = get_table_columns("financials.db", [id])
    select_columns = set(table_columns[id]).intersection(columns).union(index_col)
    col_text = ", ".join(select_columns)

  query = f"SELECT {col_text} FROM '{id}'"
  df = read_sqlite(
    "ratios.db",
    query,
    index_col=list(index_col),
    date_parser={"date": {"format": "%Y-%m-%d"}},
  )
  return df


def load_fundamentals(
  id: str,
  currency: str,
) -> DataFrame | None:
  financials = load_financials(id, currency)
  ratios = load_ratios(id)

  if financials is None or ratios is None:
    return None

  return cast(DataFrame, pd.concat([financials, ratios], axis=1))


def load_ratio_items() -> DataFrame:
  query = """SELECT item FROM items 
    WHERE unit IN ("monetary_ratio", "price_ratio", "numeric_score")
  """
  df = read_sqlite("taxonomy.db", query)
  if df is None:
    raise ValueError("Ratio items not found")

  return df


async def update_fundamentals(
  company_id: str,
  ticker_ids: list[str],
  currency: str,
  beta_years: Optional[None] = None,
) -> pd.DataFrame:
  table = f"{company_id}_{currency}"

  statements = await load_statements(company_id, currency)
  if statements is None:
    raise ValueError(f"Statements have not been seeded for {company_id}")

  fundamentals = load_fundamentals(company_id, currency)

  ratio_items = load_ratio_items()
  ratio_cols = ratio_items["item"].tolist()

  if fundamentals is None:
    fundamentals = await calculate_fundamentals(
      ticker_ids, currency, statements, beta_years
    )

    fundamentals.replace([inf, -inf], nan, inplace=True)
    ratio_cols_ = fundamentals.columns.intersection(ratio_cols)
    ratios = fundamentals[ratio_cols_]
    financials = fundamentals.drop(ratio_cols_, axis=1)

    upsert_sqlite(financials, "financials.db", table, {"date": Date})
    upsert_sqlite(ratios, "fundamentals.db", company_id, {"date": Date})
    return fundamentals

  last_statement: dt = cast(pd.MultiIndex, statements.index).levels[0].max()
  last_fundamentals: dt = cast(pd.MultiIndex, fundamentals.index).levels[0].max()
  if last_fundamentals >= last_statement:
    return fundamentals

  statements = cast(
    DataFrame,
    statements.loc[statements.index.get_level_values("date") > last_fundamentals],
  )

  fundamentals.sort_index(level="date", inplace=True)
  props = {3: (None, 8), 12: ("FY", 1)}
  months = cast(Index[int], cast(pd.MultiIndex, statements.index).levels[2].unique())
  for m in months:
    mask = (slice(None), slice(props[m][0]), m)
    fundamentals_ = cast(
      DataFrame,
      pd.concat((fundamentals.loc[mask, :].tail(props[m][1]), statements), axis=0),
    )

  fundamentals_ = await calculate_fundamentals(
    ticker_ids,
    currency,
    fundamentals_,
    beta_years,
    True,
  )
  fundamentals_ = cast(
    DataFrame, fundamentals_.loc[fundamentals_.index.difference(fundamentals.index), :]
  )

  fundamentals_.replace([inf, -inf], nan, inplace=True)
  ratio_cols_ = fundamentals_.columns.intersection(ratio_cols)
  ratios = fundamentals_[ratio_cols_]
  financials = fundamentals_.drop(ratio_cols_, axis=1)

  upsert_sqlite(financials, "financials.db", table, {"date": Date})
  upsert_sqlite(ratios, "fundamentals.db", company_id, {"date": Date})
  return cast(DataFrame, pd.concat((fundamentals, fundamentals_), axis=0))
