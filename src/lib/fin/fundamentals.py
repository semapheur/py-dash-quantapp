from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast
from ordered_set import OrderedSet
from sqlalchemy.types import Date

from numpy import inf
import orjson
import pandas as pd
from pandera.typing import DataFrame, Index, Series
import polars as pl

from lib.db.lite import read_sqlite, upsert_sqlite, select_sqlite, polars_from_sqlite
from lib.fin.calculation_pandas import calculate_items, trailing_twelve_months
from lib.fin.metrics_pandas import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.fin.models import CloseQuote, SharePrice
from lib.fin.statement import load_statements, stock_splits
from lib.fin.quote import load_ohlcv
from lib.fin.taxonomy import TaxonomyCalculation
from lib.morningstar.ticker import Stock
from lib.yahoo.ticker import Ticker


def load_schema(query: str | None = None) -> dict[str, TaxonomyCalculation]:
  if query is None:
    query = """
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    """

  df = polars_from_sqlite("taxonomy.db", query)
  if df is None or df.is_empty():
    raise ValueError("Could not load taxonomy!")

  calculations = [orjson.loads(x) for x in df["calculation"]]
  items = df.get_column("item").to_list()

  schema = {
    item: TaxonomyCalculation(**calc) for item, calc in zip(items, calculations)
  }
  return schema


def stock_split_adjust(
  financials: pl.LazyFrame, ratios: dict[str, float] | None
) -> pl.LazyFrame:
  price_cols = {
    "share_price_close",
    "share_price_open",
    "share_price_high",
    "share_price_low",
    "share_price_average",
  }
  price_cols_ = list(price_cols.intersection(financials.collect_schema().names()))

  for col in price_cols_:
    financials = financials.with_columns(pl.col(col).alias(f"{col}_adjusted"))

  if ratios is None:
    return financials

  for date, ratio in ratios.items():
    financials = financials.with_columns(
      [
        pl.when(pl.col("date") < date)
        .then(pl.col(f"{col}_adjusted") / ratio)
        .otherwise(pl.col(f"{col}_adjusted"))
        .alias(f"{col}_adjusted")
        for col in price_cols_
      ]
    )

  return financials


def stock_split_adjust_pandas(
  financials: DataFrame, ratios: Series | None
) -> DataFrame:
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


def merge_share_price(financials: pl.LazyFrame, price: pl.DataFrame) -> pl.LazyFrame:
  wasob = "weighted_average_shares_outstanding_basic"
  price_lf = price.lazy().sort("date")
  financials_indexed = financials.with_row_index("financials_ix")

  date_ranges = financials_indexed.select(
    [
      "financials_ix",
      "date",
      "months",
      (pl.col("date") - pl.duration(days=pl.col("months") * 30)).alias("start_date"),
      pl.col("date").alias("end_date"),
    ]
  )

  price_indexed = price_lf.with_row_index("price_ix")
  price_filtered = price_indexed.join(date_ranges, how="cross").filter(
    (pl.col("date") >= pl.col("start_date")) & (pl.col("date") <= pl.col("end_date"))
  )
  price_cols = [col for col in price.columns if col != "date"]

  if len(price_cols) > 1:
    wasob_cols = [col for col in financials.collect_schema().names() if wasob in col]

    if wasob_cols:
      price_weighted = price_filtered.join(
        financials_indexed.select(["financials_ix"] + wasob_cols), on="financials_ix"
      )

      weighted_expressions: list[pl.Expr] = []
      for col in price_cols:
        weight_col = f"{wasob}.{col}"
        if weight_col in wasob_cols:
          weight_expr = (pl.col(col) * pl.col(weight_col) / pl.col(wasob)).alias(
            f"weighted_{col}"
          )
        else:
          weight_expr = pl.col(col).alias(f"weighted_{col}")

        weighted_expressions.append(weight_expr)

      price_filtered = price_weighted.with_columns(weighted_expressions).with_columns(
        pl.sum_horizontal([f"weighted_{col}" for col in price_cols]).alias("close")
      )

  price_aggregated = price_filtered.group_by("financials_ix").agg(
    [
      pl.col("close").last().alias("share_price_close"),
      pl.col("close").first().alias("share_price_open"),
      pl.col("close").max().alias("share_price_high"),
      pl.col("close").min().alias("share_price_low"),
      pl.col("close").mean().alias("share_price_average"),
    ]
  )
  result = financials_indexed.join(
    price_aggregated, on="financials_ix", how="left"
  ).drop("financials_ix")

  result = result.with_columns(
    [
      pl.col("share_price_close").fill_null(float("nan")),
      pl.col("share_price_open").fill_null(float("nan")),
      pl.col("share_price_high").fill_null(float("nan")),
      pl.col("share_price_low").fill_null(float("nan")),
      pl.col("share_price_average").fill_null(float("nan")),
    ]
  )

  return result


def merge_share_price_pandas(
  financials: DataFrame, price: DataFrame[CloseQuote]
) -> DataFrame:
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
      share_price_close=float("nan"),
      share_price_open=float("nan"),
      share_price_high=float("nan"),
      share_price_low=float("nan"),
      share_price_average=float("nan"),
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
  financials: pl.LazyFrame,
  beta_years: int | None = None,
  update: bool = False,
) -> pl.DataFrame:
  start_date = financials.select("date").min().collect().item(0, 0) - relativedelta(
    years=beta_years or 1
  )

  prices: list[pl.DataFrame] = []
  for id in ticker_ids:
    ohlcv_fetcher = partial(Stock(id, currency).ohlcv)

    price = await load_ohlcv(
      f"{id}_{currency}",
      "stock",
      ohlcv_fetcher,
      start_date=start_date,
      cols=["close"],
    )
    prices.append(price)

  price_df = pl.concat(prices, how="diagonal")

  financials = trailing_twelve_months(financials)
  if update:
    levels = (
      financials.select("months").unique().collect().get_column("months").to_list()
    )
    if 3 in levels:
      financials = pl.concat(
        [
          financials.filter(pl.col("months") == 3).tail(2),
          financials.filter(pl.col("months") == 12),
        ]
      )

  wasob = "weighted_average_shares_outstanding_basic"
  if len(ticker_ids) > 1:
    share_cols = [f"{wasob}.{id}" for id in ticker_ids]
    financials = financials.with_columns(
      sum([pl.col(c).alias(wasob) for c in share_cols]).alias(wasob)
    )

  financials = merge_share_price(financials, price_df)
  split_ratios = stock_splits(ticker_ids[0])
  financials = stock_split_adjust(financials, split_ratios)

  schema = load_schema()
  financials = calculate_items(financials, schema)

  market_close = await load_ohlcv(
    "GSPC",
    "index",
    partial(Ticker("^GSPC").ohlcv),
    start_date=start_date,
    cols=["close"],
  )
  riskfree_rate = await load_ohlcv(
    "TNX", "index", partial(Ticker("^TNX").ohlcv), start_date=start_date, cols=["close"]
  )

  price_returns = price_df.with_columns(
    pl.mean([pl.col(id) for id in ticker_ids]).alias("close")
  )
  price_returns = price_df.select(pl.col("close").pct_change().alias("equity_return"))
  market_return = market_close.select(pl.col("close").pct_change())

  financials = await beta(
    financials, price_returns, market_return, riskfree_rate, beta_years
  )

  financials = weighted_average_cost_of_capital(financials)
  financials = f_score(financials)
  financials = m_score(financials)

  if "altmann_z_score" not in financials.collect_schema().names():
    financials = z_score(financials)

  return financials


async def calculate_fundamentals_pandas(
  ticker_ids: list[str],
  currency: str,
  financials: DataFrame,
  beta_years: int | None = None,
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
      await load_ohlcv(
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

  financials = merge_share_price_pandas(financials, price_df)
  split_ratios = stock_splits(id)
  financials = stock_split_adjust_pandas(financials, split_ratios)
  schema = load_schema()
  financials = calculate_items(financials, schema)
  # financials.to_csv(f"{ticker_ids[0]}_financials.csv")

  market_fetcher = partial(Ticker("^GSPC").ohlcv)
  market_close = cast(
    DataFrame[CloseQuote],
    await load_ohlcv(
      "GSPC", "index", market_fetcher, start_date=start_date, cols=["close"]
    ),
  )
  riskfree_fetcher = partial(Ticker("^TNX").ohlcv)
  riskfree_rate = cast(
    DataFrame[CloseQuote],
    await load_ohlcv(
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


def load_fundamentals(
  id: str, currency: str, columns: OrderedSet[str] | None = None, where: str = ""
) -> DataFrame | None:
  index_columns = ["date", "period", "months"]

  db_table = (
    ("financials.db", f"{id}_{currency}"),
    ("fundamentals.db", id),
  )

  fundamentals = [
    select_sqlite(db, table, columns, index_columns, where) for db, table in db_table
  ]
  fundamentals = [i for i in fundamentals if i is not None]

  if not fundamentals:
    return None

  return cast(DataFrame, pd.concat(fundamentals, axis=1))


def load_ratio_items() -> DataFrame:
  query = """SELECT item FROM items 
    WHERE type = 'fundamental'
  """
  df = read_sqlite("taxonomy.db", query)
  if df is None:
    raise ValueError("Ratio items not found")

  return df


async def update_fundamentals(
  company_id: str,
  ticker_ids: list[str],
  currency: str,
  beta_years: int | None = None,
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

    fundamentals.replace([inf, -inf], float("nan"), inplace=True)
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

  fundamentals_.replace([inf, -inf], float("nan"), inplace=True)
  ratio_cols_ = fundamentals_.columns.intersection(ratio_cols)
  ratios = fundamentals_[ratio_cols_]
  financials = fundamentals_.drop(ratio_cols_, axis=1)

  upsert_sqlite(financials, "financials.db", table, {"date": Date})
  upsert_sqlite(ratios, "fundamentals.db", company_id, {"date": Date})
  return cast(DataFrame, pd.concat((fundamentals, fundamentals_), axis=0))
