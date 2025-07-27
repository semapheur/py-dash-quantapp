from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Annotated

import pandas as pd
from pandera.typing import DataFrame

from lib.fin.calculation_pandas import calculate_items, trailing_twelve_months
from lib.fin.metrics_pandas import (
  f_score,
  m_score,
  z_score,
  beta,
  weighted_average_cost_of_capital,
)
from lib.fin.models import CloseQuote, SharePrice
from lib.fin.quote import load_ohlcv
from lib.fin.statement import stock_splits
from lib.fin.taxonomy import load_schema
from lib.morningstar.ticker import Stock
from lib.yahoo.ticker import Ticker


def stock_split_adjust(
  financials: pd.DataFrame, ratios: pd.Series | None
) -> pd.DataFrame:
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


def merge_share_price(
  financials: pd.DataFrame, price: Annotated[pd.DataFrame, DataFrame[CloseQuote]]
) -> pd.DataFrame:
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

    price_ = price.loc[start_date:end_date].copy().sort_index()
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

  return pd.concat((financials, price_data), axis=1)


async def calculate_fundamentals(
  ticker_ids: list[str],
  currency: str,
  financials: pd.DataFrame,
  beta_years: int | None = None,
  update: bool = False,
) -> pd.DataFrame:
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

  price_df = pd.concat(prices, axis=1)

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
