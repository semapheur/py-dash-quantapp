import asyncio
from dataclasses import dataclass
from datetime import date as Date, datetime as dt, timezone as tz
import re
from typing import cast, Optional
import time

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
import polars as pl

from lib.db.lite import read_sqlite
from lib.yahoo.models import (
  QuoteInterval,
  QuotePeriod,
  Item,
  ItemMeta,
  ItemRecord,
  QuoteData,
  PriceHistory,
)
from lib.utils.time import handle_date
from lib.fin.models import Quote
from lib.scrap import fetch_json

BASE_URL = "https://query2.finance.yahoo.com/v8/finance"


@dataclass(slots=True)
class Ticker:
  ticker: str

  async def ohlcv(
    self,
    start_date: dt | Date | None = None,
    end_date: dt | Date | None = None,
    period: QuotePeriod | None = None,
    interval: QuoteInterval = "1d",
    adjusted_close=False,
  ) -> DataFrame[Quote]:
    async def parse_json(
      start_stamp: int, end_stamp: int, interval: QuoteInterval
    ) -> QuoteData:
      params = {
        "formatted": "true",
        "lang": "en-US",
        "region": "US",
        "includeAdjustedClose": "true",
        "interval": interval,
        "period1": str(start_stamp),
        "period2": str(end_stamp),
        "events": "div|split",
        "userYfid": "true",
      }
      url = f"{BASE_URL}/chart/{self.ticker}"
      data: PriceHistory = await fetch_json(url, params=params)

      return data["chart"]["result"][0]

    if (start_date is not None) and (period is None):
      start_date = handle_date(start_date)
      start_stamp = int(cast(dt, start_date).replace(tzinfo=tz.utc).timestamp())

    elif period == "max":
      start_stamp = int(dt.today().timestamp())
      start_stamp -= 3600 * 24

    else:
      start_stamp = int(dt(1980, 1, 1).replace(tzinfo=tz.utc).timestamp())

    if end_date is not None:
      end_date = handle_date(end_date)
      end_stamp = int(cast(dt, end_date).replace(tzinfo=tz.utc).timestamp())
    else:
      end_stamp = int(dt.now().replace(tzinfo=tz.utc).timestamp())
      end_stamp += 3600 * 24

    parse = await parse_json(start_stamp, end_stamp, interval)

    if period == "max":
      start_stamp = parse["meta"]["firstTradeDate"]
      parse = await parse_json(start_stamp, end_stamp, interval)

    timestamps = parse["timestamp"]
    timezone = parse["meta"]["exchangeTimezoneName"]

    date = pl.from_epoch(timestamps, time_unit="s")
    if interval in {"1d", "5d", "1wk", "1mo", "3mo"}:
      date = date.dt.date()
    else:
      date = date.dt.replace_time_zone(timezone).dt.convert_time_zone("UTC")

    ohlcv = parse["indicators"]["quote"][0]
    close = (
      parse["indicators"]["adjclose"][0]["adjclose"]
      if adjusted_close
      else ohlcv["close"]
    )

    data = {
      "date": date,
      "open": ohlcv["open"],
      "high": ohlcv["high"],
      "low": ohlcv["low"],
      "close": close,
      "volume": ohlcv["volume"],
    }

    df = pl.DataFrame(data)
    zero_cols = [
      c
      for c in df.columns
      if c != "date"
      if (df[c].fill_null(0).sum() == 0) or (df[c].null_count() == df.height)
    ]

    if zero_cols:
      df = df.drop(zero_cols)

    return cast(DataFrame[Quote], df)

  async def financials(self) -> pd.DataFrame:
    async def parse(period: str):
      end_stamp = int(dt.now().timestamp()) + 3600 * 24

      params = {
        "lang": "en-US",
        "region": "US",
        "symbol": self.ticker,
        "padTimeSeries": "true",
        "type": ",".join([period + i for i in cast(DataFrame, items)["yahoo"]]),
        "merge": "false",
        "period1": "493590046",
        "period2": str(end_stamp),
        "corsDomain": "finance.yahoo.com",
      }
      url = (
        "https://query2.finance.yahoo.com/ws/"
        f"fundamentals-timeseries/v1/finance/timeseries/{self.ticker}"
      )
      parse = await fetch_json(url, params=params)

      dfs: list[pd.DataFrame] = []
      pattern = r"^(annual|quarterly)"
      financials = cast(Item, parse["timeseries"]["result"])
      for i in financials:
        item = cast(ItemMeta, i["meta"])["type"][0]

        if item not in i:
          continue

        scrap: dict[dt, int] = {}
        records: list[ItemRecord] = list(filter(None, cast(list[ItemRecord], i[item])))
        for r in records:
          date = dt.strptime(r["asOfDate"], "%Y-%m-%d")
          scrap[date] = r["reportedValue"]["raw"]

        df = pd.DataFrame.from_dict(
          scrap, orient="index", columns=[re.sub(pattern, "", item)]
        )

        dfs.append(df)

      df = pd.concat(dfs, axis=1)
      df["period"] = period[0]

      return df

    query = """
      SELECT json_each.value AS yahoo, item FROM items 
      JOIN json_each(yahoo) ON 1=1
      WHERE yahoo IS NOT NULL
    """
    items = read_sqlite("taxonomy.db", query)
    if items is None:
      raise ValueError("Yahoo financial items not seeded!")

    dfs = []
    for p in ("annual", "quarterly"):
      dfs.append(parse(p))

    df = pd.concat(dfs)

    col_map = {k: v for k, v in zip(items["yahoo"], items["item"])}
    df.rename(columns=col_map, inplace=True)
    df.set_index("period", append=True, inplace=True)
    df.index.names = ["date", "period"]

    return df

  async def option_chains(self) -> pd.DataFrame:
    def parse_option_chain(parse: dict, stamp):
      def get_entry(opt: dict, key: str):
        if key in opt:
          if isinstance(opt[key], dict):
            entry = opt[key].get("raw")
          elif isinstance(opt[key], bool):
            entry = opt[key]
        else:
          entry = np.nan

        return entry

      options = parse["optionChain"]["result"][0]["options"][0]

      cols = (
        "strike",
        "impliedVolatility",
        "openInterest",
        "lastPrice",
        "ask",
        "bid",
        "inTheMoney",
      )
      cells = np.zeros((len(cols), (len(options["calls"]) + len(options["puts"]))))

      # Calls
      for i, opt in enumerate(options["calls"]):
        for j, c in enumerate(cols):
          cells[j, i] = get_entry(opt, c)

      # Puts
      for i, opt in enumerate(options["puts"]):
        for j, c in enumerate(cols):
          cells[j, i + len(options["calls"])] = get_entry(opt, c)

      data = {k: v for k, v in zip(cols, cells)}
      data["optionType"] = np.array(
        ["call"] * len(options["calls"]) + ["put"] * len(options["puts"])
      )

      # Parse to data frame
      df = pd.DataFrame.from_records(data)

      # Add expiry date
      date = dt.fromtimestamp(stamp, tz=tz.utc)
      df["expiry"] = date

      return df

    params = {
      "formatted": "true",
      "lang": "en-US",
      "region": "US",
      "corsDomain": "finance.yahoo.com",
    }

    url = f"{BASE_URL}/options/{self.ticker}"
    parse = await fetch_json(url, params=params)

    # Maturity dates
    stamps = parse["optionChain"]["result"][0]["expirationDates"]

    # Parse first option chain to dataframe
    dfs = []
    dfs.append(parse_option_chain(parse, stamps[0]))

    # Parse remaining option chains
    for i in range(1, len(stamps)):
      time.sleep(1)

      params["date"] = stamps[i]

      parse = await fetch_json(url, params=params)

      dfs.append(parse_option_chain(parse, stamps[i]))

    # Concatenate
    df = pd.concat(dfs)

    return df


async def batch_ohlcv(
  tickers: list[str],
  start_date: Optional[dt] = None,
  end_date: Optional[dt] = None,
  period: Optional[QuotePeriod] = None,
  interval: QuoteInterval = "1d",
):
  tasks: list[asyncio.Task] = []

  for t in tickers:
    ticker = Ticker(t)

    tasks.append(
      asyncio.create_task(ticker.ohlcv(start_date, end_date, period, interval, True))
    )

  dfs: list[pd.DataFrame] = await asyncio.gather(*tasks)
  return pd.concat(dfs, axis=1)


async def exchange_rate(
  ticker: str, start_date: dt, end_date: dt, interval: QuoteInterval
):
  if not ticker.endswith("=X"):
    ticker += "=X"

  quotes = await Ticker(ticker).ohlcv(start_date, end_date, None, interval)
  return quotes["close"].mean()
