import pandas as pd

from datetime import datetime as dt

import requests
import json
import io

import re

# Local
from lib.finlib import finItemRenameDict


class Ticker:
  apiKey = ""

  def __init__(self, ticker):
    self._ticker = ticker

  def financials(self):
    # timeFrame: annual/quarter

    finItems = finItemRenameDict("FMP")

    interval = {"a": "", "q": "period=quarter"}

    sheets = ["income-statement", "balance-sheet-statement", "cash-flow-statement"]

    dfs = []
    for k, v in interval.items():
      dfSheets = []
      for i, sheet in enumerate(sheets):
        url = (
          f"https://financialmodelingprep.com/api/v3/{sheet}/"
          f"{self._ticker}?{v}&apikey={self.apiKey}"
        )

        with requests.Session() as s:
          rc = s.get(url)
          parse = json.loads(rc.text)

        temp = pd.DataFrame.from_records(parse)
        temp.set_index("date", inplace=True)
        temp.drop(["symbol", "acceptedDate", "period", "link"], axis=1, inplace=True)
        temp.sort_index(inplace=True)

        dfSheets.append(temp)

      temp = pd.concat(dfSheets, axis=1)
      temp.insert(0, "period", [k] * len(temp))
      dfs.append(temp)

    df = pd.concat(dfs, axis=0)  # Column wise
    df.index = pd.to_datetime(df.index)
    df.index.names = ["date"]
    df.sort_index(inplace=True)

    # Drop excess columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(
      ["cashAtEndOfPeriod", "cashAtBeginningOfPeriod", "operatingCashFlow"],
      axis=1,
      inplace=True,
    )

    # Rename items
    df.rename(columns=finItems, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # Additional posts
    df["sgaEx"] = df["gaEx"] + df["smEx"]
    df["intCvg"] = df["opInc"] / df["intEx"]
    df["taxRate"] = df["taxEx"] / df["ebt"]
    df["tngEqt"] = df["totEqt"] - df["gwItngAst"]

    return df

  def ohlcv(self, startDate="1980-01-01"):
    # OHLC
    urls = [
      (
        "https://financialmodelingprep.com/api/v3/historical-price-full/"
        f"{self._ticker}?from={startDate}&apikey={self.apiKey}"
      ),
      (
        "https://financialmodelingprep.com/api/v3/"
        f"historical-daily-discounted-cash-flow/{self._ticker}?apikey={self.apiKey}"
      ),
    ]

    # Rating
    # url = ('https://financialmodelingprep.com/api/v3/rating/'
    #        f'{self._ticker}?apikey={self.apiKey}')

    dfs = []

    for i, url in enumerate(urls):
      with requests.Session() as s:
        rc = s.get(url)
        parse = json.loads(rc.text)

      if i == 0:
        temp = pd.DataFrame.from_records(parse["historical"])

      elif i == 1:
        temp = pd.DataFrame.from_records(parse)

      temp["date"] = pd.to_datetime(temp["date"], format="%Y-%m-%d")
      temp.set_index("date", inplace=True)
      temp.sort_index(inplace=True)

      dfs[i] = temp

    df = pd.concat(dfs, axis=1)

    return df


def getTickers():
  apiKey = Ticker.apiKey

  # Complete ticker list
  url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={apiKey}"

  with requests.Session() as s:
    rc = s.get(url)
    parse = json.loads(rc.text)

  dfStock = pd.DataFrame(parse, columns=["ticker"])
  dfStock.drop("price", axis=1, inplace=True)
  dfStock.rename(columns={"symbol": "ticker"}, inplace=True)
  dfStock.sort_values(by=["ticker"], inplace=True)

  # Tickers with available financial statements
  url = (
    "https://financialmodelingprep.com/api/v3/"
    f"financial-statement-symbol-lists?apikey={apiKey}"
  )

  with requests.Session() as s:
    rc = s.get(url)
    parse = json.loads(rc.text)

  dfStockSheets = pd.DataFrame(parse, columns=["ticker"])
  dfStockSheets["fmpFinancialStatement"] = True

  df = dfStock.merge(dfStockSheets, on="ticker", how="outer")
  df["fmpFinancialStatement"].fillna(False, inplace=True)

  # Trim tickers
  pattern = r"(\s|-|_|/|\'|^0+(?=[1-9]\d*)|[inxp]\d{4,5}$)"
  df["tickerTrim"] = df["ticker"].apply(
    lambda x: re.sub(pattern, "", x.lower().split(".")[0])
  )

  # Add security type
  df["type"] == "stock"
  mask = df["name"].str.contains(r"\b(et(c|f|n|p)(s)?)\b", regex=True, flags=re.I)
  df.loc[mask, "type"] = "etf"

  return df
