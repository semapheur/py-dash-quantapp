# Data analysis
import numpy as np
import pandas as pd

# Date
from datetime import datetime as dt
# from datetime import timezone as tz
# from dateutil.relativedelta import relativedelta

# API
import simfin as sf
from simfin.names import *

# Web scrapping
import requests
import bs4 as bs
import json

# Utils
from pathlib import Path
# import re
# from tqdm import tqdm

# Local
from lib.finlib import finItemRenameDict


class Ticker:
  apiKey = ""

  def __init__(self, ticker):
    self._ticker = ticker

  def financial(self, startYear="2000"):
    finItems = finItemRenameDict("SimFin")

    # define the periods that we want to retrieve
    periods = ["q1", "q2", "q3", "q4"]
    endYear = dt.now().year  # 2020

    urlSheet = "https://simfin.com/api/v2/companies/statements"

    # Parameters
    paramSheet = {
      "statement": "",
      "ticker": self._ticker,
      "period": "",
      "fyear": 0,
      "api-key": self.apiKey,
    }

    # Financial statements
    statements = ["pl", "bs", "cf", "derived"]

    dfs = [None] * len(statements)
    for i, statement in enumerate(statements):
      # Set sheet
      paramSheet["statement"] = statement

      cols = None
      output = []

      for year in range(startYear, endYear + 1):
        # Set year in parameters
        paramSheet["fyear"] = year
        for period in periods:
          # Set period in parameters
          paramSheet["period"] = period

          with requests.Session() as s:
            rc = s.get(urlSheet, params=paramSheet)
            parse = json.loads(rc.text)

          # Check if data is found
          if parse[0]["data"]:
            # Parse columns
            if cols is None:
              cols = parse[0]["columns"]

            # Parse data
            output += parse[0]["data"]

      temp = pd.DataFrame(output, columns=cols)
      if statement == "pl" or statement == "cf":
        for c in temp.columns[10:]:
          temp[c + "_"] = temp[c].rolling(4, min_periods=4).sum()

      elif statement == "bs":
        temp.rename(columns={"Minority Interest": "accruedMinInterest"}, inplace=True)

      elif statement == "derived":
        ttmCols = [
          "EBITDA",
          "Free Cash Flow",
          "Earnings Per Share, Basic",
          "Earnings Per Share, Diluted",
          "Sales Per Share",
          "Free Cash Flow Per Share",
          "Dividends Per Share",
        ]

        for c in ttmCols:
          temp[c + "_"] = temp[c].rolling(4, min_periods=4).sum()

      dfs[i] = temp

    df = pd.concat(dfs, axis=1)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Delete excess columns
    delCol = [
      "SimFinId",
      "Ticker",
      "Fiscal Period",
      "Fiscal Year",
      "TTM",
      "Value Check",
      "Net Income/Starting Line",
      "Net Income from Discontinued Operations",
    ]
    df.drop(delCol, axis=1, inplace=True)

    # Rename columns
    rnm1 = {"Interest Expense": "Gross Interest Expense"}
    df.rename(columns=rnm1, inplace=True)

    ttmPosts = {f"{k}_": v[:-1] for k, v in finItems.items()}

    for d in [finItems, ttmPosts]:
      df.rename(columns=d, inplace=True)

    # Set index and sort
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.set_index("date", inplace=True)

    # Shares outstanding
    urlShare = "https://simfin.com/api/v2/companies/shares"
    paramShare = {
      "type": "",
      "ticker": self._ticker,
      "period": "",
      "fyear": 0,
      "api-key": self.apiKey,
    }

    typ = ["wa-basic", "wa-diluted"]
    startYear = df.index.min().year
    dfs = [None] * len(typ)
    for i, t in enumerate(typ):
      # Set sheet
      paramShare["type"] = t

      cols = None
      output = []

      for year in range(startYear, endYear + 1):
        # Set year in parameters
        paramShare["fyear"] = year
        for period in periods:
          # Set period in parameters
          paramShare["period"] = period

          with requests.Session() as s:
            rc = s.get(urlShare, params=paramShare)
            parse = json.loads(rc.text)

          # Check if data is found
          if parse[0]["data"]:
            # Parse columns
            if cols is None:
              cols = parse[0]["columns"]

            # Parse data
            output += parse[0]["data"]

      temp = pd.DataFrame(output, columns=cols)
      sfx = t.split("-")[-1].capitalize()
      temp.rename(columns={"Value": "shares" + sfx}, inplace=True)

      dfs[i] = temp

    dfShare = pd.concat(dfs, axis=1)

    # Remove duplicate columns
    dfShare = dfShare.loc[:, ~dfShare.columns.duplicated()]

    # Delete columns
    delCols = ["SimFinId", "Ticker", "Fiscal Period", "Fiscal Year", "TTM"]
    dfShare.drop(delCols, axis=1, inplace=True)

    # Rename column
    dfShare.rename(columns={"Report Date": "date"}, inplace=True)

    # Set index and sort
    dfShare["date"] = pd.to_datetime(dfShare["date"], format="%Y-%m-%d")
    dfShare.set_index("date", inplace=True)

    # Merge shares outstanding
    df = pd.concat(df, dfShare)

    # Additional items
    df["intCvg"] = df["opInc"] / df["intEx"]
    df["taxRate"] = df["taxEx"] / df["ebt"]
    df["wrkCap"] = (
      df["cce"] + df["stInv"] + 0.75 * df["rcv"] + 0.5 * df["invnt"]
    ).rolling(2, min_periods=0).mean() - df["totCrtLbt"].rolling(
      2, min_periods=0
    ).mean()

    # Capital expenditure
    # df['capEx'] = df['ppe'].diff() + df['da']

    # Tangible common equity
    df["tgbEqt"] = df["totEqt"] - df["prfEqt"] - df["gw"] - df["itgbAst"]

    # Drop nan columns
    posts = [
      "sgaEx",
      "rdEp",
      "cce",
      "stInv",
      "ivty",
      "rcv",
      "totCrtAst",
      "ltInv",
      "gw",
      "itgbAst",
      "stDbt",
      "ltCapLeas",
      "prfEqt",
      "dvd",
      "wrkCap",
    ]
    for p in posts:
      df[p] = df[p].fillna(0)
    df.dropna(axis=1, how="all", inplace=True)

    return df


def getTickers():
  # Parameters
  params = {"api-key": Ticker.apiKey}

  url = "https://simfin.com/api/v2/companies/list"

  with requests.Session() as s:
    rc = s.post(url, params=params)
    parse = json.loads(rc.text)

  cols = parse[0]["columns"]
  output = parse[0]["data"]

  df = pd.DataFrame(output, columns=cols)
  df.rename(columns={"SimFinId": "simfinId", "Ticker": "ticker"}, inplace=True)

  return df


def bulkFinancials(market="us"):
  # period: annual/quarterly/ttm
  sf.set_api_key(Ticker.apiKey)

  pth = Path.cwd() / "data"
  sf.set_data_dir(pth)

  dfs = []
  dropCols = [
    "SimFinId",
    "Fiscal Year",
    "Fiscal Period",
    "Publish Date",
    "Restated Date",
  ]
  for sheet in ["income", "balance", "cashflow"]:
    dfPeriod = []
    for p in ["annual", "quarterly"]:
      temp = sf.load(
        dataset=sheet,
        variant=p,
        market=market,
        index=[TICKER, REPORT_DATE],
        parse_dates=[REPORT_DATE],
      )  # refresh_days=30

      temp.drop(dropCols, axis=1, inplace=True)
      temp["period"] = p[0]
      temp.set_index("period", append=True, inplace=True)
      temp.index.names = ["ticker", "date", "period"]

      dfPeriod.append(temp)

    dfSheet = pd.concat(dfPeriod, axis=0)

    if sheet == "cashflow":
      dfSheet.drop("Net Income/Starting Line", axis=1, inplace=True)

    dfs.append(dfSheet)

  df = dfs.pop(0)
  for i in dfs:
    df = df.combine_first(i)

    diffCols = i.columns.difference(df.columns)
    df = df.join(i[diffCols], how="outer")

  colRnm = finItemRenameDict("SimFin")
  df.rename(columns=colRnm, inplace=True)

  return df
