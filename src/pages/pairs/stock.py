import asyncio
from datetime import date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from itertools import combinations

import aiometer
from dash import State, callback, dcc, html, no_update, register_page, Output, Input
import dash_ag_grid as dag
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
# import statsmodels.api as sm

# from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint  # adfuller

from lib.db.lite import read_sqlite
from lib.fin.quote import load_ohlcv
from lib.morningstar.ticker import Stock
from lib.styles import BUTTON_STYLE

from components.input import InputAIO

register_page(__name__, path="/pair/stock")

query = """
  SELECT
    CASE
      WHEN e.market_name IS NOT NULL AND e.country IS NOT NULL 
        THEN e.market_name || " (" || s.mic || ":" || e.country || ")" 
      ELSE s.mic
    END AS label,
    s.mic || "|" || e.currency AS value 
  FROM (
    SELECT DISTINCT mic FROM stock
  ) s
  LEFT JOIN exchange e ON s.mic = e.mic
  ORDER BY s.mic
"""
exchanges = read_sqlite("ticker.db", query)

column_defs = [
  {
    "field": "stock_1",
    "headerName": "Stock 1",
    "cellDataType": "text",
  },
  {
    "field": "stock_2",
    "headerName": "Stock 2",
    "cellDataType": "text",
  },
  {
    "field": "correlation",
    "headerName": "Correlation",
    "cellDataType": "number",
  },
  {
    "field": "test_stat",
    "headerName": "Test Statistic",
    "cellDataType": "number",
  },
  {
    "field": "p_value",
    "headerName": "P-Value",
    "cellDataType": "number",
  },
]

form = html.Form(
  className="grid grid-cols-[auto_1fr_auto_auto] gap-2 p-2",
  children=[
    html.Button(
      "Search",
      id="button:pair:search",
      className=BUTTON_STYLE,
      type="button",
    ),
    dcc.Dropdown(
      id="dropdown:pair:exchange",
      placeholder="Exchange",
      options=exchanges.to_dict("records"),
    ),
    dcc.DatePickerSingle(
      id="datepicker:pair:start",
      display_format="YYYY-MM-DD",
      date=date.today() - relativedelta(years=3),
      min_date_allowed=date(2000, 1, 1),
      max_date_allowed=date.today() - relativedelta(years=1),
      clearable=True,
      first_day_of_week=1,
    ),
    InputAIO(
      "pair:correlation-threshold",
      "100%",
      input_props={"type": "number", "value": 0.95, "min": -1, "max": 1},
    ),
  ],
)

layout = html.Main(
  className="size-full grid grid-rows-[auto_1fr]",
  children=[
    form,
    dag.AgGrid(
      id="table:pair-cointegration",
      columnDefs=column_defs,
      columnSize="autoSize",
      style={"height": "100%"},
    ),
  ],
)


def pairs_data(prices: pd.DataFrame, corr_threshold: float) -> pd.DataFrame:
  corr_df = prices.corr()

  col_pairs = list(combinations(prices.columns, 2))
  results: list[tuple[str, str, float, float, float]] = []

  for col1, col2 in col_pairs:
    corr = corr_df.loc[col1, col2]
    if np.isnan(corr) or corr < corr_threshold:
      continue

    x, y = prices[col1], prices[col2]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    score, pvalue, _ = coint(x, y)

    results.append((col1, col2, corr, score, pvalue))

  pairs = pd.DataFrame(
    results, columns=["stock_1", "stock_2", "correlation", "test_stat", "p_value"]
  )
  return pairs


def compute_correlation(row: pd.Series, df: pd.DataFrame):
  x = df[row["stock_1"]]
  y = df[row["stock_2"]]
  mask = x.notna() & y.notna()
  return np.corrcoef(x[mask], y[mask])[0, 1]


async def load_prices(
  ids: list[str],
  tickers: list[str],
  currency: str,
  start_date: date,
):
  tasks = [
    partial(
      load_ohlcv,
      f"{i}_{currency}",
      "stock",
      partial(Stock(i, currency).ohlcv),
      start_date=start_date,
      cols=["close"],
    )
    for i in ids
  ]

  prices: list[DataFrame] = await aiometer.run_all(
    tasks, max_per_second=5
  )  # await asyncio.gather(*tasks)
  prices = [
    p.rename(columns={"close": ticker})
    for ticker, p in zip(tickers, prices)
    if p is not None
  ]
  return pd.concat(prices, axis=1)


@callback(
  Output("table:pair-cointegration", "rowData"),
  Input("button:pair:search", "n_clicks"),
  State("dropdown:pair:exchange", "value"),
  State("datepicker:pair:start", "date"),
  State(InputAIO.aio_id("pair:correlation-threshold"), "value"),
  prevent_initial_call=True,
  # background=True,
)
def search(n_clicks: int, exchange_currency: str, start: str, threshold: float):
  if not (n_clicks and exchange_currency and start and threshold):
    return no_update

  exchange, currency = exchange_currency.split("|")

  start_date = dt.strptime(start, "%Y-%m-%d").date()
  query = "SELECT security_id, ticker FROM stock WHERE mic = :exchange"
  tickers = read_sqlite("ticker.db", query, {"exchange": exchange})

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  prices = loop.run_until_complete(
    load_prices(tickers["security_id"], tickers["ticker"], currency, start_date)
  )
  prices.sort_index(inplace=True)

  pairs = pairs_data(prices, threshold)

  return pairs.to_dict("records")
