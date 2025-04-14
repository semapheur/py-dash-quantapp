import asyncio
from datetime import date, datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from itertools import combinations
from typing import cast, TypedDict

import aiometer
from dash import State, callback, dcc, html, no_update, register_page, Output, Input
import dash_ag_grid as dag
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
import plotly.express as px
import statsmodels.api as sm

# from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint  # adfuller

from lib.db.lite import read_sqlite
from lib.fin.quote import load_ohlcv, load_ohlcv_batch
from lib.morningstar.ticker import Stock
from lib.statistics.hurst import hurst_variance
from lib.styles import BUTTON_STYLE

from components.input import InputAIO


class PairData(TypedDict):
  security_id_1: str
  security_id_2: str
  correlation: float
  intercept: float
  slope: float
  test_statistic: float
  p_value: float
  hurst: float


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

number_fields = {
  "correlation": {},
  "slope": {},
  "intercept": {},
  "test_statistic": {"name": "Test Statistic"},
  "p_value": {"name": "P-Value", "format": ".2e"},
  "hurst": {},
}

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
] + [
  {
    "field": k,
    "headerName": v.get("name", k.capitalize()),
    "cellDataType": "number",
    "valueFormatter": {
      "function": f"d3.format('{v.get('format', '.3f')}')(params.value)"
    },
  }
  for k, v in number_fields.items()
]

form = html.Form(
  className="col-span-2 grid grid-cols-[auto_1fr_auto_auto] gap-2 p-2",
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
  className="size-full grid grid-cols-2",
  children=[
    form,
    dag.AgGrid(
      id="table:pair-cointegration",
      columnDefs=column_defs,
      columnSize="autoSize",
      dashGridOptions={"rowSelection": "single"},
      style={"height": "100%"},
    ),
    dcc.Graph(id="graph:pair:price"),
  ],
)


def compute_pairs_data(
  prices: pd.DataFrame, corr_threshold: float, tickers: pd.DataFrame
) -> pd.DataFrame:
  corr_df = prices.corr()

  col_pairs = list(combinations(prices.columns, 2))
  results: list[PairData] = []

  for col1, col2 in col_pairs:
    corr = cast(float, corr_df.at[col1, col2])
    if np.isnan(corr) or corr < corr_threshold:
      continue

    p_1, p_2 = prices[col1], prices[col2]
    mask = p_1.notna() & p_2.notna()
    x, y = p_1[mask].to_numpy(), p_2[mask].to_numpy()
    score, pvalue, _ = coint(x, y)

    y_const = sm.add_constant(y)
    model = sm.OLS(x, y_const).fit()
    intercept, slope = model.params

    spread = x - (slope * y + intercept)
    hurst = hurst_variance(spread)

    results.append(
      PairData(
        security_id_1=col1,
        security_id_2=col2,
        correlation=corr,
        intercept=intercept,
        slope=slope,
        test_statistic=score,
        p_value=pvalue,
        hurst=hurst,
      )
    )

  pairs = pd.DataFrame.from_records(results)
  pairs["stock_1"] = pairs["security_id_1"].apply(lambda x: tickers.at[x, "ticker"])
  pairs["stock_2"] = pairs["security_id_2"].apply(lambda x: tickers.at[x, "ticker"])
  return pairs


def compute_correlation(row: pd.Series, df: pd.DataFrame):
  x = df[row["stock_1"]]
  y = df[row["stock_2"]]
  mask = x.notna() & y.notna()
  return np.corrcoef(x[mask], y[mask])[0, 1]


async def load_prices(
  ids: list[str],
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
    tasks, max_at_once=3, max_per_second=6
  )
  prices = [
    p.rename(columns={"close": i}) for i, p in zip(ids, prices) if p is not None
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
def update_table(n_clicks: int, exchange_currency: str, start: str, threshold: float):
  if not (n_clicks and exchange_currency and start and threshold):
    return no_update

  exchange, currency = exchange_currency.split("|")

  start_date = dt.strptime(start, "%Y-%m-%d").date()
  query = "SELECT security_id, ticker FROM stock WHERE mic = :exchange"
  tickers = read_sqlite(
    "ticker.db", query, params={"exchange": exchange}, index_col="security_id"
  )

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  prices = loop.run_until_complete(
    load_prices(tickers.index.to_list(), currency, start_date)
  )
  prices.sort_index(inplace=True)

  pairs = compute_pairs_data(prices, threshold, tickers)

  return pairs.to_dict("records")


@callback(
  Output("graph:pair:price", "figure"),
  Input("table:pair-cointegration", "selectedRows"),
  State("dropdown:pair:exchange", "value"),
  State("datepicker:pair:start", "date"),
  prevent_initial_call=True,
)
def update_graph(row: list[dict], exchange_currency: str, start_date_text: str):
  if not row:
    return no_update

  currency = exchange_currency.split("|")[1]
  tables = [
    f"{row[0]['security_id_1']}_{currency}",
    f"{row[0]['security_id_2']}_{currency}",
  ]

  start_date = dt.strptime(start_date_text, "%Y-%m-%d").date()
  prices = load_ohlcv_batch(tables, "stock", start_date=start_date, cols=["close"])

  if prices is None:
    return no_update

  slope = row[0]["slope"]
  intercept = row[0]["intercept"]
  prices["residual"] = prices[tables[0]] - (slope * prices[tables[1]] + intercept)

  return px.line(prices, x=prices.index, y="residual")
