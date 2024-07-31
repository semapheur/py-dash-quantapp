import asyncio
from functools import partial

from dash import (
  ALL,
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
  Patch,
)
import pandas as pd
from pandera.typing import DataFrame
from skfolio import RatioMeasure, RiskMeasure, PerfMeasure
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

from components.ticker_select import TickerSelectAIO
from lib.fin.quote import load_ohlcv
from lib.morningstar.ticker import Stock

register_page(__name__, path="/portfolio")

layout = html.Main(
  className="h-full grid grid-rows-[auto_1fr]",
  children=[
    html.Form(
      className="grid grid-cols-[1fr_auto]",
      action="",
      children=[
        TickerSelectAIO(
          "portfolio:ticker-select",
          dropdown_props={"multi": True},
        ),
        html.Button("Optimize", id="button:portfolio:optimize", type="button"),
      ],
    ),
    dcc.Graph(id="graph:portfolio:optimize"),
  ],
)


async def load_prices(id_currency: list[str]):
  tasks = []
  for ic in id_currency:
    id, currency = ic.split("_")
    ohlcv_fetcher = partial(Stock(id, currency).ohlcv)
    tasks.append(
      asyncio.create_task(load_ohlcv(ic, "stock", ohlcv_fetcher, cols=["close"]))
    )

  prices: list[DataFrame] = await asyncio.gather(*tasks)
  prices = [p for p in prices if p is not None]
  return pd.concat(prices, axis=1)


@callback(
  Output("graph:portfolio:optimize", "figure"),
  Input("button:portfolio:optimize", "n_clicks"),
  State(TickerSelectAIO.aio_id("portfolio:ticker-select"), "value"),
  background=True,
)
def update_graph(n_clicks: int, id_currency: list[str]):
  if not (n_clicks and id_currency):
    return no_update

  prices = asyncio.run(load_prices(id_currency))
  returns = prices_to_returns(prices)

  model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=10,
    portfolio_params=dict(name="Variance"),
  )
  model.fit(returns)
  population = model.predict(returns)

  figure = population.plot_measures(
    x=RiskMeasure.ANNUALIZED_VARIANCE,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
  )

  return figure
