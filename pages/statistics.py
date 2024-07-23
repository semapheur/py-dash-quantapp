from functools import partial

import numpy as np
import pandas as pd
from dash import (
  callback,
  ctx,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  Patch,
  State,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gennorm, t
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import STL
# import fathon
# from fathon import fathonUtils as fu

from lib.fracdiff import fast_frac_diff as frac_diff
from lib.morningstar.ticker import Stock
from lib.fin.quote import load_ohlcv
from components.ticker_select import TickerSelectAIO
from components.statistical_plots import acf_trace, qqplot_trace

register_page(__name__, path="/statistics", location="header")

main_style = "h-full flex flex-col gap-2 p-2"
form_style = "grid grid-cols-[2fr_1fr_1fr] gap-2 p-2 shadow rounded-md"
overview_style = "h-full grid grid-cols-[1fr_1fr] shadow rounded-md"
dropdown_style = "w-1/3"

layout = html.Main(
  className=main_style,
  children=[
    html.Form(
      className=form_style,
      children=[
        TickerSelectAIO(id="stats"),
        dcc.Input(
          id="input:stats:diff-order",
          type="number",
          className="border rounded pl-2",
          min=0,
          max=10,
          value=0,
        ),
        dcc.Dropdown(
          id="dd:stats:transform", options=[{"label": "Log", "value": "log"}], value=""
        ),
      ],
    ),
    dcc.Tabs(
      id="tabs:stats",
      value="tab-transform",
      className="inset-row",
      content_className=overview_style,
      parent_className="h-full",
      children=[
        dcc.Tab(
          label="Transform",
          value="tab-transform",
          className="inset-row",
          children=[
            dcc.Graph(id="grap:stats:price", responsive=True),
            dcc.Graph(id="graph:stats:transform", responsive=True),
          ],
        ),
        dcc.Tab(
          label="Distribution",
          value="tab-distribution",
          className="inset-row",
          children=[
            dcc.Graph(id="graph:stats:distribution"),
            html.Div(
              className="h-full flex flex-col",
              children=[
                dcc.Graph(id="graph:stats:qq"),
                dcc.Dropdown(
                  id="dd:stats:qq-distribution",
                  className="pb-2 px-2 drop-up",
                  options=[
                    {"label": "Normal", "value": "norm"},
                    {"label": "Laplace", "value": "laplace"},
                    {"label": "Generalized Normal", "value": "gennorm"},
                    {"label": "Student-t", "value": "t"},
                  ],
                  placeholder="Distribution",
                  value="norm",
                ),
              ],
            ),
          ],
        ),
        dcc.Tab(
          label="Autocorrelation",
          value="tab-autocorrelation",
          className="inset-row",
          children=[
            dcc.Graph(id="graph:stats:acf", responsive=True),
            dcc.Graph(id="graph:stats:pacf", responsive=True),
          ],
        ),
        dcc.Tab(
          label="Trend",
          value="tab-trend",
          className="inset-row",
          children=[
            html.Div(
              className="flex flex-col",
              children=[
                dcc.Graph(id="stats-graph:trend"),
                html.Form(
                  className="pb-2 pl-2",
                  children=[
                    dcc.Input(
                      id="input:stats:period",
                      className="border rounded pl-2",
                      type="number",
                      placeholder="Period",
                      min=2,
                      max=365,
                      step=1,
                      value=252,
                    ),
                  ],
                ),
              ],
            )
          ],
        ),
        dcc.Tab(
          label="Regimes",
          value="tab-regimes",
          className="inset-row",
          children=[
            html.Div(
              className="flex flex-col",
              children=[
                dcc.Graph(id="graph:stats:regimes", className="h-full"),
                html.Form(
                  className="pb-2 pl-2",
                  children=[
                    dcc.Input(
                      id="input:stats:regimes",
                      className="border rounded pl-2",
                      type="number",
                      placeholder="Regimes",
                      min=1,
                      max=5,
                      step=1,
                      value=2,
                    ),
                  ],
                ),
              ],
            ),
            dcc.Graph(id="graph:stats:regime-distribution"),
          ],
        ),
        dcc.Tab(
          label="Memory",
          value="tab-memory",
          className="inset-row",
          children=[
            html.Div(
              className="flex flex-col",
              children=[
                dcc.Graph(id="graph:stats:memory"),
                html.Form(
                  className="pb-2 pl-2",
                  children=[
                    dcc.Input(
                      id="input:stats:window",
                      className="border rounded pl-2",
                      type="number",
                      placeholder="Regimes",
                      min=3,
                      value=60,
                    ),
                  ],
                ),
              ],
            )
          ],
        ),
      ],
    ),
    dcc.Store(id="store:stats:price"),
    dcc.Store(id="store:stats:transform"),
    dcc.Store(id="store:stats:model"),
  ],
)


@callback(
  Output("store:stats:price", "data"),
  Input(TickerSelectAIO.aio_id("stats"), "value"),
)
async def update_store(query: str):
  if not query:
    return no_update

  id, currency = query.split("|")

  fetcher = partial(Stock(id, currency).ohlcv)
  price = await load_ohlcv(id, "stock", fetcher, cols=["close"])

  price.reset_index(inplace=True)
  return price.to_dict("list")


@callback(
  Output("store:stats:transform", "data"),
  Input("store:stats:price", "data"),
  Input("input:stats:diff-order", "value"),
  Input("dd:stats:transform", "value"),
)
def update_transform_store(data, diff_order, transform):
  if not data:
    return no_update

  price = pd.DataFrame.from_dict(data, orient="columns")

  if transform == "log":
    price["close"] = np.log(price["close"])

  if diff_order:
    if isinstance(diff_order, int):
      price["close"] = np.diff(
        price["close"], n=diff_order, prepend=[np.nan] * diff_order
      )
    elif isinstance(diff_order, float):
      price["close"] = frac_diff(price["close"].to_numpy(), diff_order)

    price.dropna(inplace=True)

  price.dropna(inplace=True)
  price.rename(columns={"close": "transform"}, inplace=True)
  price.reset_index(inplace=True)
  return price.to_dict("list")


@callback(
  Output("store:stats:model", "data"),
  Input("store:stats:transform", "data"),
  Input("input:stats:regimes", "value"),
)
def update_model_store(data, regimes):
  if not (data and regimes):
    return no_update

  msdr = MarkovRegression(
    data["transform"], k_regimes=regimes, trend="c", switching_variance=True
  ).fit()

  return {"model": msdr.smoothed_marginal_probabilities}


@callback(
  Output("graph:stats:price", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:price", "data"),
  Input("store:stats:transform", "data"),
)
def update_graph(tab, price, transform):
  if not price or tab != "tab-transform":
    return no_update

  store_id = ctx.triggered_id

  if store_id == "store:stats:price":
    fig = go.Figure()
    fig = make_subplots(shared_xaxes=True, rows=2, cols=1)
    fig.add_scatter(
      x=price["date"],
      y=price["close"],
      mode="lines",
      showlegend=False,
    )
    fig.add_scatter(
      x=transform["date"],
      y=transform["transform"],
      mode="lines",
      showlegend=False,
    )
  elif store_id == "store:stats:transform":
    fig = Patch()
    fig["data"][1]["x"] = transform["date"]
    fig["data"][1]["y"] = transform["transform"]

  return fig


@callback(
  Output("graph:stats:transform", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
)
def update_transform_graph(tab, data):
  if not data or tab != "tab-transform":
    return no_update

  fig = go.Figure()
  fig.add_scatter(
    x=data["date"],
    y=data["transform"],
    mode="lines",
    showlegend=False,
  )
  fig.update_layout(title="Transformation")
  return fig


@callback(
  Output("graph:stats:distribution", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
)
def update_distribution_graph(tab, data):
  if not data or tab != "tab-distribution":
    return no_update

  x = np.linspace(np.min(data["transform"]), np.max(data["transform"]), 100)
  mean = np.mean(data["transform"])
  std = np.std(data["transform"])
  ggd_params = gennorm.fit(data["transform"], floc=mean, fscale=std)
  t_params = t.fit(data["transform"], floc=mean, fscale=std)

  fig = go.Figure()
  fig.add_histogram(
    x=data["transform"], histnorm="probability density", name="Transform"
  )
  fig.add_scatter(
    x=x,
    y=gennorm.pdf(x, beta=ggd_params[0], loc=ggd_params[1], scale=ggd_params[2]),
    mode="lines",
    name=f"GGD ({ggd_params[0]:.2f})",
  )
  fig.add_scatter(
    x=x,
    y=t.pdf(x, df=t_params[0], loc=t_params[1], scale=t_params[2]),
    mode="lines",
    name=f"S-t ({t_params[0]:.2f})",
  )
  fig.update_layout(
    title="Distribution", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )
  return fig


@callback(
  Output("graph:stats:qq", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
  Input("dd:stats:qq-distribution", "value"),
)
def update_qq_graph(tab, data, dist):
  if not data or not dist or tab != "tab-distribution":
    return no_update

  transform = np.array(data["transform"])

  params = [np.mean(transform), np.std(transform)]
  if dist == "gennorm":
    params = gennorm.fit(transform, floc=params[0], fscale=params[1])
  elif dist == "t":
    params = t.fit(transform, floc=params[0], fscale=params[1])

  trace = qqplot_trace(transform, dist, tuple(params))

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(
    showlegend=False,
    title="Quantile-quantile plot",
    xaxis_title="Theoretical quantiles",
    yaxis_title="Sample quantile",
  )
  return fig


@callback(
  Output("graph:stats:acf", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
)
def update_acf(tab, data):
  if not data or tab != "tab-autocorrelation":
    return no_update

  transform = np.array(data["transform"])
  trace = acf_trace(transform, False)

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(showlegend=False, title="Autocorrelation")
  # fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor="#000000")

  return fig


@callback(
  Output("graph:stats:pacf", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
)
def update_pacf_graph(tab, data):
  if not data or tab != "tab-autocorrelation":
    return no_update

  transform = np.array(data["transform"])
  trace = acf_trace(transform, True)

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(
    showlegend=False,
    title="Partial autocorrelation",
  )
  # fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor="#000000")

  return fig


@callback(
  Output("graph:stats:trend", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
  Input("input:stats:period", "value"),
)
def update_trend_graph(tab, data, period):
  if not data or tab != "tab-trend" or period < 2:
    return no_update

  s = pd.Series(data["transform"], index=data["date"])
  stl = STL(s, period=period).fit()

  fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
  fig.add_scatter(
    x=stl.seasonal.index,
    y=stl.seasonal.values,
    mode="lines",
    name="Season",
    row=1,
    col=1,
  )
  fig.add_scatter(
    x=stl.trend.index, y=stl.trend.values, mode="lines", name="Trend", row=2, col=1
  )
  fig.add_scatter(
    x=stl.resid.index, y=stl.resid.values, mode="lines", name="Residual", row=3, col=1
  )
  return fig


@callback(
  Output("graph:stats:regimes", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:model", "data"),
  State("store:stats:transform", "data"),
  State("store:stats:price", "data"),
)
def update_regimes_graph(tab, model, transform, price):
  if not model or tab != "tab-regimes":
    return no_update

  model = np.array(model["model"])
  regimes = model.shape[-1]

  state = pd.Series(np.argmax(model, axis=1), index=transform["date"])

  price = pd.Series(price["close"], index=price["date"])
  price = price.loc[state.index]

  fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
  fig.add_scatter(
    x=price.index, y=price.values, mode="lines", name="Close", row=2, col=1
  )

  for r in range(regimes):
    fig.add_scatter(
      x=transform["date"], y=model[:, r], mode="lines", name=f"Regime {r}", row=1, col=1
    )

    temp = price.loc[state == r]

    fig.add_scattergl(
      x=temp.index,
      y=temp.values,
      mode="markers",
      marker_size=3,
      name=f"Regime {r}",
      row=2,
      col=1,
    )

  fig.update_layout(
    title="Regime probabilities",
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
    ),
  )

  return fig


@callback(
  Output("graph:stats:regime-distribution", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:model", "data"),
  State("store:stats:transform", "data"),
)
def update_regime_distribution_graph(tab, model, transform):
  if not model or tab != "tab-regimes":
    return no_update

  model = np.array(model["model"])
  regimes = model.shape[-1]

  state = pd.Series(np.argmax(model, axis=1), index=transform["date"])
  transform = pd.Series(transform["transform"], index=transform["date"])

  fig = go.Figure()

  for r in range(regimes):
    temp = transform.loc[state == r]

    fig.add_histogram(
      x=temp.values,
      histnorm="probability density",
      name=f"Regime {r}",
    )

  fig.update_layout(
    title="Regime distributions",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
  )

  return fig


@callback(
  Output("graph:stats:memory", "figure"),
  Input("tabs:stats", "value"),
  Input("store:stats:transform", "data"),
  Input("input:stats:window", "value"),
)
def update_memory_graph(tab, data, window):
  if not (data and window) or tab != "tab-memory" or window < 3:
    return no_update

  # ht = fathon.HT(fu.toAggregated(data['transform']))
  # hurst = ht.computeHt(window, mfdfaPolOrd=1, polOrd=1)[0]

  # fig = go.Figure()
  # fig.add_scatter(x=data['date'][-len(hurst) :], y=hurst, mode='lines')

  # return fig
