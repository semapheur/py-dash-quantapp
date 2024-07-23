import re
from typing import Literal

from dash import dcc, Patch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class QuoteGraphAIO(dcc.Graph):
  @staticmethod
  def aio_id(id: str):
    return {"component": "QuoteGraphAIO", "aio_id": id}

  def __init__(self, id: str, graph_props: dict | None = None):
    graph_props = graph_props.copy() if graph_props else {}

    super().__init__(id=self.__class__.aio_id(id), **graph_props)


def quote_rangeselector(
  selectors: tuple[str, ...],
) -> dict[str, list[dict[str, int | str]]]:
  pattern = r"^(?P<count>\d+)(?P<step>\w)$"
  step_code = dict(d="day", m="month", w="week", y="year")

  buttons: list[dict[str, int | str]] = []
  for s in selectors:
    m = re.match(pattern, s)
    if m is not None:
      details = m.groupdict()
      buttons.append(
        dict(
          count=int(details["count"]),
          label=s,
          step=step_code[details["step"].lower()],
          stepmode="backward",
        )
      )
    elif s.lower() == "ytd":
      buttons.append(dict(count=1, label=s, step="year", stepmode="todate"))
    elif s.lower() == "all":
      buttons.append(dict(label=s, step="all"))
  return dict(buttons=buttons)


def quote_candlestick(data: dict):
  if not {"open", "high", "low", "close"}.issubset(data.keys()):
    return {}

  return go.Candlestick(
    x=data["date"],
    open=data["open"],
    high=data["high"],
    low=data["low"],
    close=data["close"],
    showlegend=False,
  )


def quote_line(data: dict) -> go.Scatter:
  return go.Scatter(x=data["date"], y=data["close"], mode="lines", showlegend=False)


def quote_graph(
  data: dict[str, list[str | float | int]],
  plot: Literal["line", "candlestick"] = "line",
  rangeselector: tuple[str, ...] | None = None,
  rangeslider=False,
) -> go.Figure:
  xaxis = dict(type="date", rangeslider=dict(visible=False))
  if rangeselector:
    xaxis["rangeselector"] = quote_rangeselector(rangeselector)

  if rangeslider:
    xaxis["rangeslider"] = dict(visible=True)

  trace = []
  if plot == "line":
    trace.append(quote_line(data))

  elif plot == "candlestick":
    trace.append(quote_candlestick(data))

  fig = go.Figure(data=trace, layout_xaxis=xaxis)
  return fig


def quote_volume_graph(
  data: dict[str, list[str | float | int]],
  plot: Literal["line", "candlestick"] = "line",
  rangeselector: tuple[str, ...] | None = None,
  rangeslider=False,
) -> go.Figure:
  if "volume" not in data.keys():
    return quote_graph(data, plot, rangeselector, rangeslider)

  fig = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.02,
    shared_xaxes=True,
  )
  if plot == "line":
    fig.add_trace(quote_line(data), row=1, col=1)
  elif plot == "candlestick":
    fig.add_trace(quote_candlestick(data), row=1, col=1)

  fig.add_trace(
    go.Bar(
      x=data["date"],
      y=data["volume"],
      showlegend=False,
      marker_color="orange",
    ),
    row=2,
    col=1,
  )

  xaxis = dict(type="date", rangeslider=dict(visible=False))
  if rangeselector:
    xaxis["rangeselector"] = quote_rangeselector(rangeselector)

  if rangeslider:
    xaxis["rangeslider"] = dict(visible=True)

  fig.update_layout(xaxis=xaxis)

  return fig


def quote_graph_range(
  figure: go.Figure,
  start_date: str,
  end_date: str,
):
  figure_patched = Patch()

  for key in figure["layout"]:
    if key.startswith("xaxis"):
      figure_patched["layout"][key]["range"] = [start_date, end_date]

    elif key.startswith("yaxis"):
      axis_label = key.replace("yaxis", "")

      y_min = []
      y_max = []
      for trace in figure["data"]:
        if trace["yaxis"] != axis_label:
          continue

        data = pd.Series(data=trace["y"], index=trace["x"])
        data = data.between_time(start_date, end_date)

        y_min.append(data.min())
        y_max.append(data.max())

      figure_patched["layout"][key]["range"] = [min(y_min), max(y_max)]
      figure_patched["layout"][key]["autorange"] = False

      figure_patched["layout"][key]["autorange"] = False

  return figure_patched


def quote_graph_relayout(
  relayout: dict,
  figure: go.Figure,
):
  def get_axes(relayout: dict) -> set[str]:
    return {x.split(".")[0] for x in relayout.keys()}

  if all(x in relayout.keys() for x in ["xaxis.range[0]", "xaxis.range[1]"]):
    figure_patched = quote_graph_range(
      figure, relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
    )
  elif "xaxis.autorange" in relayout.keys():
    figure_patched = Patch()

    for axis in get_axes(relayout):
      figure_patched["layout"][axis]["autorange"] = True

  return figure_patched
