import asyncio
from functools import partial
import re
from typing import Literal

from dash import (
  callback,
  clientside_callback,
  ClientsideFunction,
  dcc,
  no_update,
  MATCH,
  Input,
  Output,
  Patch,
  State,
)
import pandas as pd
from pandera.typing import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.fin.models import Quote
from lib.fin.quote import load_ohlcv
from lib.morningstar.ticker import Stock

# from components.quote_store import QuoteStoreAIO
from components.quote_graph_type import QuoteGraphTypeAIO
from components.ticker_select import TickerSelectAIO


class QuoteDatePickerAIO(dcc.DatePickerRange):
  @staticmethod
  def aio_id(id: str):
    return {"component": "QuoteDatePickerAIO", "aio_id": id}

  def __init__(self, id: str, datepicker_props: dict | None = None):
    datepicker_props = datepicker_props.copy() if datepicker_props else {}

    datepicker_props["clearable"] = True
    datepicker_props["updatemode"] = "bothdates"

    if "display_format" not in datepicker_props:
      datepicker_props["display_format"] = "YYYY-M-D"

    super().__init__(id=self.__class__.aio_id(id), **datepicker_props)

  # @callback(
  #  Output(aio_id(MATCH), "min_date_allowed"),
  #  Output(aio_id(MATCH), "max_date_allowed"),
  #  Input(QuoteStoreAIO.aio_id(MATCH), "data"),
  # )
  # def update_datepicker(data):
  #  dates = cast(pd.DatetimeIndex, pd.to_datetime(data["date"], format="ISO8601"))
  #  return dates.min(), dates.max()


class QuoteGraphAIO(dcc.Graph):
  @staticmethod
  def aio_id(id: str):
    return {"component": "QuoteGraphAIO", "aio_id": id}

  def __init__(self, id: str, graph_props: dict | None = None):
    graph_props = graph_props.copy() if graph_props else {}

    super().__init__(id=self.__class__.aio_id(id), **graph_props)

  @callback(
    Output(aio_id(MATCH), "figure", allow_duplicate=True),
    Input(TickerSelectAIO.aio_id(MATCH), "value"),
    State(QuoteGraphTypeAIO.aio_id(MATCH), "value"),
    background=True,
    prevent_initial_call=True,
  )
  def update_data(id_currency: str, plot_type: Literal["line", "candlestick"]):
    if not id_currency:
      return no_update

    id, currency = id_currency.split("_")
    fetcher = partial(Stock(id, currency).ohlcv)
    ohlcv = asyncio.run(load_ohlcv(id_currency, "stock", fetcher))
    return quote_volume_graph(
      ohlcv, plot_type, rangeselector=("1M", "6M", "YTD", "1Y", "All")
    )


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="updateQuoteGraph"),
  Output(QuoteGraphAIO.aio_id(MATCH), "figure"),
  Input(QuoteGraphTypeAIO.aio_id(MATCH), "data"),
  Input(QuoteGraphAIO.aio_id(MATCH), "relayoutData"),
  Input(QuoteDatePickerAIO.aio_id(MATCH), "start_date"),
  Input(QuoteDatePickerAIO.aio_id(MATCH), "end_date"),
  State(QuoteGraphAIO.aio_id(MATCH), "figure"),
  prevent_initial_call=True,
)

clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="updateQuoteDatePicker"),
  Output(QuoteDatePickerAIO.aio_id(MATCH), "min_date_allowed"),
  Output(QuoteDatePickerAIO.aio_id(MATCH), "max_date_allowed"),
  Input(QuoteGraphAIO.aio_id(MATCH), "figure"),
)


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


def quote_candlestick(ohlcv: DataFrame[Quote]) -> go.Candlestick:
  if not {"open", "high", "low", "close"}.issubset(ohlcv.columns):
    return go.Candlestick()

  return go.Candlestick(
    x=ohlcv.index,
    open=ohlcv["open"],
    high=ohlcv["high"],
    low=ohlcv["low"],
    close=ohlcv["close"],
    showlegend=False,
  )


def quote_line(ohlcv: DataFrame[Quote]) -> go.Scatter:
  return go.Scatter(
    x=ohlcv.index,
    y=ohlcv["close"],
    customdata=ohlcv[["open", "high", "low", "close"]].to_numpy(),
    mode="lines",
    showlegend=False,
  )


def quote_graph(
  ohlcv: DataFrame[Quote],
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
    trace.append(quote_line(ohlcv))

  elif plot == "candlestick":
    trace.append(quote_candlestick(ohlcv))

  fig = go.Figure(data=trace, layout_xaxis=xaxis)
  return fig


def quote_volume_graph(
  ohlcv: DataFrame[Quote],
  plot: Literal["line", "candlestick"] = "line",
  rangeselector: tuple[str, ...] | None = None,
  rangeslider=False,
) -> go.Figure:
  if "volume" not in ohlcv.keys():
    return quote_graph(ohlcv, plot, rangeselector, rangeslider)

  fig = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.02,
    shared_xaxes=True,
  )
  if plot == "line":
    fig.add_trace(quote_line(ohlcv), row=1, col=1)
  elif plot == "candlestick":
    fig.add_trace(quote_candlestick(ohlcv), row=1, col=1)

  fig.add_trace(
    go.Bar(
      x=ohlcv.index,
      y=ohlcv["volume"],
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
