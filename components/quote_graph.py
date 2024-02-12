import re
from typing import Optional
import uuid

from dash import dcc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class QuoteGraphAIO(dcc.Graph):
  @staticmethod
  def id(aio_id):
    return {'component': 'QuoteGraphAIO', 'aio_id': aio_id}

  def __init__(self, aio_id: str | None = None, graph_props: dict | None = None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    graph_props = graph_props.copy() if graph_props else {}

    super().__init__(id=self.__class__.id(aio_id), **graph_props)


def quote_rangeselector(selector: list[str]) -> dict[str, list[dict[str, int | str]]]:
  pattern = r'^(?P<count>\d+)(?P<step>\w)$'
  step_code = dict(d='day', m='month', w='week', y='year')

  buttons: list[dict[str, int | str]] = []
  for s in selector:
    m = re.match(pattern, s)
    if m is not None:
      details = m.groupdict()
      buttons.append(
        dict(
          count=int(details['count']),
          label=s,
          step=step_code[details['step'].lower()],
          stepmode='backward',
        )
      )
    elif s.lower() == 'ytd':
      buttons.append(dict(count=1, label=s, step='year', stepmode='todate'))
    elif s.lower() == 'all':
      buttons.append(dict(label=s, step='all'))
  return dict(buttons=buttons)


def quote_candlestick(data: dict):
  if not {'open', 'high', 'low', 'close'}.issubset(data.keys()):
    return {}

  return go.Candlestick(
    x=data['date'],
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    showlegend=False,
  )


def quote_line(data: dict) -> go.Scatter:
  return go.Scatter(x=data['date'], y=data['close'], mode='lines', showlegend=False)


def quote_graph(
  data: dict[str, list[str | float | int]],
  plot='line',
  rangeselector: list[str] = [],
  rangeslider=False,
) -> go.Figure:
  xaxis = dict(type='date', rangeslider=dict(visible=False))
  if rangeselector:
    xaxis['rangeselector'] = quote_rangeselector(rangeselector)

  if rangeslider:
    xaxis['rangeslider'] = dict(visible=True)

  trace = []
  if plot == 'line':
    trace.append(quote_line(data))

  elif plot == 'candlestick':
    trace.append(quote_candlestick(data))

  fig = go.Figure(data=trace, layout_xaxis=xaxis)
  return fig


def quote_volume_graph(
  data: dict[str, list[str | float | int]],
  plot='line',
  rangeselector: Optional[tuple[str, ...]] = None,
  rangeslider=False,
) -> go.Figure:
  if 'volume' not in data.keys():
    return quote_graph(data, plot, rangeselector, rangeslider)

  fig = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.02,
    shared_xaxes=True,
  )
  if plot == 'line':
    fig.add_trace(quote_line(data), row=1, col=1)
  elif plot == 'candlestick':
    fig.add_trace(quote_candlestick(data), row=1, col=1)

  fig.add_trace(
    go.Bar(
      x=data['date'],
      y=data['volume'],
      showlegend=False,
      marker_color='orange',
    ),
    row=2,
    col=1,
  )

  xaxis = dict(type='date', rangeslider=dict(visible=False))
  if rangeselector:
    xaxis['rangeselector'] = quote_rangeselector(rangeselector)

  if rangeslider:
    xaxis['rangeslider'] = dict(visible=True)

  fig.update_layout(xaxis=xaxis)

  return fig


def quote_graph_range(
  data: dict[str, list[str | float | int]],
  cols: list[str],
  fig: go.Figure,
  start_date: str,
  end_date: str,
) -> go.Figure:
  ohlcv = pd.DataFrame.from_dict(data)
  ohlcv.set_index('date', inplace=True)
  ohlcv.index = pd.to_datetime(ohlcv.index)

  ohlcv = ohlcv.loc[start_date:end_date, cols]

  fig['layout']['xaxis']['range'] = [start_date, end_date]
  margin = np.abs(ohlcv[cols[0]].min() / 100)
  fig['layout']['yaxis']['range'] = [
    ohlcv[cols[0]].min() - margin,
    ohlcv[cols[0]].max() + margin,
  ]
  fig['layout']['yaxis']['autorange'] = False

  for i in range(1, len(cols)):
    margin = np.abs(ohlcv[cols[i]].min() / 100)
    fig['layout'][f'yaxis{i+1}']['range'] = [
      ohlcv[cols[i]].min() - margin,
      ohlcv[cols[i]].max() + margin,
    ]
    fig['layout'][f'yaxis{i+1}']['autorange'] = False

  return fig


def quote_graph_relayout(
  relayout: dict,
  data: dict[str, list[str | float | int]],
  cols: list[str],
  fig: go.Figure,
) -> go.Figure:
  if all(x in relayout.keys() for x in ['xaxis.range[0]', 'xaxis.range[1]']):
    fig = quote_graph_range(
      data, cols, fig, relayout['xaxis.range[0]'], relayout['xaxis.range[1]']
    )
  elif 'xaxis.autorange' in relayout.keys():
    fig['layout']['xaxis']['autorange'] = True
    fig['layout']['yaxis']['autorange'] = True

    for i in range(1, len(cols)):
      fig['layout'][f'yaxis{i+1}']['autorange'] = True

  return fig
