import numpy as np
import pandas as pd
from dash import callback, ctx, dcc, html, no_update, register_page, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.time_series import fast_frac_diff as frac_diff
from lib.morningstar import get_ohlcv
from components.ticker_select import TickerSelectAIO
from components.autocorrelation_plot import autocorrelation_traces

register_page(__name__, path='/statistics')

main_style = 'h-full flex flex-col'
dropdown_style = 'w-1/3'
layout = html.Main(className=main_style, children=[
  html.Form(className='grid grid-cols-[2fr_1fr] gap-2 p-2', children=[
    TickerSelectAIO(aio_id='stats'),
    dcc.Input(id='stats:input-diff-order', type='number', 
      className='border rounded pl-2',
      min=0, max=10, value=0)
  ]),
  html.Div(className='overflow-y-scroll', children=[
    dcc.Graph(id='stats:return-plots'),
    dcc.Graph(id='stats:acf-plots'),
  ]),
  dcc.Store(id='stats:return-store')
])

@callback(
  Output('stats:return-store', 'data'),
  Input(TickerSelectAIO._id('stats'), 'value'),
  Input('stats:input-diff-order', 'value')
)
def update_store(query, diff_order):
  if not query:
    return no_update
    
  id, currency = query.split('|')
  price = get_ohlcv(id, 'stock', currency, cols=['close'])

  if diff_order:
    price['close'] = np.log(price['close'])

    if isinstance(diff_order, int):
      price['close'] = np.diff(price['close'], n=diff_order, prepend=[np.nan] * diff_order)
    elif isinstance(diff_order, float):
      price['close'] = frac_diff(price['close'].to_numpy(), diff_order)

    price.dropna(inplace=True)

  price['log_return'] = np.log(price['close'] / price['close'].shift(1))
  price.reset_index(inplace=True)
  return price.to_dict('list')

@callback(
  Output('stats:return-plots', 'figure'),
  Input('stats:return-store', 'data')
)
def update_graph(data):
  
  fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=['Price', 'Log return', 'Log return distribution']
  )
  fig.add_scatter(
    x=data['date'],
    y=data['close'],
    mode='lines',
    showlegend=False,
    row=1, col=1
  )
  fig.add_scatter(
    x=data['date'],
    y=data['log_return'],
    mode='lines',
    showlegend=False,
    row=1, col=2
  )
  fig.add_histogram(
    x=data['log_return'], 
    showlegend=False,
    row=1, col=3
  )
  return fig

@callback(
  Output('stats:acf-plots', 'figure'),
  Input('stats:return-store', 'data')
) 
def update_acf(data):
  price = pd.Series(data=data['log_return'], index=data['date'])

  acf_trace = autocorrelation_traces(price)
  pacf_trace = autocorrelation_traces(price, True)

  fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['ACF', 'PACF']
  )

  for i, traces in enumerate([acf_trace, pacf_trace], start=1):
    fig.add_traces(traces, rows=1, cols=i)

  fig.update_traces(showlegend=False)
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig