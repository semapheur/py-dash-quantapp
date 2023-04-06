import numpy as np
import pandas as pd
from dash import callback, ctx, dcc, html, no_update, register_page, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.time_series import fast_frac_diff as frac_diff
from lib.morningstar import get_ohlcv
from components.ticker_select import TickerSelectAIO
from components.statistical_plots import acf_traces, qqplot_traces

register_page(__name__, path='/statistics')

main_style = 'h-full flex flex-col gap-2 p-2'
form_style = 'grid grid-cols-[2fr_1fr] gap-2 p-2 shadow rounded-md'
tabs_style = 'h-min w-max'
tab_style = 'p-2 rounded-t-md'
tab_selected_style = ''
overview_style = 'h-full grid grid-cols-[1fr_1fr] shadow rounded-md'
dropdown_style = 'w-1/3'
layout = html.Main(className=main_style, children=[
  html.Form(className=form_style, children=[
    TickerSelectAIO(aio_id='stats'),
    dcc.Input(id='stats:input-diff-order', type='number', 
      className='border rounded pl-2',
      min=0, max=10, value=0)
  ]),
  dcc.Tabs(id='stats:tabs', value='tab-returns',
    className=tabs_style,
    parent_className='h-full',
    content_className=overview_style, 
    children=[
      dcc.Tab(label='Returns', value='tab-returns', 
        className=tab_style,
        children=[
          dcc.Graph(id='stats:price-return-plot', responsive=True),
      ]),
      dcc.Tab(label='Normality', value='tab-normality', 
        className=tab_style,
        children=[
          dcc.Graph(id='stats:distribution-plot', responsive=True),
          dcc.Graph(id='stats:qq-plot', responsive=True)
      ]),
      dcc.Tab(label='Stationarity', value='tab-stationarity', 
        className=tab_style,
        children=[
          dcc.Graph(id='stats:acf-plot', responsive=True),
          dcc.Graph(id='stats:pacf-plot', responsive=True)
      ]), 
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
  price.dropna(inplace=True)
  price.reset_index(inplace=True)
  return price.to_dict('list')

@callback(
  Output('stats:price-return-plot', 'figure'),
  Input('stats:return-store', 'data')
)
def update_graph(data):
  
  fig = make_subplots(
    rows=1, cols=2,
    shared_xaxes=True,
    horizontal_spacing=0.01,
    subplot_titles=['Close price', 'Log return']
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
  return fig

@callback(
  Output('stats:distribution-plot', 'figure'),
  Input('stats:return-store', 'data')
)
def update_graph(data):
  log_return = np.array(data['log_return'])
  
  fig = go.Figure()
  fig.add_histogram(
    x=log_return,
    #histnorm='probability'
  )
  fig.update_layout(
    title='Log return distribution'
  )
  return fig

@callback(
  Output('stats:qq-plot', 'figure'),
  Input('stats:return-store', 'data')
)
def update_graph(data):
  log_return = np.array(data['log_return'])
  qq_trace = qqplot_traces(log_return)

  fig = go.Figure()
  fig.add_traces(qq_trace)
  fig.update_layout(
    showlegend=False,
    title='Quantile-quantile plot',
    xaxis_title='Theoretical quantiles',
    yaxis_title='Sample quantile',
  )
  return fig

@callback(
  Output('stats:acf-plot', 'figure'),
  Input('stats:return-store', 'data')
) 
def update_acf(data):
  log_return = np.array(data['log_return'])
  acf_trace = acf_traces(log_return, False)

  fig = go.Figure()
  fig.add_traces(acf_trace)
  fig.update_layout(
    showlegend=False,
    title='Autocorrelation'
  )
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig

@callback(
  Output('stats:pacf-plot', 'figure'),
  Input('stats:return-store', 'data')
) 
def update_graph(data):
  log_return = np.array(data['log_return'])
  pacf_trace = acf_traces(log_return, True)

  fig = go.Figure()
  fig.add_traces(pacf_trace)
  fig.update_layout(
    showlegend=False,
    title='Partial autocorrelation',
  )
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig