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
form_style = 'grid grid-cols-[2fr_1fr_1fr] gap-2 p-2 shadow rounded-md'
overview_style = 'h-full grid grid-cols-[1fr_1fr] shadow rounded-md'
dropdown_style = 'w-1/3'

layout = html.Main(className=main_style, children=[
  html.Form(className=form_style, children=[
    TickerSelectAIO(aio_id='stats'),
    dcc.Input(id='stats-input:diff-order', type='number', 
      className='border rounded pl-2',
      min=0, max=10, value=0),
    dcc.Dropdown(id='stats-dropdown:transform',
      options=[
        {'label': 'Log', 'value': 'log'}
      ],
      value=''
    )
  ]),
  dcc.Tabs(value='tab-transform',
    className='inset-row',
    content_className=overview_style, 
    children=[
      dcc.Tab(label='Transform', value='tab-transform', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:price', responsive=True),
          dcc.Graph(id='stats-graph:transform', responsive=True)
      ]),
      dcc.Tab(label='Normality', value='tab-normality', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:distribution', responsive=True),
          dcc.Graph(id='stats-graph:qq', responsive=True)
      ]),
      dcc.Tab(label='Stationarity', value='tab-stationarity', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:acf', responsive=True),
          dcc.Graph(id='stats-graph:pacf', responsive=True)
      ]), 
  ]),
  dcc.Store(id='stats-store:price'),
  dcc.Store(id='stats-store:transform')
])

@callback(
  Output('stats-store:price', 'data'),
  Input(TickerSelectAIO._id('stats'), 'value'),
)
def update_store(query):
  if not query:
    return no_update
    
  id, currency = query.split('|')
  price = get_ohlcv(id, 'stock', currency, cols=['close'])

  price.reset_index(inplace=True)
  return price.to_dict('list')

@callback(
  Output('stats-store:transform', 'data'),
  Input('stats-store:price', 'data'),
  Input('stats-input:diff-order', 'value'),
  Input('stats-dropdown:transform', 'value')
)
def update_store(data, diff_order, transform):
  if not data:
    return no_update

  price = pd.DataFrame.from_dict(data, orient='columns')    

  if transform == 'log':
    price['close'] = np.log(price['close'])

  if diff_order:
    if isinstance(diff_order, int):
      price['close'] = np.diff(price['close'], n=diff_order, prepend=[np.nan] * diff_order)
    elif isinstance(diff_order, float):
      price['close'] = frac_diff(price['close'].to_numpy(), diff_order)

    price.dropna(inplace=True)

  price.dropna(inplace=True)
  price.rename(columns={'close': 'transform'}, inplace=True)
  price.reset_index(inplace=True)
  return price.to_dict('list')

@callback(
  Output('stats-graph:price', 'figure'),
  Input('stats-store:price', 'data')
)
def update_graph(data):
  
  fig = go.Figure()
  fig.add_scatter(
    x=data['date'],
    y=data['close'],
    mode='lines',
    showlegend=False,
  )
  fig.update_layout(
    title='Price'
  )
  return fig

@callback(
  Output('stats-graph:transform', 'figure'),
  Input('stats-store:transform', 'data')
)
def update_graph(data):
  
  fig = go.Figure()
  fig.add_scatter(
    x=data['date'],
    y=data['transform'],
    mode='lines',
    showlegend=False,
  )
  fig.update_layout(
    title='Transformation'
  )
  return fig

@callback(
  Output('stats-graph:distribution', 'figure'),
  Input('stats-store:transform', 'data')
)
def update_graph(data):
  
  fig = go.Figure()
  fig.add_histogram(
    x=data['transform'],
    #histnorm='probability'
  )
  fig.update_layout(
    title='Distribution'
  )
  return fig

@callback(
  Output('stats-graph:qq', 'figure'),
  Input('stats-store:transform', 'data')
)
def update_graph(data):
  transform = np.array(data['transform'])
  qq_trace = qqplot_traces(transform)

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
  Output('stats-graph:acf', 'figure'),
  Input('stats-store:transform', 'data')
) 
def update_acf(data):
  transform = np.array(data['transform'])
  acf_trace = acf_traces(transform, False)

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
  Output('stats-graph:pacf', 'figure'),
  Input('stats-store:transform', 'data')
) 
def update_graph(data):
  transform = np.array(data['transform'])
  pacf_trace = acf_traces(transform, True)

  fig = go.Figure()
  fig.add_traces(pacf_trace)
  fig.update_layout(
    showlegend=False,
    title='Partial autocorrelation',
  )
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig