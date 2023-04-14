import numpy as np
import pandas as pd
from dash import callback, ctx, dcc, html, no_update, register_page, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import laplace, norm, gennorm, t
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from lib.fracdiff import fast_frac_diff as frac_diff
from lib.morningstar import get_ohlcv
from components.ticker_select import TickerSelectAIO
from components.statistical_plots import acf_trace, qqplot_trace, msdr_trace

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
    ),
  ]),
  dcc.Tabs(
    id='stats-tabs',
    value='tab-transform',
    className='inset-row',
    content_className=overview_style,
    parent_className='h-full',
    children=[
      dcc.Tab(label='Transform', value='tab-transform', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:price', responsive=True),
          dcc.Graph(id='stats-graph:transform', responsive=True)
      ]),
      dcc.Tab(label='Distribution', value='tab-distribution', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:distribution'),
          html.Div(className='h-full flex flex-col', children=[
            dcc.Graph(id='stats-graph:qq'),
            dcc.Dropdown(id='stats-dropdown:qq-distribution',
              className='pb-2 px-2 drop-up',
              options=[
                {'label': 'Normal', 'value': 'norm'},
                {'label': 'Laplace', 'value': 'laplace'},
                {'label': 'Generalized Normal', 'value': 'gennorm'},
                {'label': 'Student-t', 'value': 't'}
              ],
              placeholder='Distribution',
              value='norm'
            ),
          ]),
      ]),
      dcc.Tab(label='Autocorrelation', value='tab-autocorrelation', 
        className='inset-row',
        children=[
          dcc.Graph(id='stats-graph:acf', responsive=True),
          dcc.Graph(id='stats-graph:pacf', responsive=True)
      ]),
      dcc.Tab(label='Regimes', value='tab-regimes', 
        className='inset-row',
        children=[
          html.Div(className='flex flex-col', children=[
            dcc.Graph(id='stats-graph:regimes', className='h-full'),
            html.Form(className='pb-2 pl-2', children=[
              dcc.Input(id='stats-input:regimes', 
                className='border rounded pl-2', 
                type='number', placeholder='Regimes',
                min=1, max=5, step=1, value=2
              ),
            ])
          ]),
          dcc.Graph(id='stats-graph:regime-distribution')
      ]),  
  ]),
  dcc.Store(id='stats-store:price'),
  dcc.Store(id='stats-store:transform'),
  dcc.Store(id='stats-store:model')
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
  Output('stats-store:model', 'data'),
  Input('stats-store:transform', 'data'),
  Input('stats-input:regimes', 'value')
)
def update_store(data, regimes):
  if not (data and regimes):
    return no_update

  msdr = MarkovRegression(
    data['transform'], 
    k_regimes=regimes, 
    trend='c', 
    switching_variance=True
  ).fit()
  
  return {'model': msdr.smoothed_marginal_probabilities}

@callback(
  Output('stats-graph:price', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:price', 'data')
)
def update_graph(tab, data):
  if not data or tab != 'tab-transform':
    return no_update
  
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
  Input('stats-tabs', 'value'),
  Input('stats-store:transform', 'data')
)
def update_graph(tab, data):
  if not data or tab != 'tab-transform':
    return no_update

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
  Input('stats-tabs', 'value'),
  Input('stats-store:transform', 'data')
)
def update_graph(tab, data):
  if not data or tab != 'tab-distribution':
    return no_update

  x = np.linspace(np.min(data['transform']), np.max(data['transform']), 100)
  mean = np.mean(data['transform'])
  std = np.std(data['transform'])
  ggd_params = gennorm.fit(data['transform'], floc=mean, fscale=std)
  t_params = t.fit(data['transform'], floc=mean, fscale=std)
  
  fig = go.Figure()
  fig.add_histogram(
    x=data['transform'],
    histnorm='probability density',
    name='Transform'
  )
  fig.add_scatter(
    x=x,
    y=gennorm.pdf(x, beta=ggd_params[0], loc=ggd_params[1], scale=ggd_params[2]),
    mode='lines',
    name=f'GGD ({ggd_params[0]:.2f})'
  )
  fig.add_scatter(
    x=x,
    y=t.pdf(x, df=t_params[0], loc=t_params[1], scale=t_params[2]),
    mode='lines',
    name=f'S-t ({t_params[0]:.2f})'
  )
  fig.update_layout(
    title='Distribution',
    legend=dict(
      yanchor='top',
      y=0.99,
      xanchor='left',
      x=0.01
    )
  )
  return fig

@callback(
  Output('stats-graph:qq', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:transform', 'data'),
  Input('stats-dropdown:qq-distribution', 'value')
)
def update_graph(tab, data, dist):
  if not data or not dist or tab != 'tab-distribution':
    return no_update

  transform = np.array(data['transform'])

  params = [np.mean(transform), np.std(transform)]
  if dist == 'gennorm':
    params = gennorm.fit(transform, floc=params[0], fscale=params[1])
  elif dist == 't':
    params = t.fit(transform, floc=params[0], fscale=params[1])

  trace = qqplot_trace(transform, dist, tuple(params))

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(
    showlegend=False,
    title='Quantile-quantile plot',
    xaxis_title='Theoretical quantiles',
    yaxis_title='Sample quantile',
  )
  return fig

@callback(
  Output('stats-graph:acf', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:transform', 'data')
) 
def update_acf(tab, data):
  if not data or tab != 'tab-autocorrelation':
    return no_update

  transform = np.array(data['transform'])
  trace = acf_trace(transform, False)

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(
    showlegend=False,
    title='Autocorrelation'
  )
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig

@callback(
  Output('stats-graph:pacf', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:transform', 'data')
) 
def update_graph(tab, data):
  if not data or tab != 'tab-autocorrelation':
    return no_update

  transform = np.array(data['transform'])
  trace = acf_trace(transform, True)

  fig = go.Figure()
  fig.add_traces(trace)
  fig.update_layout(
    showlegend=False,
    title='Partial autocorrelation',
  )
  #fig.update_xaxes(range=[-1,42])
  fig.update_yaxes(zerolinecolor='#000000')

  return fig

@callback(
  Output('stats-graph:regimes', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:model', 'data'),
  State('stats-store:transform', 'data'),
  State('stats-store:price', 'data'),
) 
def update_graph(tab, model, transform, price):
  if not model or tab != 'tab-regimes':
    return no_update

  model = np.array(model['model'])
  regimes = model.shape[-1]

  state = pd.Series(
    np.argmax(model, axis=1),
    index=transform['date']
  )

  price = pd.Series(price['close'], index=price['date'])
  price = price.loc[state.index]

  fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True
  )

  fig.add_scatter(
    x=price.index,
    y=price.values,
    mode='lines',
    name=f'Close',
    row=2, col=1
  )

  for r in range(regimes):
    fig.add_scatter(
      x=transform['date'],
      y=model[:,r],
      mode='lines',
      name=f'Regime {r}',
      row=1, col=1
    )

    temp = price.loc[state == r]

    fig.add_scattergl(
      x=temp.index,
      y=temp.values,
      mode='markers',
      marker_size=3,
      name=f'Regime {r}',
      row=2, col=1
    )
  
  fig.update_layout(
    title='Regime probabilities',
    legend=dict(
      orientation='h',
      yanchor='bottom',
      y=1.02,
      xanchor='right',
      x=1,
    )
  )

  return fig

@callback(
  Output('stats-graph:regime-distribution', 'figure'),
  Input('stats-tabs', 'value'),
  Input('stats-store:model', 'data'),
  State('stats-store:transform', 'data'),
) 
def update_graph(tab, model, transform):
  if not model or tab != 'tab-regimes':
    return no_update

  model = np.array(model['model'])
  regimes = model.shape[-1]

  state = pd.Series(
    np.argmax(model, axis=1),
    index=transform['date']
  )
  transform = pd.Series(
    transform['transform'], 
    index=transform['date']
  )

  fig = go.Figure()

  for r in range(regimes):
    temp = transform.loc[state == r]

    fig.add_histogram(
      x=temp.values,
      histnorm='probability density',
      name=f'Regime {r}',
    )
  
  fig.update_layout(
    title='Regime distributions',
    legend=dict(
      yanchor='top',
      y=0.99,
      xanchor='left',
      x=0.01
    )
  )

  return fig