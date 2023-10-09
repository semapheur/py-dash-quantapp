from enum import Enum

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input, State)
import dash_ag_grid as dag
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px

from components.stock_header import StockHeader
from lib.fin.dcf import discount_cashflow, make_distribution, terminal_value
from lib.ticker.fetch import stock_label

register_page(__name__, path_template='/stock/<_id>/valuation', title=stock_label)

distributions = [
  'Normal',
  'Skewnormal',
  'Triangular',
  'Uniform'
]

factors = {
  'years': {
    'initial': ('Uniform', '5, 10'),
    'terminal': ('∞', '')
  },
  'revenue_growth': {
    'initial': ('Normal', '0.15, 0.05'),
    'terminal': ('Normal', '0.02, 0.001'),
  },
  'operating_margin': {
    'initial': ('Normal', '0.10, 0.02'),
    'terminal': ('Normal', '0.10, 0.02'),
  },
  'tax_rate': {
    'initial': ('Normal', '0.25, 0.05'),
    'terminal': ('Normal', '0.25, 0.05'),
  },
  'reinvestment_rate': {
    'initial': ('Normal', '0.10, 0.05'),
    'terminal': ('Normal', '0.10, 0.05'),
  },
  'risk_free_rate': {
    'initial': ('Normal', '0.04, 0.01'),
    'terminal': ('Normal', '0.02, 0.05'),
  },
  'yield_spread': {
    'initial': ('Normal', '0.02, 0.01'),
    'terminal': ('Normal', '0.01, 0.005'),
  },
  'equity_risk_premium': {
    'initial': ('Normal', '0.02, 0.01'),
    'terminal': ('Normal', '0.01, 0.005'),
  },
  'equity_to_capital': {
    'initial': ('Normal', '0.5, 0.3'),
    'terminal': ('Normal', '0.5, 0.3'),
  },
  'beta': {
    'initial': ('Normal', '0.5, 0.3'),
    'terminal': ('Normal', '0.5, 0.3'),
  }
}

Factors = Enum('Factors', list(factors.keys()), start=0)

correlation = {
  ('risk_free_rate', 'yield_spread'): 0.9,
  ('risk_free_rate', 'equity_risk_premium'): 0.9,
  ('equity_risk_premium', 'revenue_growth'): 0.4,
  ('reinvestment_rate', 'operating_margin'): 0.8,
  ('yield_spread', 'operating_margin'): 0.8
}

def layout(_id: str|None = None):

  column_defs = [
    {
      'field': 'factor', 'headerName': 'Factor', 'editable': False,
      'pinned': 'left', 'lockPinned': True, 'cellClass': 'lock-pinned',
    },
    {
      'headerName': 'Phase 1', 'children': [
        {
          'field': 'phase_1:distribution', 'headerName': 'Distribution',
          'cellEditor': 'agSelectCellEditor',
          'cellEditorParams': {
            'values': distributions
          },
        },
        {
          'field': 'phase_1:parameters', 'headerName': 'Parameters',
          'cellEditor': {'function': 'ParameterInput'}
        }
      ]
    },
    {
      'headerName': 'Terminal Phase', 'children': [
        {
          'field': 'terminal:distribution', 'headerName': 'Distribution',
          'pinned': 'right', 'lockPinned': True, 'cellClass': 'lock-pinned',
          'cellEditor': 'agSelectCellEditor',
          'cellEditorParams': {
            'values': distributions
          },
        },
        {
          'field': 'terminal:parameters', 'headerName': 'Parameters',
          'pinned': 'right', 'lockPinned': True, 'cellClass': 'lock-pinned',
          'cellEditor': {'function': 'ParameterInput'}
        }
      ]
    },
  ]

  row_data = [{
    'factor': k.replace('_', ' ').capitalize(), 
    'phase_1:distribution': factors[k]['initial'][0],
    'phase_1:parameters': factors[k]['initial'][1],
    'terminal:distribution': factors[k]['terminal'][0],
    'terminal:parameters': factors[k]['terminal'][1],
  } for k in factors]

  return html.Main(className='h-full flex flex-col', children=[
    StockHeader(_id),
    html.Div(children=[
      html.Button('Add', id='button:stock-valuation:dcf-add'),
      html.Button('Calc', id='button:stock-valuation:dcf-sim')
    ]),
    dag.AgGrid(
      id='table:stock-valuation:dcf',
      columnDefs=column_defs,
      rowData=row_data,
      columnSize='autoSize',
      defaultColDef={'editable': True},
      dashGridOptions={'singleClickEdit': True},
      style={'height': '100%'}
    ),
    dcc.Graph(id='graph:stock-valuation:dcf')
  ])

@callback(
  Output('table:stock-valuation:dcf', 'columnDefs'),
  Output('table:stock-valuation:dcf', 'rowData'),
  Input('button:stock-valuation:dcf-add', 'n_clicks'),
  State('table:stock-valuation:dcf', 'columnDefs'),
  State('table:stock-valuation:dcf', 'rowData'),
)
def update_table(n_clicks: int, cols: list[dict], rows: list[dict]):
  if not n_clicks:
    return no_update

  phase = len(cols) - 1
  df = pd.DataFrame.from_records(rows)
  df.loc[:, f'phase_{phase}:distribution'] = 'Normal'
  df.loc[:, f'phase_{phase}:parameters'] = ''

  cols.append({
    'headerName': f'Phase {phase}', 'children': [
      {
        'field': f'phase_{phase}:distribution', 'headerName': 'Distribution',
        'cellEditor': 'agSelectCellEditor',
        'cellEditorParams': {
          'values': distributions
        },
      },
      {
        'field': f'phase_{phase}:parameters', 'headerName': 'Parameters',
        'cellEditor': {'function': 'ParameterInput'},
      }
    ]
  })
  return cols, df.to_dict('records')

@callback(
  Output('graph:stock-valuation:dcf', 'figure'),
  Input('button:stock-valuation:dcf-sim', 'n_clicks'),
  State('table:stock-valuation:dcf', 'rowData'),
  State('store:ticker-search:financials', 'data')
)
def monte_carlo(n_clicks: int, rowData: list[dict], financials: list[dict]):
  if not (n_clicks and financials):
    return no_update
  
  n = 1000

  fin = pd.DataFrame.from_records(financials) 
  fin = fin.set_index(['date', 'months']).sort_index(level='date')       

  revenue = fin.loc[(slice(None), 12), 'revenue'].iloc[-1]

  df = pd.DataFrame.from_records(rowData)

  phases = len(df.columns) - 3 // 2

  corr_mat = ot.CorrelationMatrix(len(factors))
  for pair, value in correlation.items():
    ix = (Factors[pair[0]].value, Factors[pair[1]].value)
    corr_mat[ix] = value

  copula = ot.NormalCopula(corr_mat)

  dcf = np.array([0, revenue, 0]).repeat(n, 1)
  for p in range(1, phases):

    variables = [
      make_distribution(dist, params) for dist, params in 
      zip(df[f'phase_{p}:distribution'], df[f'phase_{p}:parameters'])
    ]
    composed_distribution = ot.ComposedDistribution(variables, copula)
    sample = np.array(composed_distribution.getSample(n))
    args = np.concatenate((dcf[:2,:], sample), axis=1)
    dcf += np.apply_along_axis(discount_cashflow, 0, args)

  dcf = dcf[2,:] 

  # terminal value
  variables = [
    make_distribution(dist, params) for dist, params in 
    zip(
      df['terminal:distribution'].iloc[1:], 
      df['terminal:parameters'].iloc[1:]
    )
  ]
  composed_distribution = ot.ComposedDistribution(variables, copula)
  sample = np.array(composed_distribution.getSample(n))
  args = np.concatenate((dcf[1,:], sample), axis=1)
  dcf += np.apply_along_axis(terminal_value, 0, args)

  fig = px.histogram(dcf)

  return fig