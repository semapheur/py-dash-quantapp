import json

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input, State, MATCH)
import dash_ag_grid as dag
import numpy as np
import openturns as ot
import pandas as pd

from components.stock_header import StockHeader
from lib.fin.dcf import discount_cashflow, make_distribution
from lib.ticker.fetch import stock_label

#register_page(__name__, path_template='/stock/<_id>/valuation', title=stock_label)
register_page(__name__, path_template='/valuation')

distributions = [
  'Normal',
  'Skewnormal',
  'Triangular',
  'Uniform'
]

factors = [
  'years'
  'revenue_growth', 
  'operating_margin', 
  'tax_rate',
  'reinvestment_rate',
  'risk_free_rate',
  'yield_spread',
  'equity_risk_premium',
  'equity_to_capital',
  'beta',
]

_factors = {
  'revenue_growth': {
    'intial': '(0.15,0.3)',
    'terminal': '(0.02,0.005)'
  }
}

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
    'factor': 'Years', 
    'phase_1:distribution': 'Uniform',
    'phase_1:parameters': '(5,10)',
    'terminal:distribution': '∞',
    'terminal:parameters': '',
  }] + [{
    'factor': i.replace('_', ' ').capitalize(), 
    'phase_1:distribution': 'Normal',
    'phase_1:parameters': '',
    'terminal:distribution': 'Normal',
    'terminal:parameters': '',
  } for i in factors[1:]]

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
    )
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
  Output('button:stock-valuation:dcf-sim','className'),
  Input('button:stock-valuation:dcf-sim', 'n_clicks'),
  State('table:stock-valuation:dcf', 'rowData'),
  State('button:stock-valuation:dcf-sim', 'className'),
  State('store:ticker-search:financials', 'data')
)
def monte_carlo(n_clicks: int, rowData: list[dict], cls: str, financials: list[dict]):
  if not n_clicks:
    return no_update
  
  n = 1000

  fin = pd.DataFrame.from_records(financials) 
  fin = fin.set_index(['date', 'months']).sort_index('date')       

  revenue = fin.loc[(slice(None), 12), 'revenue'].iloc[-1]

  df = pd.DataFrame.from_records(rowData)

  phases = len(df.columns) - 3 // 2

  corr_mat = ot.CorrelationMatrix(len(factors))
  df_factors = pd.DataFrame([dict(zip(factors, range(0, factors)))])

  for pair, value in correlation.items():
    location = df_factors[pair].values[0]
    corr_mat[location[0], location[1]] = value

  copula = ot.NormalCopula(corr_mat)

  result = np.array([0, revenue, 0]).repeat(n, 1)
  for p in range(1, phases):

    variables = [
      make_distribution(dist, params) for dist, params in 
      zip(df[f'phase_{p}:distribution'], df[f'phase_{p}:parameters'])
    ]
    composed_distribution = ot.ComposedDistribution(variables, copula)
    sample = np.array(composed_distribution.getSample(n))
    args = np.concatenate((result[:2,:], sample), axis=1)
    temp = np.apply_along_axis(discount_cashflow, 0, args)
    result += temp

  # terminal value

  return cls