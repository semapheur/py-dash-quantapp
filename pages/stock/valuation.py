import json

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input, State, MATCH)
import dash_ag_grid as dag
import pandas as pd

from components.stock_header import StockHeader
from lib.ticker.fetch import stock_label

#register_page(__name__, path_template='/stock/<_id>/valuation', title=stock_label)
register_page(__name__, path_template='/valuation')

distributions = {
  'normal': {
    'parameters': {
      'mu': 'Mean',
      'sigma': 'Scale'
    }
  },
  'skewnormal': {
    'parameters': {
      'a': 'Skewness',
      'loc': 'Mean',
      'scale': 'Scale'
    }
  },
  'triangular': {
    'parameters': {
      'a': 'Lower bound',
      'm': 'Mode',
      'b': 'Upper bound'
    }
  },
  'uniform': {
    'parameters': {
      'a': 'Lower bound',
      'b': 'Upper bound'
    }
  }
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
            'values': [k.capitalize() for k in distributions.keys()]
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
            'values': [k.capitalize() for k in distributions.keys()]
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

  factors = ['revenue_growth', 'operating_margin', 'tax_rate', 'beta', 
    'risk_free_rate']
  
  _factors = {
    'revenue_growth': {
      'intial': '(0.15,0.3)',
      'terminal': '(0.02,0.005)'
    }
  }

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
  } for i in factors]

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
          'values': [k.capitalize() for k in distributions.keys()]
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
  State('button:stock-valuation:dcf-sim', 'className')
)
def monte_carlo(n_clicks: int, data: list[dict], cls: str):
  print('yay')
  if not n_clicks:
    return no_update
  
  print(json.dumps(data, indent=2))

  return cls