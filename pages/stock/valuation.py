

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
    'label': 'Skew normal',
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

def component(index: str) -> html.Div:

  options = [
    {'label': v.get('label', k.upper()), 'value': k}
    for k, v in distributions
  ]

  return html.Div(className='flex', children=[
    dcc.Dropdown(
      id={
        'type': 'dropdown:stock-valuation:distribution', 
        'index': index
      },
      options=options),
    html.Form(
      id={
        'type': 'form:stock-valuation:parameters',
        'index': index
      })
  ])

def layout(_id: str|None = None):

  columnDefs = [
    {
      'field': 'factor', 'headerName': 'Factor', 
      'pinned': 'left', 'lockPinned': True, 'cellClass': 'lock-pinned',
    },
    {
      'field': 'phase_1', 'headerName': 'Phase 1',
    },
    {
      'field': 'terminal', 'headerName': 'Terminal Phase', 
      'pinned': 'right', 'lockPinned': True, 'cellClass': 'lock-pinned',
    },
  ]

  factors = ['years', 'revenue_growth', 'operating_margin', 'tax_rate', 'beta', 
    'risk_free_rate']

  rowData = [{
    'factor': 'Years', 
    'phase_1': component('years:1'), 
    'terminal': '∞'
  }] + [{
    'factor': i.replace('_', '').capitalize(), 
    'phase_1': component(f'{i}:1'), 'terminal': component(f'{i}:t')
  } for i in factors]

  return html.Main(className='h-full flex flex-col', children=[
    StockHeader(_id),
    html.Div(children=[
      html.Button('Add', id='button:stock-valuation:dcf-add')
    ]),
    dag.AgGrid(
      id='table:stock-valuation:dcf',
      columnDefs=columnDefs,
      rowData=rowData,
      columnSize='autoSize',
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

  df = pd.DataFrame.from_records(rows)
  df.loc[:, f'phase_{n_clicks+1}'] = 'component'

  cols.append({'field': f'phase_{n_clicks+1}', 'headerName': f'Phase {n_clicks+1}'})
  
  return cols, df.to_dict('records')

@callback(
  Output({'type': 'form:stock-valuation:parameters', 'index': MATCH}, 'children'),
  Input({'type': 'dropdown:stock-valuation:distribution', 'index': MATCH}, 'value'),
  State({'type': 'dropdown:stock-valuation:distribution', 'index': MATCH}, 'id')
)
def update_inputs(distribution: str, index=dict[str,str]):

  params: dict[str,str] = distributions[distribution].get('parameters')
  inputs = []
  for k, v in params.items():
    inputs.append(dcc.Input(
      id={
        'type': f'input:stock-valuation:{distribution}-{k}',
        'index': index['index']
      },
      type='number',
      placeholder=v
    ))

  return inputs
