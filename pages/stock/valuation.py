

from dash import callback, html, no_update, register_page, Output, Input, State
import dash_ag_grid as dag
import pandas as pd

from components.stock_header import StockHeader
from lib.ticker.fetch import stock_label

#register_page(__name__, path_template='/stock/<_id>/valuation', title=stock_label)
register_page(__name__, path_template='/valuation')

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

  rowData = [
    {'factor': 'Years', 'phase_1': 'component', 'terminal': 'component'},
    {'factor': 'Revenue Growth', 'phase_1': 'component', 'terminal': 'component'},
    {'factor': 'Operating Margin', 'phase_1': 'component', 'terminal': 'component'},
    {'factor': 'Tax Rate', 'phase_1': 'component', 'terminal': 'component'},
    {'factor': 'Beta', 'phase_1': 'component', 'terminal': 'component'},
    {'factor': 'Risk Free Rate', 'phase_1': 'component', 'terminal': 'component'}
  ]

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
