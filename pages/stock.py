import asyncio
from dash import (
  callback, dcc, html, no_update, register_page, dash_table, Output, Input
)
import pandas as pd

from lib.edgar.ticker import Ticker
from lib.ticker.fetch import stock_label, find_cik

register_page(__name__, path_template='/stock/<id>', title=stock_label)

def layout(id: str = None):

  # Check CIK
  cik = find_cik(id)

  data = {}
  if cik:
    df = asyncio.run(Ticker(cik).financials())
    data = df.reset_index().to_dict('records')

  return html.Main(className='h-full', children=[
    html.Div(id='stock-div:table-wrap', className='h-full p-2'),
    dcc.Store(id='stock-store:financials', data=data)
  ])

@callback(
  Output('stock-div:table-wrap', 'children'),
  Input('stock-store:financials', 'data'),
  #Input('stock-scope:annual/quarterly')
)
def update_table(data: list[dict]):
  if not data:
    return no_update
  
  df = pd.DataFrame.from_records(data) \
    .set_index(['date', 'period']).xs('a', level=1).T

  return dash_table.DataTable(
    df.reset_index().to_dict('records'),
    columns=[{
      'name': c,
      'id': c
    } for c in df.columns.values],
    fixed_rows={'headers': True},
    #fixed_columns={'headers': True, 'data': 1},
    style_table={'height': '100%', 'width': '100%', 'overflow': 'auto'}
  )