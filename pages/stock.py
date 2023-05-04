from typing import Optional

import asyncio
from dash import (
  callback, dcc, html, no_update, register_page, dash_table, Output, Input
)
import pandas as pd

from lib.edgar.ticker import Ticker
from lib.fin.utils import load_labels
from lib.ticker.fetch import stock_label, find_cik
from lib.utils import load_json

register_page(__name__, path_template='/stock/<id>', title=stock_label)

def layout(id: Optional[str] = None):

  # Check CIK
  cik = find_cik(id)

  data = {}
  if cik:
    df = asyncio.run(Ticker(cik).financials('%Y-%m-%d'))
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
  
  #template = load_json('fin_template.json')
  labels = load_labels()

  df = (pd.DataFrame.from_records(data) 
    .set_index(['date', 'period'])
    .rename(columns=labels)
    .xs('a', level=1) 
    .sort_index(ascending=False) 
    .T 
    .reset_index()
  )
  return dash_table.DataTable(
    df.to_dict('records'),
    columns=[{
      'name': c,
      'id': c
    } for c in df.columns],
    style_table={
      'height': '100%', 'width': '100%', 
      'overflow': 'auto'
    },
    style_header={
      'fontWeight': 'bold'
    },
    #fixed_columns={'headers': True, 'data': 1},
    fixed_rows={'headers': True},
  )