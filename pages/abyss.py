from dash import callback, html, no_update, register_page, Input, Output

from lib.db import insert_sqlite
from lib.sec_edgar import get_ciks

register_page(__name__, path='/abyss')

main_style = 'h-full grid grid-cols-[1fr_1fr] gap-2 p-2'
layout = html.Main(className=main_style, children=[
  html.Div(id='abyss-div:ticker', className='flex flex-col', children=[
    html.H4('Update tickers'),
    html.Button('SEC Edgar', id='abyss-button:cik', n_clicks=0)
  ])
])

@callback(
  Output('abyss-div:ticker', 'className'),
  Input('abyss-button:cik', 'n_clicks')
)
def update_ciks(n_clicks):
  if not n_clicks:
    return no_update
  
  df = get_ciks()
  insert_sqlite(df, 'ticker.db', 'edgar', 'overwrite')
  return no_update