from dash import callback, html, no_update, register_page, Input, Output

from components.map import choropleth_map
from lib.db import insert_sqlite, DB_DIR
from lib.morningstar import get_tickers
from lib.sec_edgar import get_ciks
from lib.virdi import real_estate_price_data

register_page(__name__, path='/board')

main_style = 'h-full grid grid-cols-[1fr_1fr] gap-2 p-2'
layout = html.Main(className=main_style, children=[
  html.Div(id='board-div:ticker', className='flex flex-col', children=[
    html.H4('Update tickers'),
    html.Button('Stocks', id='board-button:stock', n_clicks=0),
    html.Button('SEC Edgar', id='board-button:cik', n_clicks=0)
  ]),
  html.Div(id='board-div:real-estate', className='flex flex-col', children=[
    html.H4('Update real estate data'),
    html.Button('Hjemla', id='board-button:hjemla', n_clicks=0),
  ])
])

@callback(
  Output('board-div:ticker', 'className'),
  Input('board-button:stock', 'n_clicks')
)
def update_tickers(n_clicks):
  if not n_clicks:
    return no_update
  
  df = get_tickers()
  insert_sqlite(df, 'ticker.db', 'stock', 'overwrite')
  return no_update

@callback(
  Output('board-div:ticker', 'className'),
  Input('board-button:cik', 'n_clicks')
)
def update_ciks(n_clicks):
  if not n_clicks:
    return no_update
  
  df = get_ciks()
  insert_sqlite(df, 'ticker.db', 'edgar', 'overwrite')
  return no_update

@callback(
  Output('board-div:real-estate', 'className'),
  Input('board-button:hjemla', 'n_clicks')
)
def update_ciks(n_clicks):
  if not n_clicks:
    return no_update
  
  gdf = get_real_estate_data()
  path = DB_DIR / 'hjemla.json'
  gdf.to_file(path, driver='GeoJSON', encoding='utf-8')

  choropleth_map()
    
  return no_update