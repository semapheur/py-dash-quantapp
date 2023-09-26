import asyncio
from functools import partial
import json

from dash import callback, dcc, html, no_update, Input, Output, State
import pandas as pd
from sqlalchemy import create_engine, text

from lib.edgar.company import Company
from lib.fin.taxonomy import Taxonomy
from lib.fin.calculation import calculate_items
from lib.morningstar.ticker import Ticker
from lib.ticker.fetch import find_cik, get_ohlcv, search_tickers

input_style = (
  'peer min-w-[20vw] h-full p-1 bg-primary text-text '
  'rounded border border-text/10 hover:border-text/50 focus:border-secondary '
  'placeholder-transparent'
)

label_style = (
  'absolute left-1 -top-2 px-1 bg-primary text-text/500 text-xs '
  'peer-placeholder-shown:text-base peer-placeholder-shown:text-text/50 '
  'peer-placeholder-shown:top-1 peer-focus:-top-2 peer-focus:text-secondary '
  'peer-focus:text-xs transition-all'
)
link_style = 'block text-text hover:text-secondary'
nav_style = (
  'hidden peer-focus-within:flex hover:flex flex-col gap-1 '
  'absolute top-full left-1 p-1 bg-primary/50 backdrop-blur-sm shadow z-[1]'
)
def TickerSearch():
  return html.Div(className='relative h-full', children=[
    dcc.Input(
      id='input:ticker-search', 
      className=input_style, 
      placeholder='Ticker', 
      type='text'
    ),
    html.Label(
      htmlFor='input:ticker-search', 
      className=label_style, 
      children=['Ticker']
    ),
    html.Nav(
      id='nav:ticker-search',
      className=nav_style
    ),
    dcc.Store(id='store:ticker-search:financials'),
    dcc.Store(id='store:ticker-search:id')
  ])

@callback(
  Output('nav:ticker-search', 'children'),
  Input('input:ticker-search', 'value'),
)
def ticker_results(search: str) -> list[dict[str, str]]:
  if search is None or len(search) < 2:
    return no_update

  df = search_tickers('stock', search)
  links = [
    dcc.Link(label, href=href, className=link_style) 
    for label, href in zip(df['label'], df['href'])
  ]
  return links

@callback(
  Output('store:ticker-search:id', 'data'),
  Input('location:app', 'pathname'),
  State('store:ticker-search:id', 'data')
)
def update_store(path: str, id_store: dict[str,str]):
  path_split = path.split('/')

  if path_split[1] != 'stock':
    return no_update
  
  new_id = path_split[2]
  old_id = id_store.get('id', '')

  if old_id['id'] == new_id:
    return no_update

  return {'id': new_id}

@callback(
  Output('store:ticker-search:financials', 'data'),
  Input('store:ticker-search:id', 'data'),
  State('location:app', 'pathname')
)
def update_store(id_store: dict[str, str], path_name: str):
  
  id = id_store.get('id')
  if id is None:
    return no_update

  cik = find_cik(id)

  if cik is None:
    return no_update
  
  taxonomy = Taxonomy()
  financials = asyncio.run(Company(cik).financials_to_df(taxonomy))

  db_path = 'data/taxonomy.db'
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  query = text('''
    SELECT template.item, items.calculation FROM (
      SELECT item FROM statement UNION
      SELECT item FROM sankey UNION
      SELECT item FROM dupont
    ) AS template
    LEFT JOIN items ON template.item = items.item
    WHERE items.calculation IS NOT NULL
  ''')

  with engine.connect().execution_options(autocommit=True) as con:
    template = pd.read_sql(query, con=con)

  template = template.loc[template['calculation'] != 'null']
  template.loc[:,'calculation'] = (
    template['calculation'].apply(lambda x: json.loads(x))
  )
  security = path_name.split('/')[0]

  start_date = financials.index.get_level_values('date').min()
  fetcher = partial(
    Ticker(id, security, 'USD').ohlcv, 
    start_date.strftime('%Y-%m-%d')
  )

  price = get_ohlcv(id, security, fetcher, cols=['date', 'close'])
  price.rename(columns={'close': 'price'}, inplace=True)
  price = price.resample('D').ffill()

  financials = financials.merge(price, how='left', on='date')

  schema = dict(zip(template['item'], template['calculation']))
  financials = calculate_items(financials, schema)
 
  financials.index = financials.index.set_levels(
    financials.index.levels[0].strftime('%Y-%m-%d'),
    level='date'
  )
  financials.reset_index(inplace=True)
  
  return financials.to_dict('records')