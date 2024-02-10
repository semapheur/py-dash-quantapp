import asyncio
from functools import partial

from dash import callback, dcc, html, no_update, Input, Output, State

from lib.edgar.company import Company
from lib.morningstar.ticker import Stock
from lib.ticker.fetch import find_cik, get_fundamentals, search_tickers

from components.input import InputAIO

link_style = 'block text-text hover:text-secondary'
nav_style = (
  'hidden peer-focus-within:flex hover:flex flex-col gap-1 '
  'absolute top-full left-1 p-1 bg-primary/50 backdrop-blur-sm shadow z-[1]'
)


def TickerSearch():
  return html.Div(
    className='relative h-full',
    children=[
      InputAIO('ticker-search', '20vw', {'placeholder': 'Ticker', 'type': 'text'}),
      html.Nav(id='nav:ticker-search', className=nav_style),
      dcc.Store(id='store:ticker-search:financials'),
      dcc.Store(id='store:ticker-search:id', data={}),
    ],
  )


@callback(
  Output('nav:ticker-search', 'children'),
  Input(InputAIO._id('ticker-search'), 'value'),
)
def ticker_results(search: str) -> list[dict[str, str]]:
  if search is None or len(search) < 2:
    return no_update

  df = search_tickers('stock', search)
  links = [
    dcc.Link(label, href=href + '/overview', className=link_style)
    for label, href in zip(df['label'], df['href'])
  ]
  return links


@callback(
  Output('store:ticker-search:id', 'data'),
  Input('location:app', 'pathname'),
  State('store:ticker-search:id', 'data'),
)
def id_store(path: str, id_store: dict[str, str]):
  path_split = path.split('/')

  if path_split[1] != 'stock':
    return no_update

  new_id = path_split[2]
  old_id = id_store.get('id', '')

  if old_id == new_id:
    return no_update

  return {'id': new_id}


@callback(
  Output('store:ticker-search:financials', 'data'),
  Input('store:ticker-search:id', 'data'),
)
def fincials_store(id_store: dict[str, str]):
  _id = id_store.get('id')
  if _id is None:
    return no_update

  cik = find_cik(_id)

  if cik is None:
    return no_update

  financials_fetcher = partial(Company(cik).financials_to_df)
  ohlcv_fetcher = partial(Ticker(_id, 'stock', 'USD').ohlcv)

  fundamentals = asyncio.run(get_fundamentals(_id, financials_fetcher, ohlcv_fetcher))

  fundamentals.index = fundamentals.index.set_levels(
    fundamentals.index.levels[0].strftime('%Y-%m-%d'), level='date'
  )
  fundamentals.reset_index(inplace=True)

  return fundamentals.to_dict('records')
