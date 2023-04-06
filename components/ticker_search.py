from dash import callback, dcc, html, no_update, Input, Output

from lib.db import search_tickers

input_style = 'peer min-w-[20vw] h-full p-1 rounded border border-text/10 hover:border-text/50 focus:border-secondary'
link_style = 'block text-text hover:text-secondary'
nav_style = 'hidden peer-focus-within:flex flex-col gap-1 absolute top-full left-0 p-1 bg-primary/50 backdrop-blur-sm shadow'

def TickerSearch():
  return html.Div(className='relative h-full', children=[
    dcc.Input(
      id='ticker-search', 
      className=input_style, 
      placeholder='Ticker', 
      type='text'
    ),
    html.Nav(
      id='ticker-search-result',
      className=nav_style
    )
  ])

@callback(
  Output('ticker-search-result', 'children'),
  Input('ticker-search', 'value'),
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