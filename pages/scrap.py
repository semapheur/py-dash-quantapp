from dash import callback, dcc, html, no_update, register_page, Input, Output

from components.ticker_select import TickerSelectAIO

from lib.morningstar.ticker import Ticker

register_page(__name__, path='/scrap')

main_style = 'grid grid-cols-[1fr_2fr_2fr] h-full'
layout = html.Main(className=main_style, children=[
  html.Aside(children=[
    TickerSelectAIO(aio_id='scrap'),
    dcc.Dropdown(id='dropdown:scrap:pdf')
  ]
  ),
  html.Div(id='div:scrap:pdf', className='h-full w-full')
])

@callback(
  Output('dropdown:scrap:pdf', 'options'),
  Input(TickerSelectAIO._id('scrap'), 'value')
)
def update_dropdown(ticker: str):
  if not ticker:
    return no_update
  
  docs = Ticker(ticker, 'stock').documents()
  docs.rename(columns={'link': 'value'}, inplace=True)
  docs['label'] = docs['date'] + ' - ' + docs['type']
  
  return docs[['label', 'value']].to_dict('records')

@callback(
  Output('div:scrap:pdf', 'children'),
  Input('dropdown:scrap:pdf', 'value')
)
def update_object(url: str):
  if not url:
    return no_update
  
  return html.ObjectEl(data=url, width='100%', height='100%')