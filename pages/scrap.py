import camelot
from dash import callback, dcc, html, no_update, register_page, Input, Output, State

from components.ticker_select import TickerSelectAIO

from lib.morningstar.ticker import Ticker

register_page(__name__, path='/scrap')

main_style = 'grid grid-cols-[1fr_2fr_2fr] h-full'
form_style = 'flex'
layout = html.Main(className=main_style, children=[
  html.Aside(children=[
    TickerSelectAIO(aio_id='scrap'),
    dcc.Dropdown(id='dropdown:scrap:pdf'),
    html.Form(className=form_style, children=[
      dcc.Input(
        id='input:scrap:pages',  
        placeholder='Pages', 
        type='text'
      ),
      html.Button('Extract', id='button:scrap:extract', n_clicks=0)
    ])
  ]
  ),
  html.Div(id='div:scrap:pdf', className='h-full w-full'),
  html.Div(id='div:scrap:table')
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

@callback(
  Output('div:scrap:table', 'children'),
  Input('button:scrap:extract', 'n_clicks'),
  State('dropdown:scrap:pdf', 'value'),
  State('input:scrap:pages', 'value'),
)
def update_table(n_clicks: int, pdf: str, pages: str):
  if not (n_clicks and pdf and pages):
    return no_update
  
  tables = camelot.read_pdf(pdf, pages=pages)