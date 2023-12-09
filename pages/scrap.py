from dash import callback, dcc, html, no_update, register_page, Input, Output, State
import dash_ag_grid as dag
import httpx
from img2table.document import PDF
from img2table.ocr import TesseractOCR

from components.ticker_select import TickerSelectAIO

from lib.morningstar.ticker import Ticker
from lib.utils import download_file

register_page(__name__, path='/scrap')

main_style = 'grid grid-cols-[1fr_2fr_2fr] h-full'
input_style = 'p-1 rounded-l border-l border-t border-b border-text/10'
button_style = 'px-2 rounded-r bg-secondary'
layout = html.Main(className=main_style, children=[
  html.Aside(className='flex flex-col gap-2 p-2', children=[
    TickerSelectAIO(aio_id='scrap'),
    dcc.Dropdown(id='dropdown:scrap:pdf', placeholder='Document'),
    html.Form(className='flex', action='', children=[
      dcc.Input(
        id='input:scrap:pages',
        className=input_style,
        placeholder='Pages',
        type='text'),
      html.Button('Extract', 
        id='button:scrap:extract', 
        className=button_style, 
        type='button',
        n_clicks=0)
    ]),
    dcc.Checklist(
      id='checklist:scrap:options',
      options=[
        {'label': 'Borderless', 'value': 'borderless'},
        {'label': 'Implicit rows', 'value': 'implicit'}], 
      value=[])
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
  docs['label'] = docs['date'] + ' - ' + docs['type'] + ' (' + docs['language'] + ')'
  
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
  State('checklist:scrap:options', 'value')
)
def update_table(n_clicks: int, pdf_url: str, pages: str, options: list[str]):
  if not (n_clicks and pdf_url and pages):
    return no_update
  
  pdf_path = download_file(pdf_url, '.pdf')

  pages = [int(p) for p in pages.split(',')]
  pdf = PDF(src=pdf_path, pages=pages)
  
  ocr = TesseractOCR(lang='eng')

  tables = pdf.extract_tables(
    ocr=ocr, 
    borderless_tables=True if 'borderless' in options else False,
    implicit_rows=True if 'implicit' in options else False)
  
  print(tables[0].df)
  
  return []
  #columnDefs = [{'field': c} for c in tables[0].df.columns]

  #return dag.AgGrid(
  #  id='table:stock-financials',
  #  columnDefs=columnDefs,
  #  rowData=tables[0].df.to_dict('records'),
  #  columnSize='autoSize',
  #  style={'height': '100%'},
  #)