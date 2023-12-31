import io
import re
from pathlib import Path

from dash import callback, dcc, html, no_update, register_page, Input, Output, State
import dash_ag_grid as dag
import httpx
from img2table.document import PDF
from img2table.ocr import TesseractOCR

from components.ticker_select import TickerSelectAIO

from lib.const import HEADERS
from lib.morningstar.ticker import Ticker
from lib.utils import download_file

register_page(__name__, path='/scrap')

def get_doc_id(url: str) -> str:

  pattern = r'(?<=/)[a-z0-9]+(?=\.msdoc)'
  match = re.search(pattern, url)

  if not match:
    return ''

  return match.group()

main_style = 'grid grid-cols-[1fr_2fr_2fr] h-full'
input_style = 'p-1 rounded-l border-l border-t border-b border-text/10'
button_style = 'px-2 rounded-r bg-secondary/50'
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
  html.Div(className='flex flex-col', children=[
    html.Form(action='', children=[
      html.Button('Delete rows', id='button:scrap:delete', type='button', n_clicks=0),
      html.Button('Set as header', id='button:scrap:header', type='button', n_clicks=0)
    ]),
    html.Div(className='h-full', id='div:scrap:table')]
  )
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
  
  doc_id = get_doc_id(url)
  if not doc_id:
    return []
  
  pdf_path = Path(f'temp/{doc_id}.pdf')
  if not pdf_path.exists():
    download_file(url, pdf_path)
  
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
  
  #doc_id = get_doc_id(pdf_url)
  #pdf_path = Path(f'temp/{doc_id}.pdf')
  temp = io.BytesIO()
  response = httpx.get(url=pdf_url, headers=HEADERS)
  temp.write(response.content)

  pages = [int(p) - 1 for p in pages.split(',')]
  pdf = PDF(src=temp, pages=pages)
  
  ocr = TesseractOCR(lang='eng')

  tables = pdf.extract_tables(
    ocr=ocr, 
    borderless_tables=True if 'borderless' in options else False,
    implicit_rows=True if 'implicit' in options else False)
  
  #print(tables[pages[0]][0].df)
  
  #return []
  columnDefs = [{'field': str(c)} for c in tables[pages[0]][0].df.columns]
  columnDefs[0].update({'checkboxSelection': True, 'headerCheckboxSelection': True})

  return dag.AgGrid(
    id='table:scrap',
    columnDefs=columnDefs,
    rowData=tables[pages[0]][0].df.to_dict('records'),
    columnSize='autoSize',
    defaultColDef = {'editable': True},
    style={'height': '100%'},
  )

@callback(
  Output('table:scrap', 'deleteSelectedRows'),
  Input('button:scrap:delete', 'n_clicks'),
  prevent_initial_call=True
)
def selected(_):
  return True