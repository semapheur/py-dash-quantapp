import io
import re
from pathlib import Path

from dash import (
  ALL, callback, clientside_callback, ClientsideFunction, 
  dcc, html, no_update, register_page, Input, Output, State, Patch)
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
button_style = 'px-2 rounded bg-secondary/50 text-text'
group_button_style = 'px-2 rounded-r bg-secondary/50 text-text'
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
        className=group_button_style, 
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
    html.Form(action='', className='p-1 flex gap-2', children=[
      html.Button('Delete rows', 
        id='button:scrap:delete', 
        className=button_style,
        type='button', 
        n_clicks=0),
      html.Button('Rename headers', 
        id='button:scrap:headers:open-modal', 
        className=button_style,
        type='button', 
        n_clicks=0)
    ]),
    html.Div(className='h-full', id='div:scrap:table')]
  ),
  html.Dialog(
    id='dialog:scrap:headers', 
    className='m-auto max-h-[75%] max-w-[75%] rounded-md shadow-md dark:shadow-black/50', 
    children=[
      html.Div(className='flex flex-col h-full px-2 pb-2', children=[
        html.Button('X', 
          id='button:scrap:headers:close-modal',
          className='self-end text-secondary hover:text-red-600'
        ),
        html.Div(className='flex flex-col', children=[
          html.H2('Rename headers'),
          html.Form(
            id='form:scrap:headers',
            className='flex flex-col gap-1'
          ),
          html.Button('Update', 
            id='button:scrap:headers:update',
            className=button_style
          ),
        ])
      ])
    ])
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
  
  doc_id = get_doc_id(pdf_url)
  pdf_path = Path(f'temp/{doc_id}.pdf')
  pdf_src = io.BytesIO()
  if pdf_path.exists():
    with open(pdf_path, 'rb') as pdf_file:
      pdf_src.write(pdf_file.read()) 
  else:
    response = httpx.get(url=pdf_url, headers=HEADERS)
    pdf_src.write(response.content)

  pages = [int(p) - 1 for p in pages.split(',')]
  pdf = PDF(src=pdf_src, pages=pages)
  
  ocr = TesseractOCR(lang='eng')

  tables = pdf.extract_tables(
    ocr=ocr, 
    borderless_tables=True if 'borderless' in options else False,
    implicit_rows=True if 'implicit' in options else False)
  
  columnDefs = [{'field': str(c)} for c in tables[pages[0]][0].df.columns]
  columnDefs[0].update({'checkboxSelection': True, 'headerCheckboxSelection': True})

  return dag.AgGrid(
    id='table:scrap',
    columnDefs=columnDefs,
    rowData=tables[pages[0]][0].df.to_dict('records'),
    columnSize='autoSize',
    defaultColDef = {'editable': True},
    dashGridOptions={
      'undoRedoCellEditing': True,
      'undoRedoCellEditingLimit': 10
    },
    style={'height': '100%'},
  )

@callback(
  Output('table:scrap', 'deleteSelectedRows'),
  Input('button:scrap:delete', 'n_clicks'),
  prevent_initial_call=True
)
def selected(_: int):
  return True

@callback(
  Output('table:scrap', 'columnDefs'),
  Input('button:scrap:headers:update', 'n_clicks'),
  State({'type': 'input:scrap:headers', 'index': ALL}, 'value'),
  prevent_initial_call=True
)
def toggle_cols(n: int, new_names: list[str]):
  if not n:
    return no_update

  patched_grid = Patch()

  for (i, name) in enumerate(new_names):
    patched_grid[i]['headerName'] = name

  return patched_grid

@callback(
  Output('form:scrap:headers', 'children'),
  Input('table:scrap', 'columnDefs'),
  prevent_initial_call=True
)
def update_form(cols: list[dict]):
  return [dcc.Input(
    id={'type': 'input:scrap:headers', 'index': i}, 
    className='', 
    placeholder=f'Field {i}',
    value=col['field'],
    type='text'
  ) for (i, col) in enumerate(cols)]

clientside_callback(
  ClientsideFunction(
    namespace='clientside',
    function_name='handle_modal'
  ),
  Output('dialog:scrap:headers', 'id'),
  Input('button:scrap:headers:open-modal', 'n_clicks'),
  Input('button:scrap:headers:close-modal', 'n_clicks'),
  State('dialog:scrap:headers', 'id')
)