from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import io
import re
from pathlib import Path
import sqlite3

from dash import (
  ALL,
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
)
import dash_ag_grid as dag
import httpx
from img2table.document import PDF
from img2table.ocr import TesseractOCR
import pandas as pd

from components.ticker_select import TickerSelectAIO
from components.input import InputAIO
from components.modal import ModalAIO

from lib.db.lite import sqlite_path
from lib.const import HEADERS
from lib.edgar.models import (
  RawFinancials,
  Item,
  Instant,
  Interval,
  Scope,
  FiscalPeriod,
)
from lib.morningstar.ticker import Stock
from lib.utils import download_file

register_page(__name__, path='/scrap')


def get_doc_id(url: str) -> str:
  pattern = r'(?<=/)[a-z0-9]+(?=\.msdoc)'
  match = re.search(pattern, url)

  if not match:
    return ''

  return match.group()


def upsert(db_name: str, table: str, records: list[RawFinancials]):
  db_path = sqlite_path(db_name)

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  cur.execute(
    f"""CREATE TABLE IF NOT EXISTS 
    "{table}"(
      date TEXT, 
      scope TEXT, 
      period TEXT, 
      currency TEXT, 
      data TEXT
  )"""
  )
  cur.execute(
    f"""CREATE UNIQUE INDEX IF NOT EXISTS ix 
    ON "{table}" (date, scope, period)"""
  )

  query = f"""INSERT INTO "{table}" VALUES (:date, :scope, :period, :currency, :data)
      ON CONFLICT (date, scope, period) DO UPDATE SET
        currency=(
          SELECT json_group_array(value)
          FROM (
            SELECT json_each.value
            FROM json_each(currency)
            WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded.currency))
          )
        )
        data=json_patch(data, excluded.data)
  """
  cur.executemany(query, [r.model_dump() for r in records])

  con.commit()
  con.close()


main_style = 'grid grid-cols-[1fr_2fr_2fr] h-full bg-primary'
input_style = 'p-1 rounded-l border-l border-t border-b border-text/10'
button_style = 'px-2 rounded bg-secondary/50 text-text'
group_button_style = 'px-2 rounded-r bg-secondary/50 text-text'
layout = html.Main(
  className=main_style,
  children=[
    html.Aside(
      className='flex flex-col gap-2 p-2',
      children=[
        TickerSelectAIO(aio_id='scrap'),
        dcc.Dropdown(id='dropdown:scrap:document', placeholder='Document'),
        html.Form(
          className='flex',
          action='',
          children=[
            dcc.Input(
              id='input:scrap:pages',
              className=input_style,
              placeholder='Pages',
              type='text',
            ),
            html.Button(
              'Extract',
              id='button:scrap:extract',
              className=group_button_style,
              type='button',
              n_clicks=0,
            ),
          ],
        ),
        dcc.Checklist(
          id='checklist:scrap:options',
          className='flex gap-4',
          labelClassName='gap-1 text-text',
          labelStyle={'display': 'flex'},
          options=[
            {'label': 'Borderless', 'value': 'borderless'},
            {'label': 'Implicit rows', 'value': 'implicit'},
          ],
          value=[],
        ),
        InputAIO(
          'scrap:factor',
          '100%',
          {'value': 1e6, 'placeholder': 'Factor', 'type': 'number'},
        ),
        InputAIO(
          'scrap:currency',
          '100%',
          {'value': 'NOK', 'placeholder': 'Currency', 'type': 'text'},
        ),
        html.Button(
          'Delete rows',
          id='button:scrap:delete',
          className=button_style,
          type='button',
          n_clicks=0,
        ),
        html.Button(
          'Rename headers',
          id=ModalAIO.open_id('scrap:headers'),
          className=button_style,
          type='button',
          n_clicks=0,
        ),
        html.Button(
          'Export to JSON',
          id='button:scrap:export',
          className=button_style,
          type='button',
          n_clicks=0,
        ),
        InputAIO('scrap:id', '100%', {'type': 'text', 'placeholder': 'Ticker ID'}),
        InputAIO('scrap:date', '100%', {'type': 'text', 'placeholder': 'Date'}),
        dcc.Dropdown(
          id='dropdown:scrap:scope',
          placeholder='Scope',
          options=[
            {'label': 'Annual', 'value': 'annual'},
            {'label': 'Quarterly', 'value': 'quarterly'},
          ],
        ),
        dcc.Dropdown(
          id='dropdown:scrap:period',
          placeholder='Period',
          options=['FY', 'Q1', 'Q2', 'Q3', 'Q4'],
        ),
      ],
    ),
    html.Div(id='div:scrap:pdf', className='h-full w-full'),
    html.Div(className='h-full', id='div:scrap:table'),
    ModalAIO(
      'scrap:headers',
      'Rename headers',
      children=[
        html.Div(
          className='flex flex-col',
          children=[
            html.Form(id='form:scrap:headers', className='flex flex-col gap-1'),
            html.Button(
              'Update', id='button:scrap:headers:update', className=button_style
            ),
          ],
        )
      ],
    ),
  ],
)


@callback(
  Output('dropdown:scrap:document', 'options'),
  Input(TickerSelectAIO._id('scrap'), 'value'),
)
def update_dropdown(ticker: str):
  if not ticker:
    return no_update

  docs = Stock(ticker.split('|')[0]).documents()
  docs.rename(columns={'link': 'value'}, inplace=True)
  docs['label'] = docs['date'] + ' - ' + docs['type'] + ' (' + docs['language'] + ')'

  return docs[['label', 'value']].to_dict('records')


@callback(
  Output('div:scrap:pdf', 'children'), Input('dropdown:scrap:document', 'value')
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
  State('dropdown:scrap:document', 'value'),
  State('input:scrap:pages', 'value'),
  State('checklist:scrap:options', 'value'),
  State(InputAIO._id('scrap:factor'), 'value'),
  State(InputAIO._id('scrap:currency'), 'value'),
)
def update_table(
  n_clicks: int,
  pdf_url: str,
  pages_text: str,
  options: list[str],
  factor: int,
  currency: str,
):
  if not (n_clicks and pdf_url and pages_text):
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

  pages = [int(p) - 1 for p in pages_text.split(',')]
  pdf = PDF(src=pdf_src, pages=pages)

  ocr = TesseractOCR(lang='eng')

  tables = pdf.extract_tables(
    ocr=ocr,
    borderless_tables=True if 'borderless' in options else False,
    implicit_rows=True if 'implicit' in options else False,
  )

  # TODO: merge tables
  df = tables[pages[0]][0].df
  df['period'] = 'instant'
  df['factor'] = factor
  df['unit'] = currency
  diff = ['period', 'factor', 'unit']
  df = df[diff + list(df.columns.difference(diff))]

  columnDefs: list[dict[str, str | bool | dict]] = [
    {'field': str(c)} for c in df.columns
  ]
  columnDefs[0].update(
    {
      'checkboxSelection': True,
      'cellEditor': 'agSelectCellEditor',
      'cellEditorParams': {'values': ['instant', 'duration']},
    }
  )
  columnDefs[1].update({'type': 'numericColumn'})

  return dag.AgGrid(
    id='table:scrap',
    columnDefs=columnDefs,
    rowData=df.to_dict('records'),
    columnSize='autoSize',
    defaultColDef={'editable': True},
    dashGridOptions={'undoRedoCellEditing': True, 'undoRedoCellEditingLimit': 10},
    style={'height': '100%'},
  )


@callback(
  Output('table:scrap', 'deleteSelectedRows'),
  Input('button:scrap:delete', 'n_clicks'),
  prevent_initial_call=True,
)
def selected(_: int):
  return True


@callback(
  Output('form:scrap:headers', 'children'),
  Input('table:scrap', 'columnDefs'),
  prevent_initial_call=True,
)
def update_form(cols: list[dict]):
  return [
    dcc.Input(
      id={'type': 'input:scrap:headers', 'index': i},
      className='px-1 rounded border border-text/10 hover:border-text/50 focus:border-secondary',
      placeholder=f'Field {i}',
      value=col['field'],
      type='text',
    )
    for (i, col) in enumerate(cols[3:])
  ]


@callback(
  Output('table:scrap', 'columnDefs'),
  Output('table:scrap', 'rowData'),
  Input('button:scrap:headers:update', 'n_clicks'),
  State({'type': 'input:scrap:headers', 'index': ALL}, 'value'),
  State('table:scrap', 'columnDefs'),
  State('table:scrap', 'rowData'),
  prevent_initial_call=True,
)
def toggle_cols(n: int, new_names: list[str], cols: list[dict], rows: list[dict]):
  if not n:
    return no_update

  df = pd.DataFrame.from_records(rows)
  df = df[[col['field'] for col in cols]]

  col_map = {col: name for (col, name) in zip(df.columns[3:], new_names)}
  df.rename(columns=col_map, inplace=True)

  for i, name in enumerate(new_names):
    cols[i + 3]['field'] = name

  return cols, df.to_dict('records')


@callback(
  Output(InputAIO._id('scrap:id'), 'value'),
  Input(TickerSelectAIO._id('scrap'), 'value'),
)
def update_input(ticker: str):
  if not ticker:
    return no_update

  return ticker.split('|')[0]


@callback(
  Output(InputAIO._id('scrap:date'), 'value'),
  Input('dropdown:scrap:document', 'value'),
  State('dropdown:scrap:document', 'options'),
)
def update_document_dropdown(doc: str, options: list[dict[str, str]]):
  if not doc:
    return no_update

  label = [x['label'] for x in options if x['value'] == doc][0]

  pattern = r'\d{4}-\d{2}-\d{2}'
  match = re.search(pattern, label)

  if not match:
    return ''

  return match.group()


@callback(
  Output('dropdown:scrap:scope', 'value'),
  Output('dropdown:scrap:period', 'value'),
  Input('dropdown:scrap:document', 'value'),
  State('dropdown:scrap:document', 'options'),
)
def update_scope_dropdown(doc: str, options: list[dict[str, str]]):
  if not doc:
    return no_update

  label = [x['label'] for x in options if x['value'] == doc][0]

  pattern = r'(annual|quarterly)'
  match = re.search(pattern, label, flags=re.I)

  if not match:
    return '', ''

  scope = match.group().lower()
  period = 'FY' if scope == 'annual' else ''

  return (scope, period)


@callback(
  Output('button:scrap:export', 'id'),
  Input('button:scrap:export', 'n_clicks'),
  State('table:scrap', 'rowData'),
  State(InputAIO._id('scrap:id'), 'value'),
  State(InputAIO._id('scrap:date'), 'value'),
  State('dropdown:scrap:scope', 'value'),
  State('dropdown:scrap:period', 'value'),
  prevent_initial_call=True,
)
def export(
  n: int, rows: list[dict], _id: str, date: str, scope: Scope, period: FiscalPeriod
):
  def parse_period(scope: Scope, date_text: str, row: pd.Series) -> Instant | Interval:
    date = dt.strptime(date_text, '%Y-%m-%d').date()

    if row['period'] == 'instant':
      return Instant(instant=date)

    start_date = dt.strptime(period, '%Y-%m-%d').date()

    if scope == 'annual':
      start_date -= relativedelta(years=1)
      months = 12

    elif scope == 'quarterly':
      start_date -= relativedelta(months=3)
      months = 3

    return Interval(start_date=start_date, end_date=date, months=months)

  if not n:
    return no_update

  data: dict[str, list[Item]] = {}

  df = pd.DataFrame.from_records(rows)
  if 'item' not in set(df.columns):
    return 'button:scrap:export'

  dates = list(df.columns.difference(['period', 'factor', 'unit', 'item']))
  df.loc[:, dates] = df[dates].replace(r'[^\d.]', '', regex=True).astype(float)

  currencies = set()

  for _, r in df.iterrows():
    if r['unit'] != 'shares':
      currencies.add(r['unit'])

    data[r['item']] = [
      Item(
        value=r[d] * float(r['factor']),
        unit=r['unit'],
        period=parse_period(scope, d, r),
      )
      for d in dates
    ]

  records = [
    RawFinancials(
      date=dt.strptime(date, '%Y-%m-%d').date(),
      scope=scope,
      period=period,
      currency=currencies,
      data=data,
    )
  ]

  upsert('data/financials_raw.db', _id, records)

  # with open(f'{_id}.json', 'w') as f:
  #  json.dump(record, f, indent=2)

  return 'button:scrap:export'
