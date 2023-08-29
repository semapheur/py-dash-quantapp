import asyncio
from typing import Optional
from ordered_set import OrderedSet

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input
)
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Sign
import pandas as pd

from components.sparklines import make_sparkline
from lib.edgar.ticker import Ticker
from lib.fin.utils import Taxonomy, load_taxonomy, load_template
from lib.ticker.fetch import stock_label, find_cik
#from lib.utils import load_json

register_page(__name__, path_template='/stock/<id>', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = 'appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50'
radio_label_style = 'relative px-1'

def style_table(index: pd.Series, tmpl: pd.DataFrame) -> list[dict]:

  emph = tmpl.loc[tmpl['level'] == 0, 'long']
  emph_ix = [index[index == e].index[0] for e in emph]

  tab = tmpl.loc[tmpl['level'] == 2, 'long']
  tab_ix = [index[index == t].index[0] for t in tab]

  ttab = tmpl.loc[tmpl['level'] == 3, 'long']
  ttab_ix = [index[index == t].index[0] for t in ttab]

  styling: list[dict] = [
    {
      'if': {
        'column_id': 'Trend'
      },
      'font_family': 'Sparks-Dotline-Extrathick'
    },
    {
      'if': {
        'column_id': 'index'
      },
      'textAlign': 'left',
      'paddingLeft': '2rem'
    },
    {
      'if': {
        'row_index': emph_ix,
        'column_id': 'index',
      },
      'fontWeight': 'bold',
      'paddingLeft': '1rem',
    },
    {
      'if': {
        'row_index': emph_ix,
      },
      'borderBottom': '1px solid rgb(var(--color-text))'
    },
    {
      'if': {
        'row_index': tab_ix,
        'column_id': 'index'
      },
      'paddingLeft': '3rem'
    },
    {
      'if': {
        'row_index': ttab_ix,
        'column_id': 'index'
      },
      'paddingLeft': '4rem'
    }
  ]
  return styling

def format_columns(columns: list[str], index: str) -> dict:
  return [{
    'name': c,
    'id': c,
    'type': 'text' if c == index else 'numeric',
    'format': None if c == index else Format(
      group=True,
      sign=Sign.parantheses
    )
  } for c in columns]

def layout(id: Optional[str] = None):

  cik = find_cik(id)

  financials = template = {}
  if cik:
    template = load_template()
    taxonomy = Taxonomy(set(template['item']))
    template = template.merge(taxonomy.labels(), on='item', how='left')

    financials = asyncio.run(Ticker(cik).financials('%Y-%m-%d', taxonomy))

    financials = financials.reset_index().to_dict('records')
    template = template.to_dict('records')

  return html.Main(className='flex flex-col h-full', children=[
    dcc.RadioItems(id='stock-radio:sheet', className=radio_wrap_style,
      inputClassName=radio_input_style,
      labelClassName=radio_label_style,
      value='income',
      options=[
        {'label': 'Income', 'value': 'income'},
        {'label': 'Balance', 'value': 'balance'},
        {'label': 'Cash Flow', 'value': 'cash'}
      ]),
    html.Div(id='stock-div:table-wrap', className='h-full p-2'),
    dcc.Store(id='stock-store:financials', data=financials),
    dcc.Store(id='stock-store:template', data=template)
  ])

@callback(
  Output('stock-div:table-wrap', 'children'),
  Input('stock-store:financials', 'data'),
  Input('stock-store:template', 'data'),
  Input('stock-radio:sheet', 'value')
  #Input('stock-scope:annual/quarterly')
)
def update_table(fin: list[dict], tmpl: list[dict], sheet: str):
  if not fin or not tmpl:
    return no_update
  
  tmpl = pd.DataFrame.from_records(tmpl)
  tmpl = tmpl[tmpl['sheet'] == sheet]
  labels = pd.Series(tmpl['long'].values, index=tmpl['item']).to_dict()

  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'period'])
    .xs('a', level=1) 
    .sort_index(ascending=False) 
  )
  cols = list(OrderedSet(fin.columns).intersection(OrderedSet(tmpl['item'])))
  fin = fin[cols]
  fin.rename(columns=labels, inplace=True)

  fin = fin.T.reset_index()
  fin.insert(1, 'Trend', make_sparkline(fin[fin.columns[1:]]))

  return DataTable(
    fin.to_dict('records'),
    columns=format_columns(fin.columns, fin.columns[0]),
    style_header={
      'fontWeight': 'bold'
    },
    style_data_conditional=style_table(fin['index'], tmpl),
    fixed_columns={'headers': True, 'data': 1},
    fixed_rows={'headers': True},
  )