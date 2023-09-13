import asyncio
from typing import Optional
from ordered_set import OrderedSet

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input, State
)
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Sign
import pandas as pd

from components.sparklines import make_sparkline
from lib.edgar.company import Company
from lib.fin.utils import Taxonomy, calculate_items, load_template, merge_labels
from lib.ticker.fetch import stock_label, find_cik
#from lib.utils import load_json

register_page(__name__, path_template='/stock/<id>/financials', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = 'appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50'
radio_label_style = 'relative px-1'

def style_table(index: pd.Series, tmpl: pd.DataFrame) -> list[dict]:

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
  ]

  for level in tmpl['level'].unique():
    items = tmpl.loc[tmpl['level'] == level, 'short']
    row_ix = [index[index == i].index[0] for i in items] 

    styling.append({
      'if': {
        'row_index': row_ix,
        'column_id': 'index',
      },
      'paddingLeft': f'{level + 1}rem',
      'fontWeight': 'bold' if level == 0 else 'normal',
      'borderBottom': '1px solid rgb(var(--color-text))' if level == 0 else None
    })

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
    template = load_template('table')
    taxonomy = Taxonomy(set(template['item']))
    template = merge_labels(template, taxonomy)

    financials = asyncio.run(Company(cik).financials_to_df('%Y-%m-%d', taxonomy, True))
    schema = taxonomy.calculation_schema(set(template['item']))
    financials = calculate_items(financials, schema)

    financials = financials.reset_index().to_dict('records')
    template = template.to_dict('records')

  return html.Main(className='flex flex-col h-full', children=[
    html.Div(className='flex justify-around', children=[
      dcc.RadioItems(id='radio:stock-financials:sheet', className=radio_wrap_style,
        inputClassName=radio_input_style,
        labelClassName=radio_label_style,
        value='income',
        options=[
          {'label': 'Income', 'value': 'income'},
          {'label': 'Balance', 'value': 'balance'},
          {'label': 'Cash Flow', 'value': 'cashflow'}
        ]),
      dcc.RadioItems(id='radio:stock-financials:scope', className=radio_wrap_style,
        inputClassName=radio_input_style,
        labelClassName=radio_label_style,
        value='a',
        options=[
          {'label': 'Annual', 'value': 'a'},
          {'label': 'Quarterly', 'value': 'q'},
        ]),
    ]),
    html.Div(id='div:stock-financials:table-wrap', className='flex-1 overflow-x-hidden p-2'),
    dcc.Store(id='store:stock-financials:financials', data=financials),
    dcc.Store(id='store:stock-financials:template', data=template)
  ])

@callback(
  Output('div:stock-financials:table-wrap', 'children'),
  Input('store:stock-financials:financials', 'data'),
  Input('radio:stock-financials:sheet', 'value'),
  Input('radio:stock-financials:scope', 'value'),
  State('store:stock-financials:template', 'data'),
)
def update_table(fin: list[dict], tmpl: list[dict], sheet: str, scope: str):
  if not fin or not tmpl:
    return no_update
  
  tmpl = pd.DataFrame.from_records(tmpl)
  tmpl = tmpl.loc[tmpl['sheet'] == sheet]

  labels = pd.Series(tmpl['short'].values, index=tmpl['item']).to_dict()

  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'period'])
    .xs(scope, level=1) 
    .sort_index(ascending=False) 
  )
  cols = list(OrderedSet(OrderedSet(tmpl['item']).intersection(fin.columns)))
  fin = fin[cols]
  fin.rename(columns=labels, inplace=True)
  tmpl = tmpl.loc[tmpl['item'].isin(cols)]

  fin = fin.T.reset_index()
  fin.insert(1, 'Trend', make_sparkline(fin[fin.columns[1:]]))

  tooltips = [{
    'index': {
      'type': 'markdown',
      'value': long
    }
  } for long in tmpl['long']]

  return DataTable(
    fin.to_dict('records'),
    columns=format_columns(fin.columns, fin.columns[0]),
    style_header={
      'fontWeight': 'bold'
    },
    style_data_conditional=style_table(fin['index'], tmpl),
    fixed_columns={'headers': True, 'data': 1},
    fixed_rows={'headers': True},
    tooltip_data=tooltips
  )