from typing import Optional
from ordered_set import OrderedSet

from dash import (
  callback, dcc, html, no_update, register_page, Output, Input
)
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px

from components.stock_header import StockHeader
from lib.db.lite import read_sqlite
from lib.ticker.fetch import stock_label
#from lib.utils import load_json

register_page(__name__, path_template='/stock/<id>/fundamentals', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = (
  'appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50'
)
radio_label_style = 'relative px-1'

def layout(_id: Optional[str] = None):

  return html.Main(className='flex flex-col h-full', children=[
    StockHeader(_id),
    html.Div(className='flex justify-around', children=[
      dcc.RadioItems(
        id='radio:stock-fundamentals:sheet', 
        className=radio_wrap_style,
        inputClassName=radio_input_style,
        labelClassName=radio_label_style,
        value='valuation',
        options=[
          {'label': 'Valuation', 'value': 'valuation'},
        ]),
    ]),
    html.Div(
      id='div:stock-fundamentals:table-wrap', 
      className='flex-1 p-2'),
  ])

@callback(
  Output('div:stock-fundamentals:table-wrap', 'children'),
  Input('store:ticker-search:financials', 'data'),
  Input('radio:stock-fundamentals:sheet')
)
def update_table(data: list[dict], sheet: str):

  if not data:
    return no_update
  
  query = '''SELECT 
    f.item, items.short, items.long FROM fundamentals AS f 
    LEFT JOIN items ON f.item = items.item
    WHERE f.sheet = :sheet
  '''
  param = {'sheet': sheet}
  tmpl = read_sqlite('taxonomy.db', query, param)
  tmpl.loc[:, 'short'].fillna(tmpl['long'], inplace=True)

  fin = (pd.DataFrame.from_records(data)
    .set_index(['date', 'months'])
    .xs(12, level=1) 
    .sort_index(ascending=False) 
  )
  cols = list(OrderedSet(OrderedSet(tmpl['item']).intersection(fin.columns)))
  fin = fin[cols]
  fin = fin.T.reset_index()
  tmpl = (tmpl
    .set_index('item')
    .loc[cols]
    .reset_index()
  )

  fin['trend'] = ''
  for i, r in fin.iterrows():
    fig = px.line(
      r.iloc[1:-1]
    )
    fig.update_layout(
      showlegend=False,
      xaxis=dict(autorange='reversed'),
      xaxis_visible=False,
      xaxis_showticklabels=False,
      yaxis_visible=False,
      yaxis_showticklabels=False,
      margin=dict(l=0, r=0, t=0, b=0),
      template='plotly_white'
    )
    fin.at[i, 'trend'] = fig

  columnDefs = [
    {
      'field': 'index', 'headerName': 'Metric', 
      'pinned': 'left', 'lockPinned': True, 'cellClass': 'lock-pinned',
      'tooltipField': 'index',
      'tooltipComponentParams': {'labels': tmpl['long'].to_list()}
    },
    {
      'field': 'trend', 'headerName': 'Trend',
      'cellRenderer': 'TrendLine'
    }
  ] + [{
    'field': col,
    'type': 'numericColumn',
    'valueFormatter': {'function': 'd3.format("(,")(params.value)'}
  } for col in fin.columns[1:]]
  
  fin.loc[:,'index'] = fin['index'].apply(
    lambda x: tmpl.loc[tmpl['item'] == x, 'short'].iloc[0]
  )

  return dag.AgGrid(
    id='table:stock-fundamentals',
    columnDefs=columnDefs,
    rowData=fin.to_dict('records'),
    columnSize='autoSize',
    defaultColDef={'tooltipComponent': 'FinancialsTooltip'},
    style={'height': '100%'}
    #defaultColDef={'resizable': True, 'sortable': True, 'filter': True, },
  )