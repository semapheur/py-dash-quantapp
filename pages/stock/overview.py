from enum import Enum
import json
from typing import Optional

from dash import (
  callback, dcc, html, no_update, register_page,
  Output, Input, State, 
  MATCH)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from components.dupont_chart import DupontChart

from components.stock_header import StockHeader
from lib.ticker.fetch import stock_label

register_page(__name__, path_template='/stock/<id>/', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = (
  'appearance-none absolute inset-0 h-full cursor-pointer '
  'checked:bg-secondary/50'
)
radio_label_style = 'relative px-1'

span_ids = (
  'return_on_equity',
  'net_profit_margin',
  'operating_profit_margin',
  'operating_margin:operating_income_loss',
  'operating_margin:revenue',
  'tax_burden',
  'net_income_loss',
  'tax_burden:pretax_income_loss',
  'interest_burden',
  'interest_burden:pretax_income_loss',
  'interest_burden:operating_income_loss',
  'asset_turnover',
  'asset_turnover:revenue',
  'asset_turnover:average_assets',
  'equity_multiplier',
  'equity_multiplier:average_assets',
  'average_equity'
)

dupont_items = {
  item.split(':')[1] if ':' in item else item for item in span_ids 
}

def sankey_color(sign: int, opacity: float = 1) -> str:
  return f'rgba(255,0,0,{opacity})' if sign == -1 else f'rgba(0,255,0,{opacity})'

def sankey_direction(sign: int) -> str:
  match sign:
    case -1:
      return 'in'
    case 1:
      return 'out'
    case _:
      raise Exception('Invalid sign')

def layout(id: Optional[str] = None):

  return html.Main(className='flex flex-col h-full overflow-y-scroll', children=[
    StockHeader(id),
    html.Div(id='div:stock:sankey', children=[
      html.Form(className='flex justify-around', children=[
        dcc.Dropdown(
          id={'type': 'dropdown:stock:date', 'id': 'sankey'}, 
          className='w-36'),
        dcc.RadioItems(
          id='radio:stock:sheet',
          className=radio_wrap_style,
          inputClassName=radio_input_style,
          labelClassName=radio_label_style,
          value='income',
          options=[
            {'label': 'Income', 'value': 'income'},
            {'label': 'Balance', 'value': 'balance'},
            {'label': 'Cash Flow', 'value': 'cashflow'}]),
        dcc.RadioItems(
          id={'type': 'radio:stock:scope', 'id': 'sankey'}, 
          className=radio_wrap_style,
          inputClassName=radio_input_style,
          labelClassName=radio_label_style,
          value=12,
          options=[
            {'label': 'Annual', 'value': 12},
            {'label': 'Quarterly', 'value': 3},
          ]),
      ]),
      dcc.Graph(id='graph:stock:sankey', responsive=True)
    ]),
    html.Div(children=[
      html.Form(children=[
        dcc.Dropdown(
          id={'type': 'dropdown:stock:date', 'id': 'dupont'}, 
          className='w-36'),
        dcc.RadioItems(
          id={'type': 'radio:stock:scope', 'id': 'dupont'}, 
          className=radio_wrap_style,
          inputClassName=radio_input_style,
          labelClassName=radio_label_style,
          value=12,
          options=[
            {'label': 'Annual', 'value': 12},
            {'label': 'Quarterly', 'value': 3},
          ]),
      ]),
      DupontChart()
    ])
  ])

@callback(
  Output('graph:stock:sankey', 'figure'),
  Input('radio:stock:sheet', 'value'),
  Input('dd:stock:date', 'value'),
  State('radio:stock:scope', 'value'),
  State('store:ticker-search:financials', 'data'),
)
def update_graph(sheet: str, date: str, scope: str, data: list[dict]):

  if not (date and data):
    return no_update
  
  fin = (pd.DataFrame.from_records(data)
    .set_index(['date', 'months'])
    .xs((date, scope))
  )

  engine = create_engine('sqlite+pysqlite:///data/taxonomy.db')
  query = text('''SELECT 
    sankey.item, items.short, items.long, sankey.color, sankey.links FROM sankey 
    LEFT JOIN items ON sankey.item = items.item
      WHERE sankey.sheet = :sheet
  ''').bindparams(sheet=sheet)

  with engine.connect().execution_options(autocommit=True) as con:
    tmpl = pd.read_sql(query, con=con)

  tmpl = tmpl.loc[tmpl['item'].isin(set(fin.index))]
  tmpl.loc[:,'links'] = tmpl['links'].apply(lambda x: json.loads(x))
  tmpl.loc[:,'short'].fillna(tmpl['long'], inplace=True)

  Nodes = Enum('Node', tmpl['item'].tolist(), start=0)

  sources = []
  targets = []
  values = []
  link_colors = []
  node_colors = []

  for item, node_color, links in zip(tmpl['item'], tmpl['color'], tmpl['links']):    

    if not node_color:
      node_color = sankey_color(np.sign(fin.loc[item]))

    node_colors.append(node_color)

    if not links:
      continue

    for key, value in links.items():

      if key not in set(tmpl['item']):
        continue
      
      link_value = fin.loc[value.get('value', key)]

      sign = value.get('sign', np.sign(link_value))
      if sign != np.sign(link_value):
        continue

      values.append(np.abs(link_value))

      if not (direction := value.get('direction')):
        direction = sankey_direction(np.sign(link_value))

      if direction == 'in':
        source = Nodes[key].value
        target = Nodes[item].value
      else:
        source = Nodes[item].value
        target = Nodes[key].value
      
      sources.append(source)
      targets.append(target)

      if not (color := value.get('color')):
        color = sankey_color(np.sign(link_value), 0.3)

      link_colors.append(color)

  fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 10,
      line = dict(color = 'black', width = 0.5),
      label = tmpl['short'].tolist(),
      color = node_colors
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = link_colors
  ))])

  return fig

@callback(
  [Output('span:dupont:' + span_id, 'children') for span_id in span_ids],
  Input({'type': 'radio:stock:date', 'id': 'dupont'}, 'value'),
  State({'type': 'radio:stock:scope', 'id': 'dupont'}, 'value'),
  State('store:ticker-search:financials', 'data'),
)
def update_dupont(date: str, scope: int, data: list[dict]):
  if not (date and data):
    return no_update
  
  fin = (pd.DataFrame.from_records(data)
    .set_index(['date', 'months'])
    .xs((date, scope))
  )

  return (
    fin.loc[span_id.split(':')[-1]] for span_id in span_ids
  )

@callback(
  Output({'type': 'dd:stock:date', 'id': MATCH}, 'options'),
  Output({'type': 'dd:stock:date', 'id': MATCH}, 'value'),
  Input({'type': 'radio:stock:scope', 'id': MATCH}, 'value'),
  Input('store:ticker-search:financials', 'data'),
)
def update_dropdown(scope: str, data: list[dict]):
  if not data:
    return no_update

  fin = (pd.DataFrame.from_records(data)
    .set_index(['date', 'months'])
    .xs(scope, level=1) 
    .sort_index(ascending=False) 
  )
  return list(fin.index.get_level_values(0)), ''