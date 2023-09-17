import asyncio
from enum import Enum
from typing import Optional

from dash import callback, dcc,html, no_update, register_page, Output, Input, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components.stock_header import StockHeader

from lib.edgar.company import Company
from lib.fin.utils import Taxonomy, calculate_items, load_template, merge_labels
from lib.ticker.fetch import find_cik, stock_label

register_page(__name__, path_template='/stock/<id>/', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = (
  'appearance-none absolute inset-0 h-full cursor-pointer '
  'checked:bg-secondary/50'
)
radio_label_style = 'relative px-1'

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

  cik = find_cik(id)
  if cik:
    template = load_template('sankey')
    taxonomy = Taxonomy(set(template['item']))
    financials = asyncio.run(Company(cik).financials_to_df('%Y-%m-%d', taxonomy))
    template = merge_labels(template, taxonomy)

    schema = taxonomy.calculation_schema(set(template['item']))
    financials = calculate_items(financials, schema)

    #pattern = r'^(Sales)?(AndService)?Revenue'\
    #  r'(s|Net|FromContractWithCustomerExcludingAssessedTax)'\
    #  r'(?=\..+\..+$)'
    #financials.columns = financials.columns.str.replace(pattern, 'rv')
    #rv = financials.filter(regex=r'^rv\..+\..+$', axis=1)

    financials = financials.reset_index().to_dict('records')
    template = template.to_dict('records')

    return html.Main(className='flex flex-col h-full', children=[
      StockHeader(id),
      html.Div(id='div:stock:sankey', children=[
        html.Div(className='flex justify-around', children=[
          dcc.Dropdown(id='dd:stock:date', className='w-36'),
          dcc.RadioItems(id='radio:stock:sheet', className=radio_wrap_style,
          inputClassName=radio_input_style,
          labelClassName=radio_label_style,
          value='income',
          options=[
            {'label': 'Income', 'value': 'income'},
            {'label': 'Balance', 'value': 'balance'},
            {'label': 'Cash Flow', 'value': 'cashflow'}
          ]),
          dcc.RadioItems(id='radio:stock:scope', className=radio_wrap_style,
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
      dcc.Store(id='store:stock:financials', data=financials),
      dcc.Store(id='store:stock:template', data=template)
    ])

@callback(
  Output('graph:stock:sankey', 'figure'),
  Input('store:stock:financials', 'data'),
  Input('radio:stock:sheet', 'value'),
  Input('dd:stock:date', 'value'),
  State('radio:stock:scope', 'value'),
  State('store:stock:template', 'data'),
)
def update_graph(fin: list[dict], sheet: str, date: str, scope: str, tmpl: list[dict]):

  if not date:
    return no_update
  
  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'months'])
    .xs((date, scope))
  )

  tmpl = pd.DataFrame.from_records(tmpl)
  tmpl = tmpl.loc[tmpl['sheet'] == sheet]
  tmpl = tmpl.loc[tmpl['item'].isin(fin.index)]

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
  Output('dd:stock:date', 'options'),
  Output('dd:stock:date', 'value'),
  Input('store:stock:financials', 'data'),
  Input('radio:stock:scope', 'value')
)
def update_dropdown(fin: list[dict], scope: str):
  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'months'])
    .xs(scope, level=1) 
    .sort_index(ascending=False) 
  )
  return list(fin.index.get_level_values(0)), ''