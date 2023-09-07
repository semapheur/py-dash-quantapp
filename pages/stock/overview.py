import asyncio
from enum import Enum
from typing import Optional

from dash import callback, dcc,html, register_page, Output, Input, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.edgar.company import Company
from lib.fin.utils import Taxonomy, load_template, merge_labels
from lib.ticker.fetch import find_cik, stock_label

register_page(__name__, path_template='/stock/<id>/overview', title=stock_label)

radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = 'appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50'
radio_label_style = 'relative px-1'

'''
sankey = {
  ('Revenue', 'GrossProfit'): 'rgba(0,255,0,0.5)',
  ('Revenue', 'COGS'): 'rgba(255,0,0,0.5)',
  ('GrossProfit', 'OperatingIncome'): 'rgba(0,255,0,0.5)'},
  ('GrossProfit', 'OperatingExpense'): {
    'value': 51.3, 'color': 'rgba(255,0,0,0.5)'},
  ('OperatingIncome', 'PretaxIncome'): {
    'value': 119.1, 'color': 'rgba(0,255,0,0.5)'},
  ('InterestIncome', 'PretaxIncome'): {
    'value': -106, 'color': 'rgba(255,0,0,0.5)'},
  ('OtherIncome', 'GrossProfit'): {
    'value': -228, 'color': 'rgba(255,0,0,0.5)'},
  ('OperatingExpense', 'SGA'): {
    'value': 25.1, 'color': 'rgba(255,0,0,0.5)'},
  ('OperatingExpense', 'RD'): {
    'value': 26.3, 'color': 'rgba(255,0,0,0.5)'},
  ('PretaxIncome', 'NetIncome'): {
    'value': 99.8, 'color': 'rgba(0,255,0,0.5)'},
  ('PretaxIncome', 'Tax'): {
    'value': 19.3, 'color': 'rgba(255,0,0,0.5)'}
}
'''

def sankey_color(sign: int, opacity: float = 1) -> str:
  return f'rgba(0,255,0,{opacity})' if sign == -1 else f'rgba(255,0,0,{opacity})'

def layout(id: Optional[str] = None):

  cik = find_cik(id)
  if cik:
    template = load_template('sankey')
    taxonomy = Taxonomy(set(template['item']))
    financials = asyncio.run(Company(cik).financials_to_df('%Y-%m-%d', taxonomy, True))
    template = merge_labels(template, taxonomy)

    #pattern = r'^(Sales)?(AndService)?Revenue'\
    #  r'(s|Net|FromContractWithCustomerExcludingAssessedTax)'\
    #  r'(?=\..+\..+$)'
    #financials.columns = financials.columns.str.replace(pattern, 'rv')
    #rv = financials.filter(regex=r'^rv\..+\..+$', axis=1)

    financials = financials.reset_index().to_dict('records')
    template = template.to_dict('records')

    return html.Main(className='flex flex-col h-full', children=[
      html.Div(id='div.stock-overview:sankey', children=[
        html.Div(className='flex justify-around', children=[
          dcc.Dropdown(id='dd.stock-overview:date'),
          dcc.RadioItems(id='radio.stock-overview:sheet', className=radio_wrap_style,
          inputClassName=radio_input_style,
          labelClassName=radio_label_style,
          value='income',
          options=[
            {'label': 'Income', 'value': 'income'},
            {'label': 'Balance', 'value': 'balance'},
            {'label': 'Cash Flow', 'value': 'cashflow'}
          ]),
          dcc.RadioItems(id='radio.stock-overview:scope', className=radio_wrap_style,
            inputClassName=radio_input_style,
            labelClassName=radio_label_style,
            value='a',
            options=[
              {'label': 'Annual', 'value': 'a'},
              {'label': 'Quarterly', 'value': 'q'},
            ]),
        ]),
        dcc.Graph(id='graph.stock-overview:sankey', responsive=True)
      ]),
      dcc.Store(id='store.stock-overview:financials', data=financials),
    ])

@callback(
  Output('graph.stock-overview:sankey', 'figure'),
  Input('store.stock-overview:financials', 'data'),
  Input('radio.stock-overview:sheet', 'value'),
  Input('dd.stock-overview:date', 'value'),
  State('radio.stock-overview:scope', 'value'),
  State('store.store-overview:template', 'data'),
)
def update_graph(fin: list[dict], sheet: str, date: str, scope: str, tmpl: list[dict]):

  tmpl = pd.DataFrame.from_records(tmpl)
  tmpl = tmpl.loc[tmpl['sheet'] == sheet]

  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'period'])
    .xs((date, scope))
  )

  Nodes = Enum('Node', tmpl['item'].tolist(), start=0)

  sources = []
  targets = []
  values = []
  link_colors = []
  node_colors = []

  for item in tmpl['item']:
    
    color = tmpl.loc[tmpl['item'] == item, 'color']
    if not color:
      color = sankey_color(np.sign(fin[item].iloc[0]))

    node_colors.append(color)

    links = tmpl.loc[tmpl['item'] == item, 'links']
    if links:
      for key in links.keys():
        sources.append(Nodes(item))
        targets.append(Nodes(key))
        values.append(fin[key].iloc[0])

        color = links[key]
        if not color:
          color = sankey_color(np.sign(fin[key].iloc[0]))

        link_colors.append(color)

  fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = 'black', width = 0.5),
      label = [member.name for member in Nodes],
      color = node_colors
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = link_colors
  ))])

  return fig

@callback(Output('dd.stock-overview:date', 'options'),
  Input('store.stock-overview:financials', 'data'),
  Input('radio.stock-overview:scope', 'value')
)
def update_dropdown(fin: list[dict], scope: str):
  fin = (pd.DataFrame.from_records(fin)
    .set_index(['date', 'period'])
    .xs(scope, level=1) 
    .sort_index(ascending=False) 
  )

  return list(fin.get_level_values(0))