from dash import (
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  Patch,
  State,
)
from dashvis import DashNetwork

import pandas as pd

from lib.xbrl.utils import gaap_network

register_page(__name__, path='/accounting')

nodes, edges = gaap_network()
data = {'nodes': nodes, 'edges': edges}
network = DashNetwork(
  id='network:gaap',
  style={'height': '100%'},
  data=data,
  options={
    'nodes': {'shape': 'box'},
    'physics': {'enabled': False},
    'layout': {'hierarchical': {'enabled': False, 'sortMethod': 'directed'}},
  },
)

layout = html.Main(
  className='h-full grid grid-cols-[1fr_4fr]',
  children=[
    html.Aside(
      className='h-full',
      children=[
        dcc.Dropdown(
          id='dropdown:gaap:nodes',
          options=[{'value': node['id'], 'label': node['id']} for node in nodes],
          multi=True,
          placeholder='Nodes',
          value=[],
        )
      ],
    ),
    network,
    dcc.Store(id='store:gaap:data', data=data),
  ],
)


@callback(
  Output('network:gaap', 'data'),
  Output('network:gaap', 'options'),
  Input('dropdown:gaap:nodes', 'value'),
  State('store:gaap:data', 'data'),
  prevent_initial_call=True,
)
def update_data(select_nodes: list[str], data: dict):
  options = Patch()
  if not select_nodes:
    # options['layout']['hierarchical']['enabled'] = False
    # return data, options
    return no_update

  edges = pd.DataFrame.from_records(data['edges'])
  nodes = pd.DataFrame.from_records(data['nodes'])

  mask = edges['from'].isin(select_nodes) | edges['to'].isin(select_nodes)
  edges_ = edges.loc[mask]

  mask = edges['from'].isin(set(edges_['from'].unique()).difference(select_nodes))
  edges_ = pd.concat([edges_, edges.loc[mask]])

  mask = nodes['id'].isin(set(edges_['from']).union(edges_['to']))
  nodes = nodes[mask]

  network_data = {'nodes': nodes.to_dict('records'), 'edges': edges_.to_dict('records')}

  options['layout']['hierarchical']['enabled'] = True

  return network_data, options
