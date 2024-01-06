from typing import Optional
from ordered_set import OrderedSet

from dash import (
  callback,
  clientside_callback,
  ClientsideFunction,
  dcc,
  html,
  no_update,
  register_page,
  Output,
  Input,
  State,
)
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px

from components.stock_header import StockHeader
from lib.db.lite import read_sqlite
from lib.ticker.fetch import stock_label
# from lib.utils import load_json

register_page(__name__, path_template='/stock/<id_>/financials', title=stock_label)

modal_style = 'relative m-auto rounded-md'
radio_wrap_style = 'flex divide-x rounded-sm shadow'
radio_input_style = (
  'appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50'
)
radio_label_style = 'relative px-1'


def row_indices(template: pd.DataFrame, level: int) -> str:
  mask = template['level'] == level
  return str(template.loc[mask].index.to_list())
  # return str(index[index.isin(items)].index.to_list())


def layout(id_: Optional[str] = None):
  return html.Main(
    className='relative flex flex-col h-full',
    children=[
      StockHeader(id_) if id_ is not None else None,
      html.Div(
        className='flex justify-around',
        children=[
          dcc.RadioItems(
            id='radio:stock-financials:sheet',
            className=radio_wrap_style,
            inputClassName=radio_input_style,
            labelClassName=radio_label_style,
            value='income',
            options=[
              {'label': 'Income', 'value': 'income'},
              {'label': 'Balance', 'value': 'balance'},
              {'label': 'Cash Flow', 'value': 'cashflow'},
            ],
          ),
          dcc.RadioItems(
            id='radio:stock-financials:scope',
            className=radio_wrap_style,
            inputClassName=radio_input_style,
            labelClassName=radio_label_style,
            value=12,
            options=[
              {'label': 'Annual', 'value': 12},
              {'label': 'Quarterly', 'value': 3},
            ],
          ),
        ],
      ),
      html.Div(id='div:stock-financials:table-wrap', className='flex-1 p-2'),
      html.Dialog(
        id='dialog:stock-financials',
        className=modal_style,
        children=[
          dcc.Graph(id='graph:stock-financials'),
          html.Button(
            'x',
            id='button:stock-financials:close-modal',
            className='absolute top-0 left-2 text-3xl text-secondary hover:text-red-600',
          ),
        ],
      ),
    ],
  )


@callback(
  Output('div:stock-financials:table-wrap', 'children'),
  Input('store:ticker-search:financials', 'data'),
  Input('radio:stock-financials:sheet', 'value'),
  Input('radio:stock-financials:scope', 'value'),
)
def update_table(data: list[dict], sheet: str, scope: str):
  if not data:
    return no_update

  query = """SELECT 
    s.item, items.short, items.long, s.level FROM statement AS s 
    LEFT JOIN items ON s.item = items.item
    WHERE s.sheet = :sheet
  """
  param = {'sheet': sheet}
  tmpl = read_sqlite('taxonomy.db', query, param)
  tmpl.loc[:, 'short'].fillna(tmpl['long'], inplace=True)

  fin = (
    pd.DataFrame.from_records(data)
    .set_index(['date', 'months'])
    .xs(scope, level='months')
    .sort_index(ascending=False)
  )
  cols = list(
    OrderedSet(OrderedSet(tmpl['item']).intersection(OrderedSet(fin.columns)))
  )
  fin = fin[cols]
  fin = fin.T.reset_index()
  tmpl = tmpl.set_index('item').loc[cols].reset_index()

  fin['trend'] = ''
  for i, r in fin.iterrows():
    fig = px.line(r.iloc[1:-1])
    fig.update_layout(
      showlegend=False,
      xaxis=dict(autorange='reversed'),
      xaxis_visible=False,
      xaxis_showticklabels=False,
      yaxis_visible=False,
      yaxis_showticklabels=False,
      margin=dict(l=0, r=0, t=0, b=0),
      template='plotly_white',
    )
    fin.at[i, 'trend'] = fig

  columnDefs = [
    {
      'field': 'index',
      'headerName': 'Item',
      'pinned': 'left',
      'lockPinned': True,
      'cellClass': 'lock-pinned',
      'cellStyle': {
        'styleConditions': [
          {
            'condition': (f'{row_indices(tmpl, lvl)}' '.includes(params.rowIndex)'),
            'style': {'paddingLeft': f'{lvl + 1}rem'},
          }
          for lvl in tmpl['level'].unique()
        ]
      },
      'tooltipField': 'index',
      'tooltipComponentParams': {'labels': tmpl['long'].to_list()},
    },
    {'field': 'trend', 'headerName': 'Trend', 'cellRenderer': 'TrendLine'},
  ] + [
    {
      'field': col,
      'type': 'numericColumn',
      'valueFormatter': {'function': 'd3.format("(,")(params.value)'},
    }
    for col in fin.columns[1:-1]
  ]  # .difference(['index', 'trend'])

  row_style = {
    'font-bold border-b border-text': (
      f'{row_indices(tmpl, 0)}' '.includes(params.rowIndex)'
    )
  }

  fin.loc[:, 'index'] = fin['index'].apply(
    lambda x: tmpl.loc[tmpl['item'] == x, 'short'].iloc[0]
  )

  return dag.AgGrid(
    id='table:stock-financials',
    columnDefs=columnDefs,
    rowData=fin.to_dict('records'),
    columnSize='autoSize',
    defaultColDef={'tooltipComponent': 'FinancialsTooltip'},
    rowClassRules=row_style,
    style={'height': '100%'},
    dashGridOptions={'rowSelection': 'single'},
  )


@callback(
  Output('graph:stock-financials', 'figure'),
  Input('table:stock-financials', 'selectedRows'),
)
def update_store(row: list[dict]):
  if not row:
    return []

  df = pd.DataFrame(row).drop(['index', 'trend'], axis=1).T.sort_index()

  return px.line(
    x=df.index,
    y=df[0].pct_change(),
    title=row[0]['index'],
    labels={'x': 'Date', 'y': ''},
  )


clientside_callback(
  ClientsideFunction(namespace='clientside', function_name='row_select_modal'),
  Output('table:stock-financials', 'selectedRows'),
  Input('table:stock-financials', 'selectedRows'),
  State('dialog:stock-financials', 'id'),
)

clientside_callback(
  ClientsideFunction(namespace='clientside', function_name='close_modal'),
  Output('dialog:stock-financials', 'id'),
  Input('button:stock-financials:close-modal', 'n_clicks'),
  State('dialog:stock-financials', 'id'),
)
