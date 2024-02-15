from dash import (
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Output,
  Input,
  State,
)
import dash_ag_grid as dag
from pandas import DataFrame

from lib.db.lite import read_sqlite, insert_sqlite, get_tables
from lib.edgar.financials import financials_table

EXCHANGES = [{'label': 'Oslo Stock Exchange (XOSL)', 'value': 'XOSL'}]

register_page(__name__, path_template='/screener/stock')

layout = html.Main(
  className='h-full grid grid-cols-[1fr_4fr]',
  children=[
    html.Div(
      className='h-full flex flex-col',
      children=[
        dcc.Dropdown(id='dropdown:screener-stock:exchange', options=EXCHANGES, value='')
      ],
    ),
    html.Div(id='div:screener-stock:table-wrap'),
  ],
)


@callback(
  Output('div:screener-stock:table-wrap', 'children'),
  Input('dropdown:screener-stock:exchange', 'value'),
)
async def update_table(exchange: str):
  if not exchange:
    return no_update

  query = 'SELECT id, name FROM stock WHERE mic=:=exchange'
  stocks = read_sqlite('ticker.db', query, params={'exchange': exchange})

  stock_ids = set(stocks['id']).intersection(get_tables('financials.db'))
  currency = CURRENCIES.get(exchange)

  table_ids = set(get_tables('financials.db'))

  tables: list[DataFrame] = []
  for i in stock_ids:
    if (table := f'{i}_{currency}') in table_ids:
      query = f'SELECT * FROM "{table}"'
      tables.append(read_sqlite('financials.db', query))
    else:
      df = await financials_table(i, currency)
      insert_sqlite(df.reset_index(), 'financials.db', table, '')
      tables.append(df)

  return
