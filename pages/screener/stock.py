import textwrap

from dash import (
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Output,
  Input,
)
import dash_ag_grid as dag

from lib.db.lite import read_sqlite

exchanges = read_sqlite("ticker.db", "SELECT DISTINCT exchange FROM fundamentals")

register_page(__name__, path_template="/screener/stock")

layout = html.Main(
  className="h-full grid grid-cols-[1fr_4fr]",
  children=[
    html.Div(
      className="h-full flex flex-col",
      children=[
        dcc.Dropdown(
          id="dropdown:screener-stock:exchange",
          options=[] if exchanges is None else exchanges["exchange"].tolist(),
          value="",
        )
      ],
    ),
    html.Div(id="div:screener-stock:table-wrap"),
  ],
)


@callback(
  Output("div:screener-stock:table-wrap", "children"),
  Input("dropdown:screener-stock:exchange", "value"),
)
def update_table(exchange: str):
  if not exchange:
    return no_update

  query = "SELECT company_id, currency FROM fundamentals WHERE exchange = :exchange"
  companies = read_sqlite("ticker.db", query, params={"exchange": exchange})
  if companies is None:
    return no_update

  companies["table"] = companies["company_id"] + "_" + companies["currency"]

  query_parts = [
    textwrap.dedent(
      f"""SELECT '{table}' AS company, * FROM '{table}'
        WHERE date = (SELECT MAX(date) FROM '{table}' WHERE months = 12)    
    """
    )
    for table in companies["table"]
  ]

  query = " UNION ALL ".join(query_parts)
  df = read_sqlite("fundamentals.db", query)
  if df is None:
    return no_update

  df.set_index("company", inplace=True)

  column_defs = [
    {
      "field": col,
      "type": "numericColumn",
      "valueFormatter": {"function": 'd3.format("(,")(params.value)'},
    }
    for col in df.columns
  ]

  return dag.AgGrid(
    id="table:screener-stock",
    columnDefs=column_defs,
    rowData=df.to_dict("records"),
    columnSize="autoSize",
    style={"height": "100%"},
  )
