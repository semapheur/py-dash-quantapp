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

from lib.db.lite import get_table_columns, read_sqlite

query = """SELECT
  e.market_name || " (" || se.mic || ":" || e.country || ")" AS label,
  se.mic AS value FROM stored_exchanges se
  JOIN exchange e ON se.mic = e.mic
"""
exchanges = read_sqlite("ticker.db", query)

register_page(__name__, path_template="/screener/stock")

layout = html.Main(
  className="h-full grid grid-cols-[1fr_4fr]",
  children=[
    html.Div(
      className="h-full flex flex-col",
      children=[
        dcc.Dropdown(
          id="dropdown:screener-stock:exchange",
          options=[] if exchanges is None else exchanges.to_dict("records"),
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
  background=True,
)
def update_table(exchange: str):
  if not exchange:
    return no_update

  query = """SELECT f.company_id FROM financials f
    JOIN stock s ON f.company_id = s.company_id
    WHERE s.mic = :exchange
  """
  companies = read_sqlite("ticker.db", query, params={"exchange": exchange})
  if companies is None:
    return no_update

  table_columns = get_table_columns("fundamentals.db", companies["company_id"].tolist())

  common_columns = set.intersection(*[set(cols) for cols in table_columns.values()])

  union_queries = []
  for table, columns in table_columns.items():
    select_columns = [f"'{table}' AS company"]
    for col in common_columns:
      if col in columns:
        select_columns.append(f"{col}")
      else:
        select_columns.append(f"NULL as {col}")

    select_clause = ", ".join(select_columns)
    union_queries.append(
      f"SELECT {select_clause} FROM '{table}' WHERE date = (SELECT MAX(date) FROM '{table}') AND months = 12"
    )

  full_query = " UNION ALL ".join(union_queries)

  df = read_sqlite("fundamentals.db", full_query)
  if df is None:
    return no_update

  column_defs = [
    {
      "field": "company",
      "type": "text",
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
    }
  ] + [
    {
      "field": col,
      "type": "number",
      "valueFormatter": {"function": 'd3.format("(,")(params.value)'},
    }
    for col in df.columns.difference(["company"])
  ]

  return dag.AgGrid(
    id="table:screener-stock",
    columnDefs=column_defs,
    rowData=df.to_dict("records"),
    columnSize="autoSize",
    style={"height": "100%"},
  )
