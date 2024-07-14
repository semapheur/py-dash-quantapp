from typing import cast

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
from pandera.typing import DataFrame
import plotly.express as px

from components.modal import CloseModalAIO
from lib.db.lite import get_table_columns, read_sqlite

link_style = "block text-text hover:text-secondary"
modal_style = "relative m-auto rounded-md"

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
    CloseModalAIO(
      aio_id="screener-stock",
      children=[dcc.Graph(id="graph:screener-stock")],
    ),
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

  def get_ratio_details() -> DataFrame:
    query = """SELECT item, long, short FROM items 
      WHERE unit IN ('monetary_ratio', 'price_ratio', 'numeric_score')
    """

    items = read_sqlite("taxonomy.db", query)
    if items is None:
      raise ValueError("Ratio items not found")

    items.loc[:, "short"].fillna(items["long"], inplace=True)
    items.set_index("item", inplace=True)
    return items

  query = """SELECT f.company_id, c.name, c.sector, GROUP_CONCAT(s.ticker, ',') AS tickers FROM financials f
    JOIN company c ON f.company_id = c.company_id
    JOIN stock s ON f.company_id = s.company_id
    WHERE s.mic = :exchange
    GROUP BY f.company_id, c.name, c.sector
  """
  companies = read_sqlite("ticker.db", query, params={"exchange": exchange})
  if companies is None:
    return no_update

  table_columns = get_table_columns("fundamentals.db", companies["company_id"].tolist())

  common_columns = set.intersection(*[set(cols) for cols in table_columns.values()])

  union_queries = []
  for table, columns in table_columns.items():
    select_columns = [f"'{table}' AS company_id"]
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

  fundamentals = read_sqlite("fundamentals.db", full_query)
  if fundamentals is None:
    return no_update

  fundamentals = cast(DataFrame, fundamentals.merge(companies, on="company_id"))

  fundamentals["company"] = (
    fundamentals["name"]
    + " ("
    + fundamentals["tickers"]
    + ")"
    + "|"
    + fundamentals["company_id"]
  )
  fundamentals.drop(
    ["company_id", "name", "tickers", "period", "date", "months"], axis=1, inplace=True
  )

  items = get_ratio_details()

  column_defs = [
    {
      "field": "company",
      "type": "text",
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
      "cellRenderer": "CompanyLink",
    },
    {
      "field": "sector",
      "type": "text",
    },
  ] + [
    {
      "field": col,
      "headerName": items.at[col, "short"],
      "headerTooltip": items.at[col, "long"],
      "type": "number",
      "valueFormatter": {"function": "d3.format('(.2f')(params.value)"},
      "filter": "agNumberColumnFilter",
    }
    for col in fundamentals.columns.difference(["company", "sector"])
  ]

  return dag.AgGrid(
    id="table:screener-stock",
    columnDefs=column_defs,
    rowData=fundamentals.to_dict("records"),
    getRowId="params.data.company",
    columnSize="autoSize",
    style={"height": "100%"},
  )


@callback(
  Output("graph:screener-stock", "figure"),
  Input("table:screener-stock", "cellClicked"),
  # prevent_initial_call=True,
  background=True,
)
def update_store(cell: dict):
  if not cell:
    return {}

  metric = cell["colId"]
  if metric in ["sector", "company"]:
    return no_update

  table = cast(str, cell["rowId"]).split("|")[1]

  query = f"SELECT date, {metric} FROM '{table}' WHERE period = 'FY'"

  df = read_sqlite("fundamentals.db", query, index_col="date")
  if df is None:
    return no_update

  return px.line(
    df,
    title=metric,
    labels={"x": "Date", "y": ""},
  )


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="cell_click_modal"),
  Output("table:screener-stock", "id"),
  Input("table:screener-stock", "cellClicked"),
  State(CloseModalAIO.dialog_id("screener-stock"), "id"),
  prevent_initial_call=True,
)
