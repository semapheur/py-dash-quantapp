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
from lib.db.lite import get_table_columns, fetch_sqlite, read_sqlite

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
    dag.AgGrid(
      id="table:screener-stock",
      getRowId="params.data.company",
      columnSize="autoSize",
      style={"height": "100%"},
    ),
    CloseModalAIO(
      aio_id="screener-stock",
      children=[
        dcc.Loading(
          children=[dcc.Graph(id="graph:screener-stock")],
          target_components={"graph:screener-stock": "figure"},
        )
      ],
    ),
  ],
)


@callback(
  Output("table:screener-stock", "columnDefs"),
  Output("table:screener-stock", "rowData"),
  Input("dropdown:screener-stock:exchange", "value"),
  background=True,
  prevent_initial_call=True,
)
def update_table(exchange: str):
  if not exchange:
    return no_update

  def get_company_info(exchange: str) -> DataFrame | None:
    query = """SELECT f.company_id, c.name, c.sector, GROUP_CONCAT(s.ticker, ',') AS tickers FROM financials f
      JOIN company c ON f.company_id = c.company_id
      JOIN stock s ON f.company_id = s.company_id
      WHERE s.mic = :exchange
      GROUP BY f.company_id, c.name, c.sector
    """
    companies = read_sqlite("ticker.db", query, params={"exchange": exchange})
    return companies

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

  companies = get_company_info(exchange)
  if companies is None:
    return no_update

  table_columns = get_table_columns("fundamentals.db", companies["company_id"].tolist())

  all_columns = set.union(*[cols for cols in table_columns.values()])

  union_queries = []
  for table, columns in table_columns.items():
    select_columns = [f"'{table}' AS company_id"]
    for col in all_columns:
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
    + "|"
    + fundamentals["sector"]
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

  return column_defs, fundamentals.to_dict("records")


@callback(
  Output("graph:screener-stock", "figure"),
  Input("table:screener-stock", "cellClicked"),
  State("dropdown:screener-stock:exchange", "value"),
  # prevent_initial_call=True,
  background=True,
)
def update_store(cell: dict, exchange: str):
  if not cell:
    return {}

  def create_figure(data: DataFrame, title: str):
    fig = px.bar(
      company_data,
      title=metric_label,
      labels={"date": "Date", "value": "Value", "variable": "Entity"},
      barmode="group",
    )
    fig.update_layout(
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

  metric = cell["colId"]
  if metric in ["sector", "company"]:
    return no_update

  name, table, sector = cast(str, cell["rowId"]).split("|")

  query = f"SELECT date, {metric} AS '{name}' FROM '{table}' WHERE period = 'FY'"

  company_data = read_sqlite("fundamentals.db", query, index_col="date")
  if company_data is None:
    return no_update

  metric_label = fetch_sqlite(
    "taxonomy.db", f"SELECT long FROM items WHERE item = '{metric}'"
  )
  if metric_label is not None:
    metric_label = metric_label[0][0]

  query = """SELECT f.company_id, c.sector FROM financials f
    JOIN company c ON f.company_id = c.company_id
    JOIN stock s ON f.company_id = s.company_id
    WHERE s.mic = :exchange
  """
  exchange_companies = read_sqlite("ticker.db", query, {"exchange": exchange})
  if exchange_companies is None:
    return create_figure(company_data, metric_label)

  table_columns = get_table_columns(
    "fundamentals.db", exchange_companies["company_id"].tolist()
  )

  union_queries = [
    f"SELECT '{t}' as company_id, date, {metric} FROM '{t}' WHERE period = 'FY'"
    for t in table_columns
    if metric in table_columns[t]
  ]
  full_query = " UNION ALL ".join(union_queries)
  exchange_data = read_sqlite(
    "fundamentals.db", full_query, index_col=["company_id", "date"]
  )

  if exchange_data is None:
    return create_figure(company_data, metric_label)

  mask = exchange_companies["sector"] == sector

  sector_data = exchange_data.loc[
    exchange_data.index.get_level_values("company_id").isin(
      exchange_companies.loc[mask, "company_id"]
    ),
    :,
  ]

  exchange_data = cast(
    DataFrame,
    (exchange_data.groupby("date").mean().rename(columns={metric: exchange})),
  )
  sector_data = sector_data.groupby("date").mean().rename(columns={metric: sector})

  company_data = cast(DataFrame, company_data.join(exchange_data, how="outer"))
  company_data = cast(DataFrame, company_data.join(sector_data, how="outer"))

  return create_figure(company_data, metric_label)


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="cell_click_modal"),
  Output("table:screener-stock", "id"),
  Input("table:screener-stock", "cellClicked"),
  State(CloseModalAIO.dialog_id("screener-stock"), "id"),
  prevent_initial_call=True,
)
