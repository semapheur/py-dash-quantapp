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
import pandas as pd
from pandera.typing import DataFrame
import plotly.express as px
import plotly.graph_objects as go

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

register_page(__name__, path="/screener/stock")

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
      dashGridOptions={"tooltipInteraction": True},
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

  def ratio_labels() -> DataFrame:
    query = """SELECT item, long, short FROM items 
      WHERE type = 'fundamental'
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

    select_sql = ", ".join(select_columns)
    union_queries.append(
      f"SELECT {select_sql} FROM '{table}' WHERE date = (SELECT MAX(date) FROM '{table}' WHERE months = 12)"
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

  items = ratio_labels()

  column_defs = [
    {
      "field": "company",
      "cellDataType": "text",
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
      "cellRenderer": "CompanyLink",
    },
    {
      "field": "sector",
      "cellDataType": "text",
    },
  ] + [
    {
      "field": col,
      "headerName": items.at[col, "short"],
      "headerTooltip": items.at[col, "long"],
      "tooltipField": col,
      "tooltipComponent": "ScreenerTooltip",
      "tooltipComponentParams": {
        "exchange": exchange,
        "exchangeMean": fundamentals[col].mean(),
        "exchangeMin": fundamentals[col].min(),
        "exchangeMax": fundamentals[col].max(),
        "sectorMean": fundamentals.groupby("sector")[col].mean().to_dict(),
        "sectorMin": fundamentals.groupby("sector")[col].min().to_dict(),
        "sectorMax": fundamentals.groupby("sector")[col].max().to_dict(),
      },
      "cellDataType": "number",
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
      data,
      title=title,
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

  exchange_mean = (
    exchange_data.groupby("date").mean().rename(columns={metric: exchange})
  )
  exchange_min = (
    exchange_data.groupby("date").min().rename(columns={metric: f"{exchange}_min"})
  )
  exchange_max = (
    exchange_data.groupby("date").max().rename(columns={metric: f"{exchange}_max"})
  )
  exchange_data = cast(
    DataFrame,
    pd.concat(
      [exchange_mean, exchange_min, exchange_max],
      axis=1,
    ).dropna(how="all"),
  )

  sector_mean = sector_data.groupby("date").mean().rename(columns={metric: sector})
  sector_min = (
    sector_data.groupby("date").min().rename(columns={metric: f"{sector}_min"})
  )
  sector_max = (
    sector_data.groupby("date").max().rename(columns={metric: f"{sector}_max"})
  )
  sector_data = pd.concat(
    [sector_mean, sector_min, sector_max],
    axis=1,
  ).dropna(how="all")

  figure = go.Figure()

  figure.add_bar(
    x=exchange_data.index,
    y=exchange_data[f"{exchange}_max"] - exchange_data[f"{exchange}_min"],
    base=exchange_data[f"{exchange}_min"],
    marker_color="red",
    opacity=0.25,
    name=f"{exchange} (range)",
    legendrank=1,
    hovertemplate=("<b>High:</b> %{y:.2f}<br><b>Low:</b> %{base:.2f}<br>"),
  )

  figure.add_bar(
    x=sector_data.index,
    y=sector_data[f"{sector}_max"] - sector_data[f"{sector}_min"],
    base=sector_data[f"{sector}_min"],
    marker_color="blue",
    opacity=0.25,
    name=f"{sector} (range)",
    legendrank=2,
    hovertemplate=("<b>High:</b> %{y:.2f}<br><b>Low:</b> %{base:.2f}<br>"),
  )

  figure.add_scatter(
    x=exchange_data.index,
    y=exchange_data[exchange],
    mode="markers",
    marker_color="red",
    name=exchange,
    legendrank=1,
    hovertemplate="%{y:.2f}",
  )

  figure.add_scatter(
    x=sector_data.index,
    y=sector_data[sector],
    mode="markers",
    marker_color="blue",
    name=sector,
    legendrank=2,
    hovertemplate="%{y:.2f}",
  )

  figure.add_scatter(
    x=company_data.index,
    y=company_data[name],
    mode="markers",
    marker_color="green",
    name=name,
    legendrank=3,
    hovertemplate="%{y:.2f}",
  )

  figure.update_layout(
    title=metric_label,
    xaxis_title="Date",
    yaxis_title="Value",
    barmode="group",
    bargap=0.75,
    bargroupgap=0,
    hovermode="x",
    # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
  )

  return figure


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="cell_click_modal"),
  Output("table:screener-stock", "id"),
  Input("table:screener-stock", "cellClicked"),
  State(CloseModalAIO.dialog_id("screener-stock"), "id"),
  prevent_initial_call=True,
)
