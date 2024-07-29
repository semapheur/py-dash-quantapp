from typing import cast
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
from pandera.typing import DataFrame, Index, Series
import plotly.express as px

from components.modal import CloseModalAIO
from components.company_header import CompanyHeader
from lib.db.lite import fetch_sqlite, get_table_columns, read_sqlite
from lib.fin.fundamentals import load_financials
from lib.ticker.fetch import company_label

register_page(__name__, path_template="/company/<id>/financials", title=company_label)

modal_style = "relative m-auto rounded-md"
radio_wrap_style = "flex divide-x rounded-sm shadow"
radio_input_style = (
  "appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50"
)
radio_label_style = "relative px-1"


def row_indices(template: pd.DataFrame, level: int) -> str:
  mask = template["level"] == level
  return str(template.loc[mask].index.to_list())
  # return str(index[index.isin(items)].index.to_list())


def layout(id: str | None = None):
  if id is None:
    return html.Main(
      className="size-full", children=[html.H1("404", className="md-auto")]
    )

  return html.Main(
    className="relative flex flex-col h-full",
    children=[
      CompanyHeader(id) if id is not None else None,
      html.Div(
        className="flex justify-around",
        children=[
          dcc.RadioItems(
            id="radio:company-financials:sheet",
            className=radio_wrap_style,
            inputClassName=radio_input_style,
            labelClassName=radio_label_style,
            value="income",
            options=[
              {"label": "Income", "value": "income"},
              {"label": "Balance", "value": "balance"},
              {"label": "Cash Flow", "value": "cashflow"},
            ],
          ),
          dcc.RadioItems(
            id="radio:company-financials:scope",
            className=radio_wrap_style,
            inputClassName=radio_input_style,
            labelClassName=radio_label_style,
            value=12,
            options=[
              {"label": "Annual", "value": 12},
              {"label": "Quarterly", "value": 3},
            ],
          ),
        ],
      ),
      html.Div(id="div:company-financials:table-wrap", className="flex-1 p-2"),
      CloseModalAIO(
        aio_id="stock-financials",
        children=[dcc.Graph(id="graph:company-financials")],
      ),
    ],
  )


def sheet_items(sheet: str) -> DataFrame | None:
  query = """SELECT 
    s.item, items.short, items.long, s.level FROM statement AS s 
    LEFT JOIN items ON s.item = items.item
    WHERE s.sheet = :sheet
  """
  template = read_sqlite("taxonomy.db", query, {"sheet": sheet})
  if template is None:
    return None

  template.loc[:, "short"].fillna(template["long"], inplace=True)
  return template


def get_currency(id: str) -> str | None:
  currency = fetch_sqlite(
    "ticker.db", "SELECT currency FROM company WHERE company_id = :id", {"id": f"{id}"}
  )

  if currency is None:
    return None

  return currency[0][0]


def trend_data(row: Series[float]):
  return {"x": row.index.tolist(), "y": row.values.tolist()}


@callback(
  Output("div:company-financials:table-wrap", "children"),
  Input("radio:company-financials:sheet", "value"),
  Input("radio:company-financials:scope", "value"),
  State("location:app", "pathname"),
  background=True,
)
def update_table(sheet: str, scope: str, pathname: str):
  template = sheet_items(sheet)
  if template is None:
    return no_update

  company_id = pathname.split("/")[2]
  currency = get_currency(company_id)
  if currency is None:
    return no_update

  table = f"{company_id}_{currency}"

  table_columns = get_table_columns("financials.db", [table])[table]
  fin_columns = OrderedSet(template["item"]).intersection(table_columns)

  where = f"WHERE months = {scope}"
  financials = load_financials(company_id, currency, fin_columns, where)
  if financials is None:
    return no_update

  financials = cast(
    DataFrame,
    financials.reset_index(level=["period", "months"], drop=True).sort_index(
      ascending=False
    ),
  )
  financials.index = cast(pd.DatetimeIndex, financials.index).strftime("%Y-%m-%d")
  financials = cast(DataFrame, financials.T.reset_index())
  financials["trend"] = financials.iloc[:, 1:-1].apply(trend_data, axis=1)
  template = cast(
    DataFrame, template.set_index("item").loc[list(fin_columns)].reset_index()
  )

  columnDefs = [
    {
      "field": "index",
      "headerName": "Item",
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
      "cellStyle": {
        "styleConditions": [
          {
            "condition": (f"{row_indices(template, lvl)}" ".includes(params.rowIndex)"),
            "style": {"paddingLeft": f"{lvl + 1}rem"},
          }
          for lvl in template["level"].unique()
        ]
      },
      "tooltipField": "index",
      "tooltipComponentParams": {"labels": template["long"].to_list()},
    },
    {"field": "trend", "headerName": "Trend", "cellRenderer": "TrendLine"},
  ] + [
    {
      "field": col,
      "type": "numericColumn",
      "valueFormatter": {"function": 'd3.format("(,")(params.value)'},
    }
    for col in financials.columns[1:-1]
  ]  # .difference(['index', 'trend'])

  row_style = {
    "font-bold border-b border-text": (
      f"{row_indices(template, 0)}" ".includes(params.rowIndex)"
    )
  }

  financials.loc[:, "index"] = financials["index"].apply(
    lambda x: template.loc[template["item"] == x, "short"].iloc[0]
  )

  return dag.AgGrid(
    id="table:company-financials",
    columnDefs=columnDefs,
    rowData=financials.to_dict("records"),
    columnSize="autoSize",
    defaultColDef={"tooltipComponent": "FinancialsTooltip"},
    rowClassRules=row_style,
    style={"height": "100%"},
    dashGridOptions={"rowSelection": "single"},
  )


@callback(
  Output("graph:company-financials", "figure"),
  Input("table:company-financials", "selectedRows"),
)
def update_store(row: list[dict]):
  if not row:
    return []

  df = pd.DataFrame(row).drop(["index", "trend"], axis=1).T.sort_index()

  return px.line(
    x=df.index,
    y=df[0].pct_change(),
    title=row[0]["index"],
    labels={"x": "Date", "y": ""},
  )


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="cellClickModal"),
  Output("table:company-financials", "id"),
  Input("table:company-financials", "cellClicked"),
  State(CloseModalAIO.dialog_id("stock-financials"), "id"),
  prevent_initial_call=True,
)
