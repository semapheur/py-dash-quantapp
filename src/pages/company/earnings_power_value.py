from typing import cast, TypedDict, Optional

from dash import callback, dcc, html, no_update, register_page, Input, Output, State
import dash_ag_grid as dag
import numpy as np
import openturns as ot
from ordered_set import OrderedSet
import pandas as pd
from pandera.typing import DataFrame
from plotly import express as px

from lib.db.lite import fetch_sqlite
from lib.fin.epv import earnings_power
from lib.fin.fundamentals import load_fundamentals
from lib.ticker.fetch import company_label, get_currency
from lib.probability import make_distribution, nearest_postive_definite_matrix, plot_pdf

register_page(
  __name__, path_template="/company/<id>/earnings-power-value", title=company_label
)

distributions = ["Normal", "Skewnormal", "Triangular", "Uniform"]


class Factor(TypedDict, total=False):
  header: Optional[str]
  distribution: tuple[str, str]


factors: dict[str, Factor] = {
  "revenue": {"distribution": ("Normal", "1, 0.001")},
  "operating_profit_margin": {"distribution": ("Normal", "0.10, 0.02")},
  "tax_rate": {"distribution": ("Normal", "0.25, 0.05")},
  "capex_margin": {
    "header": "CAPEX margin",
    "distribution": ("Normal", "0.10, 0.05"),
  },
  "maintenance_rate": {"distribution": ("Normal", "0.10, 0.05")},
  "ddaa_margin": {
    "header": "DDAA margin",
    "distribution": ("Normal", "0.10, 0.05"),
  },
  "riskfree_rate": {
    "header": "Risk-free rate",
    "distribution": ("Normal", "0.04, 0.01"),
  },
  "yield_spread": {"distribution": ("Normal", "0.02, 0.01")},
  "equity_risk_premium": {"distribution": ("Normal", "0.02, 0.01")},
  "equity_to_capital": {"distribution": ("Normal", "0.5, 0.3")},
  "beta": {"distribution": ("Normal", "0.5, 0.3")},
}


def layout(id: str | None = None):
  if id is None:
    return html.Div("404")

  currency = get_currency(id)

  if currency is None:
    return html.Div("404")

  columns = OrderedSet(
    (
      "revenue",
      "operating_profit_margin",
      "tax_rate",
      "payment_acquisition_productive_assets",
      "depreciation_depletion_amortization_accretion",
      "riskfree_rate",
      "yield_spread",
      "equity_risk_premium",
      "equity_to_capital",
      "beta",
    )
  )
  where = "WHERE months = 12"
  df = load_fundamentals(id, currency, columns, where)
  if df is None:
    return html.Div("404")

  df["capex_margin"] = df["payment_acquisition_productive_assets"] / df["revenue"]
  df["ddaa_margin"] = (
    df["depreciation_depletion_amortization_accretion"] / df["revenue"]
  )
  df["maintenance_rate"] = 1 - df["revenue"].pct_change()
  df = cast(
    DataFrame,
    df.reset_index(level=["period", "months"], drop=True).sort_index(ascending=False),
  )
  df.index = cast(pd.DatetimeIndex, df.index).strftime("%Y-%m-%d")
  df = cast(DataFrame, df[list(factors.keys())])

  df = cast(
    DataFrame,
    df[list(factors.keys())].T.reset_index(),
  )

  df[["distribution", "parameters"]] = df["index"].apply(
    lambda f: pd.Series(factors[f]["distribution"])
  )
  df.loc[:, "index"] = df["index"].apply(
    lambda f: factors[f].get("header", f.replace("_", " ").capitalize())
  )

  epv_columns = [
    {
      "field": "index",
      "headerName": "Factor",
      "editable": False,
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
    },
    {
      "headerName": "Inputs",
      "children": [
        {
          "field": "distribution",
          "pinned": "left",
          "lockPinned": True,
          "cellClass": "lock-pinned",
          "cellEditor": "agSelectCellEditor",
          "cellEditorParams": {"values": distributions},
        },
        {
          "field": "parameters",
          "pinned": "left",
          "lockPinned": True,
          "cellClass": "lock-pinned",
          "cellEditor": {"function": "ParameterInput"},
        },
      ],
    },
  ] + [
    {
      "field": col,
      "type": "numericColumn",
      "editable": False,
      "valueFormatter": {"function": 'd3.format("(,.2g")(params.value)'},
    }
    for col in df.columns[1:-2]
  ]

  return html.Main(
    className="h-full grid grid-rows-2",
    children=[
      dag.AgGrid(
        id="table:company-epv",
        columnDefs=epv_columns,
        rowData=df.to_dict("records"),
        columnSize="autoSize",
        defaultColDef={"editable": True},
        dashGridOptions={"singleClickEdit": True, "rowSelection": "single"},
        style={"height": "100%"},
      ),
      html.Div(
        className="grid grid-cols-2",
        children=[
          dcc.Graph(id="graph:company-epv:distribution"),
          dcc.Graph(id="graph:company-epv:valuation"),
        ],
      ),
    ],
  )


@callback(
  Output("graph:company-epv:distribution", "figure"),
  Input("table:company-epv", "cellClicked"),
  State("table:company-epv", "selectedRows"),
)
def update_graph(cell: dict[str, str | int], row: list[dict]):
  if not (cell and row) or cell["colId"] != "distribution":
    return no_update

  params = [float(num) for num in row[0]["parameters"].split(", ")]

  dist = make_distribution(cast(str, cell["value"]), params)
  x, y = plot_pdf(dist)

  fig = px.line(x=x, y=y, title="Probability density function")

  fig.update_layout(xaxis_title="Value", yaxis_title="Density")

  return fig


@callback(
  Output("graph:company-epv:valuation", "figure"),
  Input("button:company-epv:simulate", "n_clicks"),
  State("table:company-epv", "rowData"),
  # State("table:company-valuation:correlation", "rowData"),
  State("location:app", "pathname"),
)
def monte_carlo(
  n_clicks: int, row_data: list[dict], corr_mat: list[dict], pathname: str
):
  if not n_clicks:
    return no_update

  company_id = pathname.split("/")[2]
  currency = get_currency(company_id)
  if currency is None:
    return no_update

  table = f"{company_id}_{currency}"
  columns = ["liquid_assets", "debt", "weighted_average_shares_outstanding_basic"]
  query = f"SELECT {', '.join(columns)} FROM '{table}' WHERE months = 12 ORDER BY date DESC LIMIT 1"
  valuation_items = fetch_sqlite("financials.db", query)
  if valuation_items is None:
    return no_update

  dcf = np.array(valuation_items).T
  dcf = dcf[:, :2]

  n = 1000

  df = pd.DataFrame.from_records(row_data)
  epv_inputs = df[["distribution", "parameters"]]
  max_revenue = df["revenue"].max()
  # corr_arr = pd.DataFrame.from_records(corr_mat).drop("factor", axis=1).to_numpy()

  corr_arr = np.eye(len(factors))
  # for pair, value in correlation.items():
  #  ix = (Factors[pair[0]].value, Factors[pair[1]].value)
  #  corr_mat[ix] = value

  corr_mat_psd = nearest_postive_definite_matrix(corr_arr)
  corr_mat_psd = ot.CorrelationMatrix(len(factors), corr_mat_psd.flatten())
  copula = ot.NormalCopula(corr_mat_psd)

  variables = [
    make_distribution(dist, params)
    for dist, params in zip(epv_inputs["distribution"], epv_inputs["parameters"])
  ]
  composed_distribution = ot.ComposedDistribution(variables, copula)
  sample = np.array(composed_distribution.getSample(n))

  args = np.array([[max_revenue]]).repeat(n, 0)

  args = np.concatenate((args, sample), axis=1)
  epv = np.apply_along_axis(lambda x: earnings_power(*x), 1, args)

  cash, debt, shares = valuation_items
  price = epv + cash - debt / shares

  fig = px.histogram(price)

  return fig
