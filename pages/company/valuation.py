from enum import Enum
from typing import cast, Optional

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
  MATCH,
)
import dash_ag_grid as dag
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from components.company_header import CompanyHeader
from lib.fin.dcf import (
  discount_cashflow,
  make_distribution,
  nearest_postive_definite_matrix,
  terminal_value,
)
from lib.ticker.fetch import company_label

register_page(__name__, path_template="/company/<id>/valuation", title=company_label)

modal_style = (
  "relative left-1/2 top-1/2 " "-translate-x-1/2 -translate-y-1/2 rounded-md"
)

distributions = ["Normal", "Skewnormal", "Triangular", "Uniform"]

factors = {
  "years": {"initial": ("Uniform", "5, 10"), "terminal": ("∞", "")},
  "revenue_growth": {
    "initial": ("Normal", "0.15, 0.05"),
    "terminal": ("Normal", "0.02, 0.001"),
  },
  "operating_profit_margin": {
    "initial": ("Normal", "0.10, 0.02"),
    "terminal": ("Normal", "0.10, 0.02"),
  },
  "tax_rate": {
    "initial": ("Normal", "0.25, 0.05"),
    "terminal": ("Normal", "0.25, 0.05"),
  },
  "reinvestment_rate": {
    "initial": ("Normal", "0.10, 0.05"),
    "terminal": ("Normal", "0.10, 0.05"),
  },
  "risk_free_rate": {
    "initial": ("Normal", "0.04, 0.01"),
    "terminal": ("Normal", "0.02, 0.05"),
  },
  "yield_spread": {
    "initial": ("Normal", "0.02, 0.01"),
    "terminal": ("Normal", "0.01, 0.005"),
  },
  "equity_risk_premium": {
    "initial": ("Normal", "0.02, 0.01"),
    "terminal": ("Normal", "0.01, 0.005"),
  },
  "equity_to_capital": {
    "initial": ("Normal", "0.5, 0.3"),
    "terminal": ("Normal", "0.5, 0.3"),
  },
  "beta": {
    "initial": ("Normal", "0.5, 0.3"),
    "terminal": ("Normal", "0.5, 0.3"),
  },
}
corr_factors: list[str] = list(factors.keys())[1:]
corr_headers = [f.replace("_", " ").capitalize() for f in corr_factors]

Factors = Enum("Factors", list(factors.keys()), start=0)

correlation = {
  ("risk_free_rate", "yield_spread"): 0.9,
  ("risk_free_rate", "equity_risk_premium"): 0.9,
  ("equity_risk_premium", "revenue_growth"): 0.4,
  ("reinvestment_rate", "operating_margin"): 0.8,
  ("yield_spread", "operating_margin"): 0.8,
}


def layout(id: Optional[str] = None):
  dcf_columns = [
    {
      "field": "factor",
      "headerName": "Factor",
      "editable": False,
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
    },
    {
      "headerName": "Phase 1",
      "children": [
        {
          "field": "phase_1:distribution",
          "headerName": "Distribution",
          "cellEditor": "agSelectCellEditor",
          "cellEditorParams": {"values": distributions},
        },
        {
          "field": "phase_1:parameters",
          "headerName": "Parameters",
          "cellEditor": {"function": "ParameterInput"},
        },
      ],
    },
    {
      "headerName": "Terminal Phase",
      "children": [
        {
          "field": "terminal:distribution",
          "headerName": "Distribution",
          "pinned": "right",
          "lockPinned": True,
          "cellClass": "lock-pinned",
          "cellEditor": "agSelectCellEditor",
          "cellEditorParams": {"values": distributions},
        },
        {
          "field": "terminal:parameters",
          "headerName": "Parameters",
          "pinned": "right",
          "lockPinned": True,
          "cellClass": "lock-pinned",
          "cellEditor": {"function": "ParameterInput"},
        },
      ],
    },
  ]

  dcf_rows = [
    {
      "factor": k.replace("_", " ").capitalize(),
      "phase_1:distribution": factors[k]["initial"][0],
      "phase_1:parameters": factors[k]["initial"][1],
      "terminal:distribution": factors[k]["terminal"][0],
      "terminal:parameters": factors[k]["terminal"][1],
    }
    for k in factors
  ]

  id_mat = np.eye(len(corr_factors))

  corr_cols = [
    {
      "field": "factor",
      "headerName": "",
      "editable": False,
      "pinned": "left",
      "lockPinned": True,
      "cellClass": "lock-pinned",
    }
  ] + [
    {
      "field": f,
      "headerName": h,
      "type": "numericColumn",
      "valueFormatter": {"function": 'd3.format(".2f")(params.value)'},
      "cellEditor": {"function": "NumberInput"},
      "cellEditorParams": {"min": -1, "max": 1},
    }
    for f, h in zip(corr_factors, corr_headers)
  ]

  corr_rows = pd.DataFrame(
    id_mat, columns=corr_factors, index=pd.Index(corr_headers, name="factor")
  ).reset_index()

  return html.Main(
    className="h-full flex flex-col",
    children=[
      CompanyHeader(id) if id is not None else None,
      html.Div(
        children=[
          html.Button("Add", id="button:stock-valuation:dcf-add"),
          html.Button("Calc", id="button:stock-valuation:dcf-sim"),
          html.Button("Correlation", id="button:stock-valuation:correlation"),
        ]
      ),
      dag.AgGrid(
        id="table:stock-valuation:dcf",
        columnDefs=dcf_columns,
        rowData=dcf_rows,
        columnSize="autoSize",
        defaultColDef={"editable": True},
        dashGridOptions={"singleClickEdit": True, "rowSelection": "single"},
        style={"height": "100%"},
      ),
      html.Div(
        className="flex",
        children=[
          dcc.Graph(id="graph:stock-valuation:distribution"),
          dcc.Graph(id="graph:stock-valuation:dcf"),
        ],
      ),
      html.Dialog(
        id={"type": "dialog:stock-valuation", "id": "factor"},
        className=modal_style,
        children=[
          dcc.Graph(id="graph:stock-valuation:factor"),
          html.Button(
            "x",
            id={"type": "button:stock-valuation:close-modal", "id": "factor"},
            className="absolute top-0 left-2 text-3xl text-secondary hover:text-red-600",
          ),
        ],
      ),
      html.Dialog(
        id={"type": "dialog:stock-valuation", "id": "correlation"},
        className="w-3/4 pt-10 " + modal_style,
        children=[
          dag.AgGrid(
            id="table:stock-valuation:correlation",
            columnDefs=corr_cols,
            rowData=corr_rows.to_dict("records"),
            columnSize="autoSize",
            defaultColDef={"editable": True},
            dashGridOptions={"singleClickEdit": True},
          ),
          html.Button(
            "x",
            id={"type": "button:stock-valuation:close-modal", "id": "correlation"},
            className="absolute top-0 left-2 text-3xl text-secondary hover:text-red-600",
          ),
          html.Button(
            "Trend",
            id="button:stock-valuation:correlation-trend",
            className="absolute top-2 right-2",
          ),
        ],
      ),
    ],
  )


# DCF input table
@callback(
  Output("table:stock-valuation:dcf", "columnDefs"),
  Output("table:stock-valuation:dcf", "rowData"),
  Input("button:stock-valuation:dcf-add", "n_clicks"),
  State("table:stock-valuation:dcf", "columnDefs"),
  State("table:stock-valuation:dcf", "rowData"),
)
def update_table(n_clicks: int, cols: list[dict], rows: list[dict]):
  if not n_clicks:
    return no_update

  phase = len(cols) - 1
  rows_ = pd.DataFrame.from_records(rows)
  rows_.loc[:, f"phase_{phase}:distribution"] = "Normal"
  rows_.loc[:, f"phase_{phase}:parameters"] = ""

  cols.append(
    {
      "headerName": f"Phase {phase}",
      "children": [
        {
          "field": f"phase_{phase}:distribution",
          "headerName": "Distribution",
          "cellEditor": "agSelectCellEditor",
          "cellEditorParams": {"values": distributions},
        },
        {
          "field": f"phase_{phase}:parameters",
          "headerName": "Parameters",
          "cellEditor": {"function": "ParameterInput"},
        },
      ],
    }
  )
  return cols, rows_.to_dict("records")


@callback(
  Output("graph:stock-valuation:factor", "figure"),
  Input("table:stock-valuation:dcf", "cellClicked"),
  State("store:ticker-search:financials", "data"),
)
def update_factor_graph(cell: dict[str, str | int], fin_data: list[dict]):
  if not (cell and fin_data) or cell["colId"] != "factor":
    return no_update

  if cell["rowIndex"] == 0:
    return no_update

  factor = cast(str, cell["value"]).lower().replace(" ", "_")

  fin_df = pd.DataFrame.from_records(fin_data)
  fin_df.set_index("date", inplace=True)

  if factor == "revenue_growth":
    factor = "revenue"

  fin_s = fin_df.loc[fin_df["months"] == 12, factor]

  fig = make_subplots(rows=2, cols=1)

  fig.add_scatter(
    x=fin_s.index,
    y=fin_s.pct_change() if factor == "revenue" else fin_s,
    mode="lines",
    row=1,
    col=1,
  )
  fig.add_histogram(
    x=fin_s.pct_change() if factor == "revenue" else fin_s, row=2, col=1
  )
  fig.update_layout(title=cell["value"])
  return fig


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="dcf_factor_modal"),
  Output("table:stock-valuation:dcf", "cellClicked"),
  Input("table:stock-valuation:dcf", "cellClicked"),
  State({"type": "dialog:stock-valuation", "id": "factor"}, "id"),
)

clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="closeModal"),
  Output({"type": "dialog:stock-valuation", "id": MATCH}, "id"),
  Input({"type": "button:stock-valuation:close-modal", "id": MATCH}, "n_clicks"),
  State({"type": "dialog:stock-valuation", "id": MATCH}, "id"),
)

clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="modal"),
  Output("button:stock-valuation:correlation", "id"),
  Input("button:stock-valuation:correlation", "n_clicks"),
  State({"type": "dialog:stock-valuation", "id": "correlation"}, "id"),
  State("button:stock-valuation:correlation", "id"),
)


@callback(
  Output("table:stock-valuation:correlation", "rowData"),
  Input("button:stock-valuation:correlation-trend", "n_clicks"),
  State("store:ticker-search:financials", "data"),
)
def update_correlation(n_clicks: int, data: list[dict]):
  if not (data and n_clicks):
    return no_update

  df = pd.DataFrame.from_records(data)
  cols = corr_factors.copy()
  cols[0] = "revenue"
  df = df.loc[df["months"] == 12, cols]

  corr_mat = df.corr()
  corr_mat.index = pd.Index(corr_headers, name="factor")
  corr_mat.reset_index(inplace=True)
  corr_mat.rename(columns={"revenue": "revenue_growth"}, inplace=True)

  return corr_mat.to_dict("records")


@callback(
  Output("graph:stock-valuation:distribution", "figure"),
  Input("table:stock-valuation:dcf", "cellClicked"),
  State("table:stock-valuation:dcf", "selectedRows"),
)
def update_graph(cell: dict[str, str | int], row: list[dict]):
  if not (cell and row) or cell["colId"] == "factor":
    return no_update

  col_id = cast(str, cell["colId"]).split(":")

  if col_id[-1] != "distribution":
    return no_update

  params = [float(num) for num in row[0][f"{col_id[0]}:parameters"].split(", ")]

  dist = make_distribution(cast(str, cell["value"]), params)
  sample = np.array(dist.getSample(1000)).flatten()

  return px.histogram(x=sample, marginal="box")


@callback(
  Output("graph:stock-valuation:dcf", "figure"),
  Input("button:stock-valuation:dcf-sim", "n_clicks"),
  State("table:stock-valuation:dcf", "rowData"),
  State("table:stock-valuation:correlation", "rowData"),
  State("store:ticker-search:financials", "data"),
)
def monte_carlo(
  n_clicks: int, dcf_input: list[dict], corr_mat: list[dict], fin_data: list[dict]
):
  if not (n_clicks and fin_data):
    return no_update

  n = 1000

  fin_df = pd.DataFrame.from_records(fin_data)
  fin_df = fin_df.set_index(["date", "months"]).sort_index(level="date")

  mask = (slice(None), 12)
  revenue = fin_df.loc[mask, "revenue"].iloc[-1]

  dcf_df = pd.DataFrame.from_records(dcf_input)

  corr_arr = pd.DataFrame.from_records(corr_mat).drop("factor", axis=1).to_numpy()

  # corr_mat = np.eye(len(factors))
  # for pair, value in correlation.items():
  #  ix = (Factors[pair[0]].value, Factors[pair[1]].value)
  #  corr_mat[ix] = value

  corr_mat_psd = nearest_postive_definite_matrix(corr_arr)
  corr_mat_psd = ot.CorrelationMatrix(len(factors), corr_mat_psd.flatten())
  copula = ot.NormalCopula(corr_mat_psd)

  dcf = np.array([[1, float(revenue), 0.0]]).repeat(n, 0)
  phases = (len(dcf_df.columns) - 3) // 2
  for p in range(1, phases):
    dcf_df.loc[:, f"phase_{p}:parameters"] = dcf_df[f"phase_{p}:parameters"].apply(
      lambda x: [float(num) for num in x.split(", ")]
    )

    variables = [
      make_distribution(dist, params)
      for dist, params in zip(
        dcf_df[f"phase_{p}:distribution"], dcf_df[f"phase_{p}:parameters"]
      )
    ]
    composed_distribution = ot.ComposedDistribution(variables, copula)
    sample = np.array(composed_distribution.getSample(n))
    args = np.concatenate((dcf[:, :2], sample), axis=1)
    dcf += np.apply_along_axis(lambda x: discount_cashflow(*x), 1, args)

  # terminal value
  dcf_df.loc[1:, "terminal:parameters"] = dcf_df.loc[1:, "terminal:parameters"].apply(
    lambda x: [float(num) for num in x.split(", ")]
  )
  variables = [
    make_distribution(dist, params)
    for dist, params in zip(
      dcf_df.loc[1:, "terminal:distribution"],
      dcf_df.loc[1:, "terminal:parameters"],
    )
  ]
  corr_mat_psd = nearest_postive_definite_matrix(corr_arr[1:, 1:])
  corr_mat_psd = ot.CorrelationMatrix(len(factors) - 1, corr_mat_psd.flatten())
  copula = ot.NormalCopula(corr_mat_psd)
  composed_distribution = ot.ComposedDistribution(variables, copula)

  sample = np.array(composed_distribution.getSample(n))
  args = np.concatenate((dcf[:, 1].reshape(-1, 1), sample), axis=1)
  tv = np.apply_along_axis(lambda x: terminal_value(*x), 1, args)

  dcf = dcf[:, 2] + tv

  cash = fin_df.loc[mask, "liquid_assets"].iloc[-1]
  debt = fin_df.loc[mask, "debt"].iloc[-1]
  shares = fin_df.loc[mask, "weighted_average_shares_outstanding_diluted"].iloc[-1]
  price = dcf + cash - debt / shares

  fig = px.histogram(price)

  return fig
