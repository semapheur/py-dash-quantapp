from enum import Enum
import json
from typing import cast, Optional

from dash import (
  callback,
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
from ordered_set import OrderedSet
import pandas as pd
from pandera.typing import DataFrame, Series
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from components.dupont_chart import DupontChart

from components.stock_header import StockHeader
from lib.db.lite import fetch_sqlite, read_sqlite
from lib.fin.fundamentals import load_fundamentals
from lib.ticker.fetch import company_label

register_page(__name__, path_template="/company/<id>/overview", title=company_label)

radio_wrap_style = "flex divide-x rounded-sm shadow"
radio_input_style = (
  "appearance-none absolute inset-0 h-full cursor-pointer " "checked:bg-secondary/50"
)
radio_label_style = "relative px-1"

span_ids = (
  "return_on_equity",
  "net_profit_margin",
  "operating_profit_margin",
  "operating_margin:operating_income_loss",
  "operating_margin:revenue",
  "tax_burden",
  "net_income_loss",
  "tax_burden:pretax_income_loss",
  "interest_burden",
  "interest_burden:pretax_income_loss",
  "interest_burden:operating_income_loss",
  "asset_turnover",
  "asset_turnover:revenue",
  "asset_turnover:average_assets",
  "equity_multiplier",
  "equity_multiplier:average_assets",
  "average_equity",
)

dupont_items = {item.split(":")[1] if ":" in item else item for item in span_ids}


def create_row_data(fin_data: pd.DataFrame, cols: list[str]) -> list[dict]:
  fin_data.sort_values("date", inplace=True)

  query = f"""SELECT item, short, long FROM items 
    WHERE item IN {str(tuple(cols))}
  """
  # params = {'columns': str(tuple(cols))}
  items = read_sqlite("taxonomy.db", query)
  if items is None:
    raise ValueError("Taxonomy items not found")

  items.loc[:, "short"].fillna(items["long"], inplace=True)
  items.set_index("item", inplace=True)

  mask = fin_data["months"] == 12
  row_data = cast(pd.DataFrame, fin_data.loc[mask, cols].T.iloc[:, -1])
  row_data.index.name = "item"

  row_data["trend"] = ""
  for i in row_data.index:
    fig = px.line(fin_data.loc[:, i])
    fig.update_layout(
      showlegend=False,
      xaxis_visible=False,
      xaxis_showticklabels=False,
      yaxis_visible=False,
      yaxis_showticklabels=False,
      margin=dict(l=0, r=0, t=0, b=0),
      template="plotly_white",
    )
    row_data.at[i, "trend"] = fig

  row_data.reset_index(inplace=True)
  row_data.loc[:, "item"] = items.loc[row_data["item"].tolist(), "short"]

  return row_data.to_dict("records")


def sankey_color(sign: int, opacity: float = 1) -> str:
  return f"rgba(255,0,0,{opacity})" if sign == -1 else f"rgba(0,255,0,{opacity})"


def sankey_direction(sign: int) -> str:
  match sign:
    case -1:
      return "in"
    case 1:
      return "out"
    case _:
      raise ValueError("Invalid sign")


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


def trend_data(row: Series[float]):
  return {"x": row.index.tolist(), "y": row.values.tolist()}
  # fig = px.line(row)
  # fig.update_layout(
  #  showlegend=False,
  #  xaxis=dict(autorange="reversed"),
  #  xaxis_visible=False,
  #  xaxis_showticklabels=False,
  #  yaxis_visible=False,
  #  yaxis_showticklabels=False,
  #  margin=dict(l=0, r=0, t=0, b=0),
  #  template="plotly_white",
  # )
  # return fig


def financial_strength(fundamentals: pd.DataFrame) -> pd.DataFrame:
  items = (
    "cash_to_debt",
    "operating_cashflow_to_debt",
    "equity_to_assets",
    "debt_to_equity",
    "interest_coverage_ratio",
    "piotroski_f_score",
    "altman_z_score",
    "beneish_m_score",
    "economic_profit_spread",
  )
  query = f"""SELECT item, short, long FROM items 
    WHERE item IN {str(items)}
  """
  template = read_sqlite("taxonomy.db", query)
  if template is None:
    raise ValueError("Taxonomy items not found")

  template.loc[:, "short"].fillna(template["long"], inplace=True)

  fundamentals = cast(
    DataFrame,
    (fundamentals.xs(12, level="months").sort_index(ascending=False)),
  )
  cols = list(
    OrderedSet(
      OrderedSet(template["item"]).intersection(OrderedSet(fundamentals.columns))
    )
  )
  fundamentals = fundamentals[cols]

  period = "TTM" if "TTM" in cast(pd.MultiIndex, fundamentals.index).levels[1] else "FY"
  mask = fundamentals.index.get_level_values("period") == period
  current_date = fundamentals.loc[mask, :].index.get_level_values("date").max()

  fundamentals = fundamentals.xs("FY", level="period").T.reset_index()

  fundamentals["trend"] = fundamentals.iloc[:, 1:-1].apply(trend_data, axis=1)
  # for i, r in fundamentals.iterrows():
  #  fig = px.line(r.iloc[1:-1])
  #  fig.update_layout(
  #    showlegend=False,
  #    xaxis=dict(autorange="reversed"),
  #    xaxis_visible=False,
  #    xaxis_showticklabels=False,
  #    yaxis_visible=False,
  #    yaxis_showticklabels=False,
  #    margin=dict(l=0, r=0, t=0, b=0),
  #    template="plotly_white",
  #  )
  #  fundamentals.at[i, "trend"] = fig

  fundamentals.rename(columns={current_date: "current"}, inplace=True)
  fundamentals.loc[:, "index"] = fundamentals["index"].apply(
    lambda x: template.loc[template["item"] == x, "short"].iloc[0]
  )
  return fundamentals[["index", "trend", "current"]]


def layout(id: str | None = None):
  if id is None:
    return html.Main(
      className="size-full", children=[html.H1("404", className="md-auto")]
    )

  column_defs = [
    {"field": "item", "headerName": "Metric"},
    {"field": "trend", "headerName": "Trend", "cellRenderer": "TrendLine"},
    {"field": "current", "headerName": "Current (TTM)"},
  ]

  currency = fetch_sqlite(
    "ticker.db", "SELECT currency FROM company WHERE company_id = :id", {"id": id}
  )[0][0]
  df = load_fundamentals(id, currency)
  print(df)

  return html.Main(
    className="flex flex-col h-full overflow-y-scroll",
    children=[
      StockHeader(id),
      html.Div(
        className="grid gric-cols-2",
        children=[
          html.Div(
            className="flex flex-col",
            children=[
              html.Div(
                className="flex flex-col",
                children=[
                  html.H4("Financial Strength"),
                  dag.AgGrid(
                    id="table:company-overview:financial-strength",
                    columnDefs=column_defs,
                    rowData=financial_strength(df).to_dict("records"),
                    columnSize="autoSize",
                    style={"height": "100%"},
                  ),
                ],
              )
            ],
          )
        ],
      ),
      html.Div(
        id="div:stock:sankey",
        children=[
          html.Form(
            className="flex justify-around",
            children=[
              dcc.Dropdown(
                id={"type": "dropdown:stock:date", "id": "sankey"}, className="w-36"
              ),
              dcc.RadioItems(
                id="radio:stock:sheet",
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
                id={"type": "radio:stock:scope", "id": "sankey"},
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
          dcc.Graph(id="graph:stock:sankey", responsive=True),
        ],
      ),
      html.Div(
        children=[
          html.Form(
            className="flex justify-center",
            children=[
              dcc.Dropdown(
                id={"type": "dropdown:stock:date", "id": "dupont"}, className="w-36"
              ),
              dcc.RadioItems(
                id={"type": "radio:stock:scope", "id": "dupont"},
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
          DupontChart(),
        ]
      ),
    ],
  )


@callback(
  Output("table:company-overview:financial-strength", "rowData"),
  Input("store:ticker-search:financials", "data"),
)
def update_table(data: list[dict]):
  if not data:
    return no_update

  df = pd.DataFrame.from_records(data)
  cols = ["piotroski_f_score", "altman_z_score", "beneish_m_score"]

  return create_row_data(df, cols)


@callback(
  Output("graph:stock:sankey", "figure"),
  Input("radio:stock:sheet", "value"),
  Input({"type": "dropdown:stock:date", "id": "sankey"}, "value"),
  State({"type": "radio:stock:scope", "id": "sankey"}, "value"),
  State("store:ticker-search:financials", "data"),
)
def update_graph(sheet: str, date: str, scope: str, data: list[dict]):
  if not (date and data):
    return no_update

  fin = pd.DataFrame.from_records(data).set_index(["date", "months"]).xs((date, scope))

  engine = create_engine("sqlite+pysqlite:///data/taxonomy.db")
  query = text(
    """SELECT 
    sankey.item, items.short, items.long, sankey.color, sankey.links FROM sankey 
    LEFT JOIN items ON sankey.item = items.item
      WHERE sankey.sheet = :sheet
  """
  ).bindparams(sheet=sheet)

  with engine.connect().execution_options(autocommit=True) as con:
    tmpl = pd.read_sql(query, con=con)

  tmpl = tmpl.loc[tmpl["item"].isin(set(fin.index))]
  tmpl.loc[:, "links"] = tmpl["links"].apply(lambda x: json.loads(x))
  tmpl.loc[:, "short"].fillna(tmpl["long"], inplace=True)

  Nodes = Enum("Node", tmpl["item"].tolist(), start=0)

  sources = []
  targets = []
  values = []
  link_colors = []
  node_colors = []

  for item, node_color, links in zip(tmpl["item"], tmpl["color"], tmpl["links"]):
    if not node_color:
      node_color = sankey_color(np.sign(fin.loc[item]))

    node_colors.append(node_color)

    if not links:
      continue

    for key, value in links.items():
      if key not in set(tmpl["item"]):
        continue

      link_value = fin.loc[value.get("value", key)]

      sign = value.get("sign", np.sign(link_value))
      if sign != np.sign(link_value):
        continue

      values.append(np.abs(link_value))

      if not (direction := value.get("direction")):
        direction = sankey_direction(np.sign(link_value))

      if direction == "in":
        source = Nodes[key].value
        target = Nodes[item].value
      else:
        source = Nodes[item].value
        target = Nodes[key].value

      sources.append(source)
      targets.append(target)

      if not (color := value.get("color")):
        color = sankey_color(np.sign(link_value), 0.3)

      link_colors.append(color)

  fig = go.Figure(
    data=[
      go.Sankey(
        node=dict(
          pad=15,
          thickness=10,
          line=dict(color="black", width=0.5),
          label=tmpl["short"].tolist(),
          color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
      )
    ]
  )

  return fig


@callback(
  [Output("span:dupont-chart:" + span_id, "children") for span_id in span_ids],
  Input({"type": "dropdown:stock:date", "id": "dupont"}, "value"),
  State({"type": "radio:stock:scope", "id": "dupont"}, "value"),
  State("store:ticker-search:financials", "data"),
)
def update_dupont(date: str, scope: int, data: list[dict]):
  if not (date and data):
    return no_update

  fin = pd.DataFrame.from_records(data).set_index(["date", "months"]).xs((date, scope))

  return tuple(f"{fin.at[span_id.split(":")[-1]]:.3G}" for span_id in span_ids)


@callback(
  Output({"type": "dropdown:stock:date", "id": MATCH}, "options"),
  Output({"type": "dropdown:stock:date", "id": MATCH}, "value"),
  Input({"type": "radio:stock:scope", "id": MATCH}, "value"),
  Input("store:ticker-search:financials", "data"),
)
def update_dropdown(scope: str, data: list[dict]):
  if not data:
    return no_update

  fin = (
    pd.DataFrame.from_records(data)
    .set_index(["date", "months"])
    .xs(scope, level=1)
    .sort_index(ascending=False)
  )
  return list(fin.index.get_level_values(0)), ""
