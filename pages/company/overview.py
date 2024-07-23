import asyncio
from enum import Enum
from functools import partial
import json
from typing import cast

from dash import (
  callback,
  ctx,
  dcc,
  html,
  no_update,
  register_page,
  Output,
  Input,
  State,
)
import dash_ag_grid as dag
import numpy as np
from ordered_set import OrderedSet
import pandas as pd
from pandera.typing import DataFrame, Series
import plotly.graph_objects as go
from components.dupont_chart import DupontChart

from components.quote_graph import quote_volume_graph
from components.stock_header import StockHeader
from lib.db.lite import fetch_sqlite, get_table_columns, read_sqlite
from lib.morningstar.ticker import Stock
from lib.fin.fundamentals import load_fundamentals
from lib.fin.quote import load_ohlcv
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
  "operating_margin:income_loss_operating",
  "operating_margin:revenue",
  "tax_burden",
  "income_loss_net",
  "tax_burden:income_loss_pretax",
  "interest_burden",
  "interest_burden:income_loss_pretax",
  "interest_burden:income_loss_operating",
  "asset_turnover",
  "asset_turnover:revenue",
  "asset_turnover:average_assets",
  "equity_multiplier",
  "equity_multiplier:average_assets",
  "average_equity",
)

dupont_items = {item.split(":")[1] if ":" in item else item for item in span_ids}


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


def get_primary_security(id: str) -> str | None:
  query = """SELECT json_extract(primary_security, '$[0]') AS primary_security 
  FROM company WHERE company_id = :id"""

  ticker = fetch_sqlite("ticker.db", query, {"id": f"{id}"})

  if ticker is None:
    return None

  return ticker[0][0]


def get_currency(id: str) -> str | None:
  currency = fetch_sqlite(
    "ticker.db", "SELECT currency FROM company WHERE company_id = :id", {"id": f"{id}"}
  )

  if currency is None:
    return None

  return currency[0][0]


def trend_data(row: Series[float]):
  return {"x": row.index.tolist(), "y": row.values.tolist()}


def select_ratios(fundamentals: DataFrame, items: tuple[str, ...]) -> DataFrame | None:
  query = f"""SELECT item, short, long FROM items 
    WHERE item IN {str(items)}
  """
  template = read_sqlite("taxonomy.db", query)
  if template is None:
    return None

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
  fundamentals = cast(DataFrame, fundamentals[cols])

  period = "TTM" if "TTM" in cast(pd.MultiIndex, fundamentals.index).levels[1] else "FY"
  mask = fundamentals.index.get_level_values("period") == period
  current_date = fundamentals.loc[mask, :].index.get_level_values("date").max()

  fundamentals = cast(DataFrame, fundamentals.xs("FY", level="period").T.reset_index())

  fundamentals["trend"] = fundamentals.iloc[:, 1:-1].apply(trend_data, axis=1)

  fundamentals.rename(columns={current_date: "current"}, inplace=True)
  fundamentals.loc[:, "index"] = fundamentals["index"].apply(
    lambda x: template.loc[template["item"] == x, "short"].iloc[0]
  )
  return cast(DataFrame, fundamentals[["index", "trend", "current"]])


def date_dropdown_params(ix: pd.MultiIndex) -> tuple[list[dict[str, str]], str]:
  def priority_sort_key(pair):
    date, period = pair
    period_priority = {"TTM": 0, "FY": 1}
    return (date, period_priority.get(period, 2))

  date_level = ix.get_level_values("date")
  period_level = ix.get_level_values("period")

  date_period_pairs = pd.Series(list(zip(date_level, period_level)))

  date_options = [
    {
      "label": f"{date.strftime("%Y-%m-%d")} ({period})",
      "value": f"{date.strftime("%Y-%m-%d")}|{period}",
    }
    for date, period in zip(date_level, period_level)
  ]

  max_date_pair = max(date_period_pairs, key=priority_sort_key)
  date_value = f"{max_date_pair[0].strftime("%Y-%m-%d")}|{max_date_pair[1]}"

  return date_options, date_value


def performance_section(fundamentals: pd.DataFrame):
  ratios = {
    "financial_strength": (
      "cash_to_debt",
      "operating_cashflow_to_debt",
      "equity_to_assets",
      "debt_to_equity",
      "interest_coverage_ratio",
      "piotroski_f_score",
      "altman_z_score",
      "beneish_m_score",
      "economic_profit_spread",
    ),
    "profitability": (
      "gross_margin",
      "operating_margin",
      "net_margin",
      "free_cashflow_margin",
      "return_on_equity",
      "return_on_assets",
      "return_on_invested_capital",
      "return_on_capital_employed",
      "cash_return_on_invested_capital",
    ),
    "liquidity": (
      "current_ratio",
      "quick_ratio",
      "cash_ratio",
      "defensive_interval_ratio",
      "cash_conversion_cycle",
    ),
  }

  column_defs = [
    {"field": "index", "headerName": "Metric"},
    {
      "field": "trend",
      "headerName": "Trend",
      "cellRenderer": "TrendLine",
    },
    {
      "field": "current",
      "headerName": "Current (TTM)",
      "valueFormatter": {"function": 'd3.format(".2f")(params.value)'},
    },
  ]

  return html.Section(
    className="grid grid-cols-3 gap-2 h-full snap-start",
    children=[
      html.Div(
        className="flex flex-col h-full",
        children=[
          html.H4(k.replace("_", " ").capitalize()),
          dag.AgGrid(
            id=f"table:company-overview:{k.replace('_', '-')}",
            columnDefs=column_defs,
            rowData=select_ratios(fundamentals, v).to_dict("records"),
            columnSize="autoSize",
          ),
        ],
      )
      for k, v in ratios.items()
    ],
  )


def sankey_section(date_options: list[dict[str, str]], date_value: str):
  return html.Section(
    className="h-full snap-start",
    children=[
      html.H3("Sankey Chart"),
      html.Form(
        className="flex justify-around",
        children=[
          dcc.Dropdown(
            id={"type": "dropdown:company:date", "id": "sankey"},
            className="w-40",
            options=date_options,
            value=date_value,
          ),
          dcc.RadioItems(
            id="radio:company:sheet",
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
        ],
      ),
      dcc.Graph(id="graph:company:sankey", responsive=True),
    ],
  )


def dupont_section(date_options: list[dict[str, str]], date_value: str):
  return html.Section(
    className="h-full snap-start",
    children=[
      html.H3("Dupont Chart"),
      html.Form(
        className="flex justify-center",
        children=[
          dcc.Dropdown(
            id={"type": "dropdown:company:date", "id": "dupont"},
            className="w-40",
            options=date_options,
            value=date_value,
          ),
        ],
      ),
      DupontChart(),
    ],
  )


not_found = html.Main(
  className="size-full", children=[html.H1("404", className="md-auto")]
)


def layout(id: str | None = None):
  if id is None:
    return not_found

  currency = get_currency(id)
  if currency is None:
    return not_found

  fundamentals = load_fundamentals(id, currency)
  if fundamentals is None:
    return not_found

  query = """SELECT 
    ticker || ' (' || mic || ':' || currency || ')' AS label, 
    security_id || '|' || currency AS value 
  FROM stock WHERE company_id = :id
  """
  ticker_options = read_sqlite("ticker.db", query, {"id": id})
  ticker_value = f"{get_primary_security(id)}|{currency}"
  print(ticker_value)

  date_options, date_value = date_dropdown_params(
    cast(pd.MultiIndex, fundamentals.index)
  )

  return html.Main(
    className="grid grid-rows-[auto_1fr] h-full",
    children=[
      StockHeader(id),
      html.Div(
        className="h-full snap-y snap-mandatory overflow-y-scroll",
        children=[
          html.Section(
            className="h-full snap-start",
            children=[
              html.Form(
                className="grid grid-cols-[2fr_1fr_auto] gap-2 px-2 pt-2",
                children=[
                  dcc.Dropdown(
                    id="dropdown:company:quote",
                    options=cast(DataFrame, ticker_options).to_dict("records"),
                    value=ticker_value,
                  )
                ],
              ),
              dcc.Graph(
                id="graph:company:quote",
              ),
            ],
          ),
          performance_section(fundamentals),
          sankey_section(date_options, date_value),
          dupont_section(date_options, date_value),
        ],
      ),
    ],
  )


@callback(
  Output("graph:company:quote", "figure"),
  Input("dropdown:company:quote", "value"),
  Input("graph:company:quote", "relayoutData"),
  State("graph:company:quote", "figure"),
  background=True,
)
def update_quote(id_currency: str, relayout: dict, fig: go.Figure):
  if not id_currency:
    return no_update

  triggered_id = ctx.triggered_id
  if triggered_id == "dropdown:company:quote":
    id, currency = id_currency.split("|")
    ohlcv_fetcher = partial(Stock(id, currency).ohlcv)
    ohlcv = asyncio.run(load_ohlcv(id, "stock", ohlcv_fetcher, cols=["open", "close"]))

    if ohlcv is None:
      return no_update

    return quote_volume_graph(
      ohlcv.reset_index().to_dict("list"),
      "line",
      rangeselector=("1M", "6M", "YTD", "1Y", "All"),
    )

  # elif triggered_id == "graph:company:quote" and relayout and fig:
  #  return quote_graph_relayout(relayout, fig["data"], ["open", "close"])

  return no_update


def quote_graph_relayout(
  relayout: dict,
  data: dict[str, list[str | float | int]],
  cols: list[str],
  fig: go.Figure,
) -> go.Figure:
  if all(x in relayout.keys() for x in ["xaxis.range[0]", "xaxis.range[1]"]):
    fig = quote_graph_range(
      data, cols, fig, relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
    )
  elif "xaxis.autorange" in relayout.keys():
    fig["layout"]["xaxis"]["autorange"] = True
    fig["layout"]["yaxis"]["autorange"] = True

    for i in range(1, len(cols)):
      fig["layout"][f"yaxis{i+1}"]["autorange"] = True

  return fig


@callback(
  Output("graph:company:sankey", "figure"),
  Input("radio:company:sheet", "value"),
  Input({"type": "dropdown:company:date", "id": "sankey"}, "value"),
  State("location:app", "pathname"),
  background=True,
)
def update_sankey(sheet: str, date_period: str, slug: str):
  if not (date_period and slug):
    return no_update

  query = """SELECT 
    sankey.item, items.short, items.long, sankey.color, sankey.links FROM sankey 
    LEFT JOIN items ON sankey.item = items.item
      WHERE sankey.sheet = :sheet
  """

  template = read_sqlite("taxonomy.db", query, {"sheet": sheet})
  if template is None:
    return no_update

  id = slug.split("/")[2]
  currency = get_currency(id)
  if currency is None:
    return no_update

  table = f"{id}_{currency}"
  columns = get_table_columns("financials.db", [table])[table]

  sankey_columns = columns.intersection(template["item"].tolist())
  column_text = ",".join(sankey_columns)

  date, period = date_period.split("|")
  query = (
    f"""SELECT {column_text} FROM "{table}" WHERE date = :date AND period = :period"""
  )
  df = read_sqlite("financials.db", query, {"date": date, "period": period})

  if df is None:
    return no_update

  template = cast(DataFrame, template.loc[template["item"].isin(set(df.index))])
  template.loc[:, "links"] = template["links"].apply(lambda x: json.loads(x))
  template.loc[:, "short"].fillna(template["long"], inplace=True)

  Nodes = Enum("Node", template["item"].tolist(), start=0)

  sources = []
  targets = []
  values = []
  link_colors = []
  node_colors = []

  for item, node_color, links in zip(
    template["item"], template["color"], template["links"]
  ):
    if not node_color:
      node_color = sankey_color(np.sign(df.loc[item]))

    node_colors.append(node_color)

    if not links:
      continue

    for key, value in links.items():
      if key not in set(template["item"]):
        continue

      link_value = df.loc[value.get("value", key)]

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
          label=template["short"].tolist(),
          color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
      )
    ]
  )

  return fig


@callback(
  [Output("span:dupont-chart:" + span_id, "children") for span_id in span_ids],
  Input({"type": "dropdown:company:date", "id": "dupont"}, "value"),
  State("location:app", "pathname"),
  background=True,
)
def update_dupont(date_period: str, slug: str):
  if not (date_period and slug):
    return no_update

  def format_numbers(x: float | None):
    if pd.isna(x):
      return "N/A"

    return f"{x:.3g}"

  id = slug.split("/")[2]
  currency = get_currency(id)
  if currency is None:
    return no_update

  dupont_items = {span_id.split(":")[-1] for span_id in span_ids}

  date, period = date_period.split("|")
  where = f"WHERE date = '{date}' AND period = '{period}'"
  df = load_fundamentals(id, currency, dupont_items, where)

  if df is None:
    return no_update

  df.reset_index(drop=True, inplace=True)
  df.fillna(np.nan, inplace=True)
  return tuple(
    f"{format_numbers(df.at[0, item])}"
    if (item := si.split(":")[-1]) in dupont_items
    else "N/A"
    for si in span_ids
  )
