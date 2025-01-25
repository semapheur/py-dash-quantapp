from typing import cast, Any
from ordered_set import OrderedSet

from dash import (
  callback,
  dcc,
  html,
  no_update,
  Output,
  Input,
  # register_page
)
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Sign
import pandas as pd
from pandera.typing import DataFrame

from components.sparklines import make_sparkline
from components.company_header import CompanyHeader
from lib.db.lite import read_sqlite
# from lib.ticker.fetch import company_label

# register_page(__name__, path_template="/company/<id>/financials", title=company_label)

radio_wrap_style = "flex divide-x rounded-xs shadow-sm"
radio_input_style = (
  "appearance-none absolute inset-0 h-full cursor-pointer checked:bg-secondary/50"
)
radio_label_style = "relative px-1"


def style_table(index: pd.Series, tmpl: pd.DataFrame) -> list[dict]:
  styling: list[dict] = [
    {"if": {"column_id": "Trend"}, "font_family": "Sparks-Dotline-Extrathick"},
    {"if": {"column_id": "index"}, "textAlign": "left", "paddingLeft": "2rem"},
  ]

  for level in tmpl["level"].unique():
    items = tmpl.loc[tmpl["level"] == level, "short"]
    row_ix = [index[index == i].index[0] for i in items]

    styling.append(
      {
        "if": {
          "row_index": row_ix,
          "column_id": "index",
        },
        "paddingLeft": f"{level + 1}rem",
        "fontWeight": "bold" if level == 0 else "normal",
        "borderBottom": "1px solid rgb(var(--color-text))" if level == 0 else None,
      }
    )

  return styling


def format_columns(columns: list[str], index: str) -> list[dict[str, Any]]:
  return [
    {
      "name": c,
      "id": c,
      "type": "text" if c == index else "numeric",
      "format": None if c == index else Format(group=True, sign=Sign.parantheses),
    }
    for c in columns
  ]


def layout(id: str | None = None):
  return html.Main(
    className="flex flex-col h-full",
    children=[
      CompanyHeader(id) if id is not None else None,
      html.Div(
        className="flex justify-around",
        children=[
          dcc.RadioItems(
            id="radio:stock-financials:sheet",
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
            id="radio:stock-financials:scope",
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
      html.Div(
        id="div:stock-financials:table-wrap", className="flex-1 overflow-x-hidden p-2"
      ),
    ],
  )


@callback(
  Output("div:stock-financials:datatable-wrap", "children"),
  Input("store:ticker-search:financials", "data"),
  Input("radio:stock-financials:sheet", "value"),
  Input("radio:stock-financials:scope", "value"),
)
def update_table(data: list[dict], sheet: str, scope: str):
  if not data:
    return no_update

  query = """SELECT 
    s.item, items.short, items.long, s.level FROM statement AS s 
    LEFT JOIN items ON s.item = items.item
    WHERE s.sheet = :sheet
  """
  param = {"sheet": sheet}
  tmpl = read_sqlite("taxonomy.db", query, param)

  tmpl.loc[:, "short"].fillna(tmpl["long"], inplace=True)
  labels = {k: v for k, v in zip(tmpl["item"], tmpl["short"])}

  fin = cast(
    pd.DataFrame,
    (
      pd.DataFrame.from_records(data)
      .set_index(["date", "months"])
      .xs(scope, level=1)
      .sort_index(ascending=False)
    ),
  )
  cols = list(
    OrderedSet(OrderedSet(tmpl["item"]).intersection(OrderedSet(fin.columns)))
  )
  fin = fin[cols]
  fin.rename(columns=labels, inplace=True)
  tmpl = tmpl.loc[tmpl["item"].isin(cols)]

  fin = fin.T.reset_index()
  fin.insert(1, "Trend", make_sparkline(cast(DataFrame, fin[fin.columns[1:]])))

  tooltips = [{"index": {"type": "markdown", "value": long}} for long in tmpl["long"]]

  return DataTable(
    fin.to_dict("records"),
    columns=format_columns(fin.columns.to_list(), fin.columns[0]),
    style_header={"fontWeight": "bold"},
    style_data_conditional=style_table(fin["index"], tmpl),
    fixed_columns={"headers": True, "data": 1},
    fixed_rows={"headers": True},
    tooltip_data=tooltips,
  )
