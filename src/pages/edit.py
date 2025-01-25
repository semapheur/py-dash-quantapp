from dash import (
  callback,
  ctx,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
  Patch,
)

from lib.db.lite import read_sqlite
from lib.ticker.fetch import stored_companies

register_page(__name__, path="/edit")

companies = stored_companies().to_dict("records")

sidebar = html.Aside(
  className="h-full flex flex-col",
  children=[
    dcc.Dropdown(id="dropdown:edit:company", options=companies, value=""),
    dcc.Dropdown(id="dropdown:edit:statement", options=[], value=""),
    html.Div(
      className="flex flex-col overflow-y-scroll",
      id="div:edit:items",
    ),
  ],
)

layout = html.Main(
  className="size-full grid grid-cols-[1fr_3fr]",
  children=[
    sidebar,
    html.Div(
      id="div:edit:statement",
      className="h-full",
    ),
  ],
)


@callback(
  Output("dropdown:edit:statement", "options"),
  Input("dropdown:edit:company", "value"),
)
def update_dropdown(value: str):
  if not value:
    return no_update

  query = f"""
    SELECT
      date || "(" || fiscal_period || ")" AS label,
      date || "_" || fiscal_period AS value
    FROM '{value}'
  """
  df = read_sqlite("statements.db", query)

  return df.to_dict("records")


@callback(
  Output("div:edit:items", "children"),
  Input("dropdown:edit:statement", "value"),
  State("dropdown:edit:company", "value"),
)
def update_items(value: str, company: str):
  if not (value or company):
    return no_update

  date, period = value.split("_")

  query = f"""
    SELECT key FROM '{company}', json_each(data)
    WHERE date = :date AND fiscal_period = :period
  """
  df = read_sqlite("statements.db", query, {"date": date, "period": period})

  buttons = []
  for item in df["key"]:
    buttons.append(
      html.Button(
        item,
        id={"type": f"dropdown:edit:{item}", "index": item},
        className="text-left text-xs",
      )
    )

  return buttons
