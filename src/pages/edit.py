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
      className="flex flex-col",
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
