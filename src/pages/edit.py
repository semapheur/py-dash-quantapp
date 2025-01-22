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

from lib.ticker.fetch import stored_companies

register_page(__name__, path="/edit")

companies = stored_companies().to_dict("records")

layout = html.Main(
  className="size-full",
  children=[dcc.Dropdown(id="dropdown:edit:company", options=companies, value="")],
)
