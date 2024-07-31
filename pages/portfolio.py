from dash import (
  ALL,
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
  Patch,
)

from components.ticker_select import TickerSelectAIO

register_page(__name__, path="/portfolio")

layout = html.Main(
  className="h-full",
  children=[
    TickerSelectAIO("portfolio:ticker-select", dropdown_props={"multi": True}),
  ],
)
