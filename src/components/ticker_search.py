from typing import cast

from dash import callback, dcc, html, no_update, Input, Output, State
from pandas import MultiIndex, DatetimeIndex

from lib.ticker.fetch import search_companies, company_currency
from lib.fin.fundamentals import load_fundamentals

from components.input import InputAIO

link_style = "block text-text hover:text-secondary"
nav_style = (
  "hidden peer-focus-within:flex hover:flex flex-col gap-1 "
  "absolute top-full left-1 p-1 bg-primary/50 backdrop-blur-xs shadow-sm z-1"
)
input_style = (
  "peer h-full w-full p-1 bg-primary text-text "
  "rounded-sm border border-text/10 hover:border-text/50 focus:border-secondary "
  "placeholder-transparent"
)
label_style = (
  "absolute left-1 -top-2 px-1 bg-primary text-text/50 text-xs "
  "peer-placeholder-shown:text-base peer-placeholder-shown:text-text/50 "
  "peer-placeholder-shown:top-1 peer-focus:-top-2 peer-focus:text-secondary "
  "peer-focus:text-xs transition-all"
)


def TickerSearch():
  return html.Div(
    className="relative h-full",
    children=[
      InputAIO(
        "ticker-search", "20vw", input_props={"placeholder": "Ticker", "type": "text"}
      ),
      html.Nav(id="nav:ticker-search", className=nav_style),
      dcc.Store(id="store:ticker-search:financials"),
      dcc.Store(id="store:ticker-search:id", data={}),
    ],
  )


@callback(
  Output("nav:ticker-search", "children"),
  Input(InputAIO.aio_id("ticker-search"), "value"),
)
def ticker_results(search: str) -> list[dict[str, str]]:
  if search is None or len(search) < 2:
    return no_update

  df = search_companies(search)
  if df is None:
    return []

  links = [
    dcc.Link(label, href=f"/company/{value}/overview", className=link_style)
    for label, value in zip(df["label"], df["value"])
  ]
  return links


@callback(
  Output("store:ticker-search:id", "data"),
  Input("location:app", "pathname"),
  State("store:ticker-search:id", "data"),
)
def id_store(path: str, id_store: dict[str, str]):
  path_split = path.split("/")

  if len(path_split) < 3 or path_split[1] != "company":
    return no_update

  new_id = path_split[2]
  old_id = id_store.get("id", "")

  if old_id == new_id:
    return no_update

  return {"id": new_id, "currency": company_currency(new_id)}


@callback(
  Output("store:ticker-search:financials", "data"),
  Input("store:ticker-search:id", "data"),
)
def fincials_store(id_store: dict[str, str]):
  id = id_store.get("id")
  if id is None:
    return no_update

  fundamentals = load_fundamentals(id, id_store["currency"])
  if fundamentals is None:
    return {}

  fundamentals.index = cast(MultiIndex, fundamentals.index).set_levels(
    cast(DatetimeIndex, cast(MultiIndex, fundamentals.index).levels[0]).strftime(
      "%Y-%m-%d"
    ),
    level="date",
  )
  fundamentals.reset_index(inplace=True)

  return fundamentals.to_dict("records")
