from dash import callback, dcc, no_update, Input, Output, MATCH

from lib.ticker.fetch import search_stocks


class TickerSelectAIO(dcc.Dropdown):
  @staticmethod
  def aio_id(id: str):
    return {"component": "TickerSelectAIO", "aio_id": id}

  def __init__(self, id: str, dropdown_props: dict | None = None):
    dropdown_props = dropdown_props.copy() if dropdown_props else {}
    dropdown_props.setdefault("placeholder", "Ticker")

    super().__init__(id=self.__class__.aio_id(id), **dropdown_props)

  @callback(Output(aio_id(MATCH), "options"), Input(aio_id(MATCH), "search_value"))
  def update_dropdown(search: str):
    if search is None or len(search) < 2:
      return no_update

    df = search_stocks(search)
    return df.to_dict("records")
