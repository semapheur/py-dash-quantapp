from typing import Optional
import uuid

from dash import callback, dcc, no_update, Input, Output, MATCH

from lib.ticker.fetch import search_stocks


class TickerSelectAIO(dcc.Dropdown):
  @staticmethod
  def id(aio_id: str):
    return {"component": "TickerSelectAIO", "aio_id": aio_id}

  def __init__(
    self, aio_id: Optional[str] = None, dropdown_props: Optional[dict] = None
  ):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    dropdown_props = dropdown_props.copy() if dropdown_props else {}
    dropdown_props.setdefault("placeholder", "Ticker")

    super().__init__(id=self.__class__.id(aio_id), **dropdown_props)

  @callback(Output(id(MATCH), "options"), Input(id(MATCH), "search_value"))
  def update_dropdown(search: str):
    if search is None or len(search) < 2:
      return no_update

    df = search_stocks(search)
    return df.to_dict("records")
