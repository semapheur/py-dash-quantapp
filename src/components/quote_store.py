import asyncio
from functools import partial

from dash import callback, dcc, no_update, Input, Output, MATCH

from lib.morningstar.ticker import Stock
from lib.fin.quote import load_ohlcv
from components.ticker_select import TickerSelectAIO


class QuoteStoreAIO(dcc.Store):
  @staticmethod
  def aio_id(id: str):
    return {"component": "QuoteStoreAIO", "aio_id": id}

  def __init__(self, id: str, store_props: dict | None = None):
    store_props = store_props.copy() if store_props else {}
    if "storage_type" not in store_props:
      store_props["storage_type"] = "memory"

    super().__init__(id=self.__class__.aio_id(id), **store_props)

  @callback(
    Output(aio_id(MATCH), "data"),
    Input(TickerSelectAIO.aio_id(MATCH), "value"),
    background=True,
  )
  def update_store(self, id_currency: str):
    if not id_currency:
      return no_update

    id, currency = id_currency.split("|")
    fetcher = partial(Stock(id, currency).ohlcv)
    ohlcv = asyncio.run(load_ohlcv(id, "stock", fetcher))
    ohlcv.reset_index(inplace=True)
    return ohlcv.to_dict("list")
