from functools import partial
import uuid

from dash import callback, dcc, no_update, Input, Output, MATCH

from lib.morningstar.ticker import Stock
from lib.ticker.fetch import get_ohlcv
from components.ticker_select import TickerSelectAIO


class QuoteStoreAIO(dcc.Store):
  @staticmethod
  def _id(aio_id):
    return {'component': 'QuoteStoreAIO', 'aio_id': aio_id}

  def __init__(self, aio_id=None, store_props=None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    store_props = store_props.copy() if store_props else {}
    if 'storage_type' not in store_props:
      store_props['storage_type'] = 'memory'

    super().__init__(id=self.__class__._id(aio_id), **store_props)

  @callback(Output(_id(MATCH), 'data'), Input(TickerSelectAIO._id(MATCH), 'value'))
  def update_store(query):
    if not query:
      return no_update

    id, currency = query.split('|')
    fetcher = partial(Stock(id, currency).ohlcv)
    ohlcv = get_ohlcv(id, 'stock', fetcher)
    ohlcv.reset_index(inplace=True)
    return ohlcv.to_dict('list')
