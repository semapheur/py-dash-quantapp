from dash import callback, dcc, no_update, Input, Output, MATCH
import uuid

from lib.db import search_tickers

class TickerSelectAIO(dcc.Dropdown):
  @staticmethod
  def _id(aio_id):
    return {
      'component': 'TickerSelectAIO',
      'aio_id': aio_id
    }
  
  def __init__(self, aio_id=None, dropdown_props=None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    dropdown_props = dropdown_props.copy() if dropdown_props else {}

    if 'placeholder' not in dropdown_props:
      dropdown_props['placeholder'] = 'Ticker'

    super().__init__(id=self.__class__._id(aio_id), **dropdown_props)

  @callback(
    Output(_id(MATCH), 'options'),
    Input(_id(MATCH), 'search_value')
  )
  def update_dropdown(search):
    if search is None or len(search) < 2:
      return no_update
    
    df = search_tickers('stock', search, False)
    options = [
      {'label': label, 'value': value}
      for label, value in zip(df['label'], df['value'])
    ]
    return options