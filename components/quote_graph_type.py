import uuid

from dash import dcc


class QuoteGraphTypeAIO(dcc.Dropdown):
  @staticmethod
  def id(aio_id):
    return {'component': 'QuoteGraphTypeAIO', 'aio_id': aio_id}

  def __init__(self, aio_id: str | None = None, dropdown_props=None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    dropdown_props = dropdown_props.copy() if dropdown_props else {}

    dropdown_props['options'] = [
      {'label': 'Line', 'value': 'line'},
      {'label': 'Candlestick', 'value': 'candlestick'},
    ]
    dropdown_props['clearable'] = False

    if 'value' not in dropdown_props:
      dropdown_props['value'] = 'line'

    super().__init__(id=self.__class__.id(aio_id), **dropdown_props)
