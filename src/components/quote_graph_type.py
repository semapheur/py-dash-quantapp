from dash import dcc


class QuoteGraphTypeAIO(dcc.Dropdown):
  @staticmethod
  def aio_id(id):
    return {"component": "QuoteGraphTypeAIO", "aio_id": id}

  def __init__(self, id: str | None = None, dropdown_props=None):
    dropdown_props = dropdown_props.copy() if dropdown_props else {}

    dropdown_props["options"] = [
      {"label": "Line", "value": "line"},
      {"label": "Candlestick", "value": "candlestick"},
    ]
    dropdown_props["clearable"] = False

    if "value" not in dropdown_props:
      dropdown_props["value"] = "line"

    super().__init__(id=self.__class__.aio_id(id), **dropdown_props)
