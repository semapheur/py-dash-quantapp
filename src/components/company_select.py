from dash import callback, dcc, no_update, Input, Output, State, MATCH

from lib.ticker.fetch import search_companies


class CompanySelectAIO(dcc.Dropdown):
  @staticmethod
  def aio_id(id: str):
    return {"component": "CompanySelectAIO", "aio_id": id}

  def __init__(self, id: str, dropdown_props: dict | None = None):
    dropdown_props = dropdown_props.copy() if dropdown_props else {}
    dropdown_props.setdefault("placeholder", "Company")

    super().__init__(id=self.__class__.aio_id(id), **dropdown_props)

  @callback(
    Output(aio_id(MATCH), "options"),
    Input(aio_id(MATCH), "search_value"),
    State(aio_id(MATCH), "options"),
    State(aio_id(MATCH), "multi"),
  )
  def update_dropdown(search: str, options: list[dict], multiple: bool):
    if search is None or len(search) < 2:
      return no_update

    df = search_companies(search, stored=False, limit=10)
    if df is None:
      return no_update

    if not (multiple and options):
      return df.to_dict("records")

    options += df.to_dict("records")

    return options
