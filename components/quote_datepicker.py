from typing import cast

import pandas as pd
from dash import callback, dcc, Input, Output, MATCH

from components.quote_store import QuoteStoreAIO


class QuoteDatePickerAIO(dcc.DatePickerRange):
  @staticmethod
  def aio_id(id: str):
    return {"component": "QuoteDatePickerAIO", "aio_id": id}

  def __init__(self, id: str, datepicker_props: dict | None = None):
    datepicker_props = datepicker_props.copy() if datepicker_props else {}

    datepicker_props["clearable"] = True
    datepicker_props["updatemode"] = "bothdates"

    if "display_format" not in datepicker_props:
      datepicker_props["display_format"] = "YYYY-M-D"

    super().__init__(id=self.__class__.aio_id(id), **datepicker_props)

  @callback(
    Output(aio_id(MATCH), "min_date_allowed"),
    Output(aio_id(MATCH), "max_date_allowed"),
    Input(QuoteStoreAIO.aio_id(MATCH), "data"),
  )
  def update_datepicker(data):
    dates = cast(pd.DatetimeIndex, pd.to_datetime(data["date"], format="%Y-%m-%d"))

    return dates.min(), dates.max()
