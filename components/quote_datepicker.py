from typing import cast, Optional
import uuid

import pandas as pd
from dash import callback, dcc, Input, Output, MATCH

from components.quote_store import QuoteStoreAIO


class QuoteDatePickerAIO(dcc.DatePickerRange):
  @staticmethod
  def id(aio_id: str):
    return {"component": "QuoteDatePickerAIO", "aio_id": aio_id}

  def __init__(
    self, aio_id: Optional[str] = None, datepicker_props: Optional[dict] = None
  ):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    datepicker_props = datepicker_props.copy() if datepicker_props else {}

    datepicker_props["clearable"] = True
    datepicker_props["updatemode"] = "bothdates"

    if "display_format" not in datepicker_props:
      datepicker_props["display_format"] = "YYYY-M-D"

    super().__init__(id=self.__class__.id(aio_id), **datepicker_props)

  @callback(
    Output(id(MATCH), "min_date_allowed"),
    Output(id(MATCH), "max_date_allowed"),
    Input(QuoteStoreAIO.id(MATCH), "data"),
  )
  def update_datepicker(data):
    dates = cast(pd.DatetimeIndex, pd.to_datetime(data["date"], format="%Y-%m-%d"))

    return dates.min(), dates.max()
