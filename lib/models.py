from datetime import datetime
from typing import Optional, TypedDict

from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index


class OHLCV(TypedDict, total=False):
  date: int | str | datetime
  open: Optional[float]
  high: Optional[float]
  low: Optional[float]
  close: float
  volume: Optional[float]


class OhlcFrame(DataFrameModel):
  date: Index[Timestamp]
  open: float
  high: float
  low: float
  close: float


class OhlcvFrame(OhlcFrame):
  volume: float = Field(gt=0)
