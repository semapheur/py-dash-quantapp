from typing import Literal, Optional

from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index


class FinancialsIndex(DataFrameModel):
  date: Index[Timestamp]
  period: Index[Literal['Q1', 'Q2', 'Q3', 'Q4', 'FY']]
  months: Index[int] = Field(ge=1, coerce=True)

  class Config:
    multiindex_coerce = True
    multiindex_unique = True


class CloseQuote(DataFrameModel):
  date: Index[Timestamp]
  close: float


class OhlcvQuote(CloseQuote):
  open: float
  high: float
  low: float
  adjusted_close: Optional[str]
  volume: int = Field(ge=0.0)
