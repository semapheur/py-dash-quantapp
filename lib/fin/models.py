from typing import Literal, Optional

from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index


class FinancialsIndex(DataFrameModel):
  date: Index[Timestamp]
  period: Index[Literal['Q1', 'Q2', 'Q3', 'Q4', 'FY', 'TTM']]
  months: Index[int] = Field(ge=1, coerce=True)

  class Config:
    multiindex_coerce = True
    multiindex_unique = True


class Quote(DataFrameModel):
  date: Index[Timestamp]
  open: Optional[float]
  high: Optional[float]
  low: Optional[float]
  adjusted_close: Optional[float]
  close: Optional[float]
  volume: Optional[int] = Field(ge=0)
