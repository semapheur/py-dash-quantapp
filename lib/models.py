from datetime import datetime
from typing import Optional, TypedDict

class OHLCV(TypedDict):
  date: int|str|datetime
  open: Optional[float]
  high: Optional[float]
  low: Optional[float]
  close: float
  volume: Optional[float]