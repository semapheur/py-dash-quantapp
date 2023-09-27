#from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, TypedDict

class Meta(TypedDict):
  id: str
  scope: Literal['annual', 'quarterly']
  date: datetime
  fiscal_end: str

class Value(TypedDict):
  value: float|int
  unit: str

class Interval(TypedDict):
  start_date: datetime
  end_date: datetime

class Instant(TypedDict):
  instant: datetime

class Member(Value):
  dim: str

class Item(Value):
  period: Interval|Instant
  members: Optional[dict[str, Member]]

class Financials:
  meta: Meta
  data: dict[str, list[Item]]