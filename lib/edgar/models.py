#from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

class Meta(TypedDict):
  id: str
  scope: Literal['annual', 'quarterly']
  date: str
  fiscal_end: str

class Value(TypedDict):
  value: float|int
  unit: str

class Interval(TypedDict):
  start_date: str
  end_date: str

class Instant(TypedDict):
  instant: str

class Member(Value):
  dim: str

class Item(Value):
  period: Interval|Instant
  members: Optional[dict[str, Member]]

class Financials:
  meta: Meta
  data: dict[str, list[Item]]