#from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, TypedDict

from attrs import define

class Meta(TypedDict):
  id: str
  scope: Literal['annual', 'quarterly']
  date: datetime
  period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4']
  fiscal_end: str
  currency: list[str]

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

@define
class Financials:
  id: Optional[str]
  scope: Literal['annual', 'quarterly']
  date: datetime
  period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4']
  fiscal_end: str
  currency: list[str]
  data: dict[str, list[Item]]