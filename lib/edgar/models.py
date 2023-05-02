#from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel

class Meta(BaseModel):
  id: str
  scope: Literal['annual', 'quarterly']
  date: str
  fiscal_end: str

class Value(BaseModel):
  value: float|int
  unit: str

class Interval(BaseModel):
  start_date: str
  end_date: str

class Instant(BaseModel):
  instant: str

class Member(Value):
  dim: str

class Item(Value):
  period: Interval|Instant
  members: Optional[dict[str, Member]] = None

class Financials:
  meta: Meta
  data: dict[str, list[Item]]