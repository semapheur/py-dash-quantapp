# from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NotRequired, TypedDict, TypeAlias

Scope: TypeAlias = Literal['annual', 'quarterly']
Quarter: TypeAlias = Literal['Q1', 'Q2', 'Q3', 'Q4']
FiscalPeriod: TypeAlias = Literal['FY'] | Quarter


class Meta(TypedDict):
  id: str
  scope: Scope
  date: datetime
  period: FiscalPeriod
  fiscal_end: str
  currency: list[str]


class Value(TypedDict):
  value: float | int
  unit: str


class Interval(TypedDict):
  start_date: datetime | str
  end_date: datetime | str
  months: NotRequired[int]


class Instant(TypedDict):
  instant: datetime | str


class Member(Value):
  dim: str


class Item(Value):
  period: Interval | Instant
  members: NotRequired[dict[str, Member]]


class Financials(TypedDict):
  id: NotRequired[str]
  scope: Scope
  date: datetime
  period: FiscalPeriod
  fiscal_end: NotRequired[str]
  currency: list[str]
  data: dict[str, list[Item]]
