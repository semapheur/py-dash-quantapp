# from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, TypedDict, TypeAlias

from pydantic import BaseModel, field_serializer
from pandas import DatetimeIndex
from pandera import DataFrameModel
from pandera.typing import Index

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


class Value(TypedDict, total=False):
  value: float | int
  unit: str


class Interval(BaseModel):
  start_date: datetime
  end_date: datetime
  months: int

  @field_serializer('start_date', 'end_date')
  def serialize_date(self, date: datetime):
    return datetime.strftime(date, '%Y-%m-%d')


class Instant(BaseModel):
  instant: datetime

  @field_serializer('instant')
  def serialize_date(self, date: datetime):
    return datetime.strftime(date, '%Y-%m-%d')


class Member(Value):
  dim: str


class Item(Value, total=False):
  period: Interval | Instant
  members: Optional[dict[str, Member]]


FinData: TypeAlias = dict[str, list[Item]]


class Financials(BaseModel):
  id: Optional[str] = None
  scope: Scope
  date: datetime
  period: FiscalPeriod
  fiscal_end: Optional[str] = None
  currency: set[str]
  data: FinData


class StockSplit(TypedDict):
  date: datetime
  stock_split_ratio: float | int


class FinancialsIndex(DataFrameModel):
  date: DatetimeIndex
  period: Index[FiscalPeriod]
  months: Index[int]

  class Config:
    multiindex_strict = True
    multiindex_unique = 'date'
