# from dataclasses import dataclass
from datetime import date as Date
import json
from typing import cast, Annotated, Literal, Optional, TypeAlias  # , TypedDict
from typing_extensions import TypedDict

from pydantic import (
  BaseModel,
  PlainSerializer,
  ValidationInfo,
  field_serializer,
  field_validator,
)
from pandas import DatetimeIndex
from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index, Object

Scope: TypeAlias = Literal['annual', 'quarterly']
Quarter: TypeAlias = Literal['Q1', 'Q2', 'Q3', 'Q4']
FiscalPeriod: TypeAlias = Literal['FY'] | Quarter


class Meta(TypedDict):
  id: str
  scope: Scope
  date: Date
  period: FiscalPeriod
  fiscal_end: str
  currency: list[str]


class Value(TypedDict, total=False):
  value: float | int
  unit: str


class Interval(BaseModel):
  start_date: Date
  end_date: Date
  months: int

  @field_serializer('start_date', 'end_date')
  def serialize_date(self, date: Date):
    return date.strftime('%Y-%m-%d')


class Instant(BaseModel):
  instant: Date

  @field_serializer('instant')
  def serialize_date(self, date: Date):
    return date.strftime('%Y-%m-%d')


class Member(Value):
  dim: str


class Item(Value, total=False):
  period: Instant | Interval
  members: Optional[dict[str, Member]]


def item_serializer(v: Item):
  obj = {'value': v['value'], 'unit': v['unit'], 'period': v['period'].model_dump()}

  if (members := v.get('members')) is not None:
    obj['members'] = members

  json.dumps(obj)


SerializedItem = Annotated[Item, PlainSerializer(item_serializer)]
FinData: TypeAlias = dict[str, list[Item]]


class RawFinancials(BaseModel):
  id: Optional[str] = None
  scope: Scope
  date: Date
  period: FiscalPeriod
  fiscal_end: Optional[str] = None
  currency: set[str]
  data: FinData  # dict[str, list[SerializedItem]]

  @field_validator('currency', mode='before')
  @classmethod
  def validate_currency(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = set(json.loads(value))
      except json.JSONDecodeError:
        raise ValueError(f'{info.field_name} must be a valid JSON array string')
      return parsed_value

    return value

  @field_serializer('date')
  def serialize_date(self, date: Date):
    return date.strftime('%Y-%m-%d')

  @field_serializer('currency')
  def serialize_currency(self, currency: set[str]):
    return json.dumps(list(currency))

  @field_serializer('data')
  def serialize_data(self, data: FinData):
    obj = {}

    for k, items in data.items():
      items_ = []
      for item in items:
        item_: dict[str, dict[str, str | int] | float | int | str | Member] = {
          'period': item['period'].model_dump()
        }
        for field in ('value', 'unit', 'members'):
          if (value := item.get(field)) is not None:
            item_[field] = cast(float | int | str | Member, value)

        items_.append(item_)

      obj[k] = items_

    return json.dumps(obj)


class RawFinancialsFrame(DataFrameModel):
  id: Optional[str] = Field(unique=True)
  scope: str = Field(isin={'annual', 'quarterly'})
  date: Timestamp
  period: str = Field(isin={'FY', 'Q1'})
  fiscal_end: Optional[str]
  currency: Object
  data: Object


class StockSplit(BaseModel):
  date: Date
  stock_split_ratio: float


class FinancialsIndex(DataFrameModel):
  date: DatetimeIndex
  period: Index[FiscalPeriod]
  months: Index[int]

  class Config:
    multiindex_strict = True
    multiindex_unique = 'date'


class Recent(TypedDict):
  accessionNumber: list[str]
  filingDate: list[str]
  reportDate: list[str]
  acceptanceDateTime: list[str]
  act: list[str]
  form: list[str]
  fileNumber: list[str]
  filmNumber: list[str]
  items: list[str]
  size: list[int]
  isXBRL: list[int]
  isInlineXBRL: list[int]
  primaryDocument: list[str]
  primaryDocDescription: list[str]


class File(TypedDict):
  name: str
  filingCount: int
  filingFrom: str
  filingTo: str


class Filings(TypedDict):
  recent: Recent
  files: Optional[list[File]]


class CikEntry(TypedDict):
  cik_str: int
  ticker: str
  title: str


class CikFrame(DataFrameModel):
  cik_str: Index[int]
  ticker: str
  title: str
