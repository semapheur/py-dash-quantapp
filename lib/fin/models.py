from datetime import date as Date
import json
from typing import cast, Literal, Optional, TypeAlias
from typing_extensions import TypedDict

from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index, Object
from pydantic import (
  BaseModel,
  ValidationInfo,
  field_serializer,
  field_validator,
)

Scope: TypeAlias = Literal['annual', 'quarterly']
Quarter: TypeAlias = Literal['Q1', 'Q2', 'Q3', 'Q4']
Ttm: TypeAlias = Literal['TTM1', 'TTM2', 'TTM3']
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


def item_dict(v: Item):
  obj = {'value': v['value'], 'unit': v['unit'], 'period': v['period'].model_dump()}

  if (members := v.get('members')) is not None:
    obj['members'] = members

  return obj


# SerializedItem = Annotated[Item, PlainSerializer(item_serializer)]
FinData: TypeAlias = dict[str, list[Item]]


class FinStatement(BaseModel):
  url: Optional[str] = None
  scope: Scope
  date: Date
  period: FiscalPeriod
  fiscal_end: Optional[str] = None
  # periods: set[Interval]
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

  @field_validator('data', mode='before')
  @classmethod
  def validate_data(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = json.loads(value)
      except json.JSONDecodeError:
        raise ValueError(f'{info.field_name} must be a valid JSON dictionary string')
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

  def to_dict(self):
    return {
      'url': self.url,
      'scope': self.scope,
      'date': self.date.strftime('%Y-%m-%d'),
      'period': self.period,
      'fiscal_end': self.fiscal_end,
      'currency': list(self.currency),
      'data': {key: [item_dict(i) for i in items] for key, items in self.data.items()},
    }


class FinStatementFrame(DataFrameModel):
  url: Optional[str] = Field(unique=True)
  scope: str = Field(isin={'annual', 'quarterly'})
  date: Timestamp
  period: str = Field(isin={'FY', 'Q1'})
  fiscal_end: Optional[str]
  periods: Optional[Object]
  currency: Object
  data: Object


class FinancialsIndex(DataFrameModel):
  date: Index[Timestamp]
  period: Index[Literal['Q1', 'Q2', 'Q3', 'Q4', 'FY', 'TTM']]
  months: Index[int] = Field(ge=1, coerce=True)

  class Config:
    multiindex_coerce = True
    multiindex_unique = True


class CloseQuote(DataFrameModel):
  date: Index[Timestamp]
  close: float


class Quote(CloseQuote):
  open: Optional[float]
  high: Optional[float]
  low: Optional[float]
  volume: Optional[int] = Field(ge=0)


class StockSplit(BaseModel):
  date: Date
  stock_split_ratio: float
