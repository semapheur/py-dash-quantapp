from collections import defaultdict
from datetime import date as Date, datetime as dt
from itertools import count
import json
import re
from typing import cast, Literal
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

type Scope = Literal["annual", "quarterly"]
type Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
type Ttm = Literal["TTM1", "TTM2", "TTM3"]
type FiscalPeriod = Literal["FY"] | Quarter


class SharePrice(TypedDict):
  share_price_close: float
  share_price_open: float
  share_price_high: float
  share_price_low: float
  share_price_average: float


class Meta(TypedDict):
  id: str
  scope: Scope
  date: Date
  fiscal_period: FiscalPeriod
  fiscal_end: str
  currency: list[str]


class Value(TypedDict, total=False):
  value: float | int
  unit: str


class Duration(BaseModel, frozen=True):
  start_date: Date
  end_date: Date
  months: int

  @field_serializer("start_date", "end_date")
  def serialize_date(self, date: Date):
    return date.strftime("%Y-%m-%d")


class Instant(BaseModel):
  instant: Date

  @field_serializer("instant")
  def serialize_date(self, date: Date):
    return date.strftime("%Y-%m-%d")


class FinPeriods:
  def __init__(self) -> None:
    self.periods: dict[str, Instant | Duration] = {}
    self.reverse_lookup: dict[Instant | Duration, str] = {}
    self.counters: defaultdict[str, count[int]] = defaultdict(lambda: count(0))

  def add_period(self, period: Instant | Duration) -> str:
    if isinstance(period, Instant):
      period_type = "i"
    elif isinstance(period, Duration):
      period_type = "d"
    else:
      raise ValueError(f"Unknown period type: {period}")

    if period in self.reverse_lookup:
      return self.reverse_lookup[period]

    key = f"{period_type}{next(self.counters[period_type])}"
    self.periods[key] = period
    self.reverse_lookup[period] = key
    return key

  def get_periods(self) -> dict[str, Instant | Duration]:
    return dict(sorted(self.periods.items()))


class Member(Value):
  dim: str


class FinRecord(Value, total=False):
  period: Instant | Duration
  members: dict[str, Member] | None


def item_dict(v: FinRecord):
  obj = {"value": v["value"], "unit": v["unit"], "period": v["period"].model_dump()}

  if (members := v.get("members")) is not None:
    obj["members"] = members

  return obj


type FinData = dict[str, list[FinRecord]]


class FinStatement(BaseModel):
  url: list[str] | None = None
  scope: Scope
  date: Date
  fiscal_period: FiscalPeriod
  fiscal_end: str
  # periods: dict[str, Instant | Duration]
  currency: set[str]
  data: FinData

  @field_validator("url", mode="before")
  @classmethod
  def validate_url(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = json.loads(value)
      except json.JSONDecodeError:
        raise ValueError(f"{info.field_name} must be a valid JSON array string")
      return parsed_value

    return value

  @field_validator("fiscal_end", mode="before")
  @classmethod
  def validate_fiscal_end(cls, value, info: ValidationInfo):
    pattern = r"(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    if not re.match(pattern, value):
      raise ValueError(f"{value} does not match the format -%m-%d")

    try:
      # Add a dummy year to parse the date
      _ = dt.strptime(f"2000-{value}", "%Y-%m-%d")
    except ValueError:
      raise ValueError(f"Invalid fiscal end: {value}")

    return value

  @field_validator("currency", mode="before")
  @classmethod
  def validate_currency(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = set(json.loads(value))
      except json.JSONDecodeError:
        raise ValueError(f"{info.field_name} must be a valid JSON array string")
      return parsed_value

    return value

  @field_validator("data", mode="before")
  @classmethod
  def validate_data(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = json.loads(value)
      except json.JSONDecodeError:
        raise ValueError(f"{info.field_name} must be a valid JSON dictionary string")
      return parsed_value

    return value

  @field_serializer("url")
  def serialize_url(self, url: list[str]):
    return url

  @field_serializer("date")
  def serialize_date(self, date: Date):
    return date.strftime("%Y-%m-%d")

  # @field_serializer('periods')
  # def serialize_periods(self, periods: set[Duration]):
  #  return json.dumps([interval.model_dump() for interval in periods])

  @field_serializer("currency")
  def serialize_currency(self, currency: set[str]):
    return list(currency)

  @field_serializer("data")
  @classmethod
  def serialize_data(cls, data: FinData):
    obj = {}

    for k, items in data.items():
      items_ = []
      for item in items:
        item_: dict[str, dict[str, str | int] | float | int | str | Member] = {
          "period": item["period"].model_dump()
        }
        for field in ("value", "unit", "members"):
          if (value := item.get(field)) is not None:
            item_[field] = cast(float | int | str | Member, value)

        items_.append(item_)

      obj[k] = items_

    return obj

  def to_dict(self):
    return {
      "url": self.url,
      "scope": self.scope,
      "date": self.date.strftime("%Y-%m-%d"),
      "fiscal_period": self.period,
      "fiscal_end": self.fiscal_end,
      "currency": list(self.currency),
      "data": {key: [item_dict(i) for i in items] for key, items in self.data.items()},
    }


class FinStatementFrame(DataFrameModel):
  url: Object | None
  scope: str = Field(isin={"annual", "quarterly"})
  date: Timestamp
  period: str = Field(isin={"FY", "Q1", "Q2", "Q3", "Q4"})
  fiscal_end: str | None
  # periods: Object | None
  currency: Object
  data: Object


class FinancialsIndex(DataFrameModel):
  date: Index[Timestamp]
  period: Index[Literal["Q1", "Q2", "Q3", "Q4", "FY", "TTM"]]
  months: Index[int] = Field(ge=1, coerce=True)
  fiscal_end_month: Index[int] = Field(ge=1, le=12, coerce=True)

  class Config:
    multiindex_coerce = True
    multiindex_unique = True


class CloseQuote(DataFrameModel):
  date: Index[Timestamp]
  close: float


class Quote(CloseQuote):
  open: float | None
  high: float | None
  low: float | None
  volume: int | None = Field(ge=0)


class StockSplit(BaseModel):
  date: Date
  stock_split_ratio: float
