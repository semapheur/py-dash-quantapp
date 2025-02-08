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
type FinData = dict[str, list[FinRecord]]


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
    self.counters: defaultdict[Literal["d", "i"], count[int]] = defaultdict(
      lambda: count(0)
    )

  def add_period(self, period: Instant | Duration) -> str:
    if isinstance(period, Instant):
      period_type: Literal["d", "i"] = "i"
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
  period: str
  members: dict[str, Member] | None


def item_dict(v: FinRecord):
  obj: FinRecord = {"value": v["value"], "unit": v["unit"], "period": v["period"]}

  members = v.get("members")
  if members is not None:
    obj["members"] = members

  return obj


class FinStatement(BaseModel):
  url: list[str] | None = None
  scope: Scope
  date: Date
  fiscal_period: FiscalPeriod
  fiscal_end: str
  currency: set[str]
  periods: dict[str, Instant | Duration]
  data: FinData

  @field_validator("url", mode="before")
  @classmethod
  def validate_url(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = json.loads(value)
        if not isinstance(parsed_value, list):
          raise ValueError(f"{info.field_name} must be a list. Invalid value: {value}")
      except json.JSONDecodeError:
        raise ValueError(
          f"{info.field_name} must be a valid JSON array string. Invalid value: {value}"
        )
      return parsed_value

    return value

  @field_validator("fiscal_end", mode="before")
  @classmethod
  def validate_fiscal_end(cls, value, info: ValidationInfo):
    try:
      # Add a dummy year to parse the date
      _ = dt.strptime(f"2000-{value}", "%Y-%m-%d")
    except ValueError:
      raise ValueError(
        f"{info.field_name} must in MM-DD format. Invalid value: {value}"
      )

    return value

  @field_validator("currency", mode="before")
  @classmethod
  def validate_currency(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = set(json.loads(value))
      except json.JSONDecodeError:
        raise ValueError(
          f"{info.field_name} must be a valid JSON array string. Invalid value: {value}"
        )
      return parsed_value

    return value

  @field_validator("periods", mode="before")
  @classmethod
  def validate_periods(cls, value, info: ValidationInfo):
    key_pattern = re.compile(r"^(i|d)\d+$")

    if isinstance(value, str):
      return parse_periods_json(value, key_pattern, info)

    for key, period in value.items():
      if not key_pattern.match(key):
        raise ValueError(
          f"{info.field_name} keys must be of the form 'iN' or 'dN'. Invlid key: {key}"
        )

      if not isinstance(period, (Instant, Duration)):
        raise ValueError(f"Invalid period format for key {key}: {period}")

    return value

  @field_validator("data", mode="before")
  @classmethod
  def validate_data(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        value = json.loads(value)
      except json.JSONDecodeError:
        raise ValueError(f"{info.field_name} must be a valid JSON dictionary string")

    if not isinstance(value, dict):
      raise ValueError(
        f"{info.field_name} must be a dictionary. Invalid value: {value}"
      )

    for key, records in value.items():
      if not isinstance(key, str):
        raise ValueError(f"{info.field_name} keys must be strings. Invalid key: {key}")

      if not isinstance(records, list):
        raise ValueError(
          f"{info.field_name} values must be lists. Invalid value: {records}"
        )

      for record in records:
        if not isinstance(record, dict):
          raise ValueError(
            f"Records under key '{key}' must be a dictionary. Invalid record: {record}"
          )

        if "value" not in record or "unit" not in record or "period" not in record:
          raise ValueError(
            f"Each record under '{key}' must contain 'value', 'unit', and 'period'. Invalid record: {record}"
          )

        if not isinstance(record["value"], (int, float)):
          raise ValueError(
            f"Record value must be a number. Invalid value: {record['value']}"
          )

        if not isinstance(record["unit"], str):
          raise ValueError(
            f"Record unit must be a string. Invalid unit: {record['unit']}"
          )

        if not isinstance(record["period"], str):
          raise ValueError(
            f"Record period must be a string. Invalid period: {record['period']}"
          )

        if "members" not in record:
          continue

        if not isinstance(record["members"], dict):
          raise ValueError(
            f"Record members must be a dictionary. Invalid members: {record['members']}"
          )

        for member_key, member_value in record["members"].items():
          if not isinstance(member_value, dict):
            raise ValueError(
              f"Record member '{member_key}' must be a dictionary. Invalid member: {member_value}"
            )

          if "value" not in member_value or "unit" not in member_value:
            raise ValueError(
              f"Record member '{member_key}' must contain 'value' and 'unit'. Invalid member: {member_value}"
            )

          if not isinstance(member_value["value"], (int, float)):
            raise ValueError(
              f"Record member '{member_key}' value must be a number. Invalid value: {member_value['value']}"
            )

          if not isinstance(member_value["unit"], str):
            raise ValueError(
              f"Record member '{member_key}' unit must be a string. Invalid unit: {member_value['unit']}"
            )

    return value

  @field_serializer("url")
  def serialize_url(self, url: list[str]):
    return url

  @field_serializer("date")
  def serialize_date(self, date: Date):
    return date.strftime("%Y-%m-%d")

  @field_serializer("currency")
  def serialize_currency(self, currency: set[str]):
    return list(currency)

  # @field_serializer('periods')
  # def serialize_periods(self, periods: set[Duration]):
  #  return json.dumps([interval.model_dump() for interval in periods])

  @field_serializer("data")
  @classmethod
  def serialize_data(cls, data: FinData):
    obj = {}
    for k, items in data.items():
      obj[k] = [
        {
          **{
            field: item[field]
            for field in ("period", "value", "unit", "members")
            if field in item
          },
        }
        for item in items
      ]
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


def parse_periods_json(
  periods: str, key_pattern: re.Pattern[str], info: ValidationInfo
) -> dict[str, Instant | Duration]:
  try:
    value = json.loads(periods)
  except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON string: {value}")

  if not isinstance(value, dict):
    raise ValueError(
      f"{info.field_name} must be a JSON dictionary string. Invalid value: {value}"
    )

  parsed_periods: dict[str, Instant | Duration] = {}
  for key, period in value.items():
    if not key_pattern.match(key):
      raise ValueError(
        f"{info.field_name} keys must be of the form 'iN' or 'dN'. Invalid key: {key}"
      )

    if not isinstance(value, dict):
      raise ValueError(f"Invalid period format for key {key}: {period}")

    if "instant" in period:
      parsed_periods[key] = Instant(**period)
    elif all(k in period for k in ["start_date", "end_date", "months"]):
      parsed_periods[key] = Duration(**period)
    else:
      raise ValueError(f"Invalid period format for key {key}: {period}")

  return parsed_periods
