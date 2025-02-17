from datetime import date as Date, datetime as dt
import json
import re
from typing import cast, Literal
from typing_extensions import TypedDict

from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index
from pydantic import (
  BaseModel,
  NonNegativeInt,
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
  value: int | float
  unit: NonNegativeInt


class Duration(BaseModel, frozen=True):
  start_date: Date
  end_date: Date
  months: NonNegativeInt

  @field_serializer("start_date", "end_date")
  def serialize_date(self, date: Date):
    return int(date.strftime("%Y%m%d"))


class Instant(BaseModel, frozen=True):
  instant: Date

  @field_serializer("instant")
  def serialize_date(self, date: Date):
    return int(date.strftime("%Y%m%d"))


class FinPeriods(TypedDict):
  d: list[Duration]
  i: list[Instant]


class FinPeriodStore:
  def __init__(self) -> None:
    self.periods: dict[Literal["d", "i"], set[Instant | Duration]] = {
      "d": set(),
      "i": set(),
    }

  def add_period(self, period: Instant | Duration):
    if isinstance(period, Instant):
      period_type: Literal["d", "i"] = "i"
    elif isinstance(period, Duration):
      period_type = "d"
    else:
      raise ValueError(f"Unknown period type: {period}")

    self.periods[period_type].add(period)

  def get_periods(self) -> tuple[FinPeriods, dict[Instant | Duration, str]]:
    sorted_durations = sorted(
      cast(set[Duration], self.periods["d"]), key=lambda x: (x.start_date, x.end_date)
    )
    sorted_instant = sorted(
      cast(set[Instant], self.periods["i"]), key=lambda x: x.instant
    )

    reverse_lookup: dict[Instant | Duration, str] = {}

    for j, d in enumerate(sorted_durations):
      reverse_lookup[d] = f"d{j}"

    for j, i in enumerate(sorted_instant):
      reverse_lookup[i] = f"i{j}"

    fin_periods = FinPeriods(
      d=sorted_durations,
      i=sorted_instant,
    )

    return fin_periods, reverse_lookup


class UnitStore:
  def __init__(self) -> None:
    self.units: set[str] = set()

  def add_unit(self, unit: str):
    self.units.add(unit)

  def get_units(self) -> tuple[list[str], dict[str, int]]:
    sorted_units = sorted(self.units)
    reverse_lookup = {unit: i for i, unit in enumerate(sorted_units)}

    return sorted_units, reverse_lookup


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
  periods: FinPeriods
  units: list[str]
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

  @field_validator("date", mode="before")
  @classmethod
  def validate_date(cls, value, info: ValidationInfo):
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(
          f"As integer '{info.field_name}' must be in YYYYMMDD format. Invalid value: {value}"
        )

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

  @field_validator("units", mode="before")
  @classmethod
  def validate_units(cls, value, info: ValidationInfo):
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

    return value

  @field_serializer("url")
  def serialize_url(self, url: list[str]):
    return url

  @field_serializer("date")
  def serialize_date(self, date: Date):
    return int(date.strftime("%Y%m%d"))

  @field_serializer("currency")
  def serialize_currency(self, currency: set[str]):
    return list(currency)

  @field_serializer("data")
  @classmethod
  def serialize_data(cls, data: FinData):
    obj = {}
    for k, items in data.items():
      obj[k] = [
        {
          "period": item["period"],
          **{
            field: item[field]
            for field in ("value", "unit", "members")
            if field in item
          },
        }
        for item in items
      ]
    return obj

  def merge(self, other: object):
    if not isinstance(other, FinStatement):
      return NotImplemented

    if other.date != self.date and other.fiscal_period != self.fiscal_period:
      raise ValueError("Cannot merge statements with different dates or periods")

    if self.url is None:
      self.url = []

    if other.url is not None:
      new_urls = set(other.url).difference(set(self.url))
      if new_urls:
        self.url.extend(new_urls)

    self.currency.update(other.currency)

    common_items = set(self.data.keys()).intersection(other.data.keys())

    diff_periods: dict[str, Instant | Duration] = {}
    diff_periods_lookup: dict[Instant | Duration, str] = {}
    period_index_count = {
      "d": len(self.periods["d"]),
      "i": len(self.periods["i"]),
    }

    old_periods: dict[Literal["d", "i"], set[Instant | Duration]] = {
      "d": set(self.periods["d"]),
      "i": set(self.periods["i"]),
    }

    diff_units: list[str] = []
    unit_index_count = len(self.units)

    for item in common_items:
      for entry in other.data[item]:
        period_type: Literal["d", "i"] = cast(Literal["d", "i"], entry["period"][0])
        period_index = int(entry["period"][1:])
        period = other.periods[period_type][period_index]

        if period in old_periods[period_type]:
          continue

        new_key = diff_periods_lookup.get(period, "")
        if new_key not in diff_periods:
          new_key = f"{period_type}{period_index_count[period_type]}"
          period_index_count[period_type] += 1
          diff_periods[new_key] = period
          diff_periods_lookup[period] = new_key

        entry["period"] = new_key

        unit = other.units[entry["unit"]]

        if unit in self.units:
          entry["unit"] = self.units.index(unit)
          self.data[item].append(entry)
          continue

        if unit not in diff_units:
          diff_units.append(unit)
          unit_index_count += 1

        entry["unit"] = unit_index_count

        self.data[item].append(entry)

    diff_items = set(other.data.keys()).difference(common_items)

    for item in diff_items:
      for i, entry in enumerate(other.data[item]):
        period_type = cast(Literal["d", "i"], entry["period"][0])
        period_index = int(entry["period"][1:])
        period = other.periods[period_type][period_index]

        if period in old_periods[period_type]:
          new_key = f"{period_type}{self.periods[period_type].index(period)}"
          other.data[item][i]["period"] = new_key
          continue

        new_key = diff_periods_lookup.get(period, "")
        if period not in diff_periods:
          new_key = f"{period_type}{period_index_count[period_type]}"
          period_index_count[period_type] += 1
          diff_periods[new_key] = period
          diff_periods_lookup[period] = new_key

        other.data[item][i]["period"] = new_key

        unit = other.units[entry["unit"]]

        if unit in self.units:
          other.data[item][i]["unit"] = self.units.index(unit)
          continue

        if unit not in diff_units:
          diff_units.append(unit)
          unit_index_count += 1

        other.data[item][i]["unit"] = unit_index_count

      self.data[item] = other.data[item]

    diff_periods = dict(sorted(diff_periods.items()))

    for k, v in diff_periods.items():
      period_type = cast(Literal["d", "i"], k[0])
      self.periods[period_type].append(v)

    period_sorted = {
      "d": sorted(
        cast(set[Duration], self.periods["d"]), key=lambda x: (x.start_date, x.end_date)
      ),
      "i": sorted(cast(set[Instant], self.periods["i"]), key=lambda x: x.instant),
    }

    period_remap: dict[str, str] = {}
    for k, v in self.periods.items():
      for i, p in enumerate(v):
        remap_index = cast(list[Duration | Instant], period_sorted[k]).index(
          cast(Duration | Instant, p)
        )
        period_remap[f"{k}{i}"] = f"{k}{remap_index}"

    unit_sorted = sorted(self.units)
    unit_remap = [unit_sorted.index(unit) for unit in self.units]

    for k in self.data:
      for i, entry in enumerate(self.data[k]):
        self.data[k][i]["period"] = period_remap[entry["period"]]
        self.data[k][i]["unit"] = unit_remap[entry["unit"]]

      self.data[k] = sorted(self.data[k], key=lambda x: x["period"])


class FinStatementFrame(DataFrameModel):
  url: list[str]
  scope: str = Field(isin={"annual", "quarterly"})
  date: Timestamp
  period: str = Field(isin={"FY", "Q1", "Q2", "Q3", "Q4"})
  fiscal_end: str
  currency: set[str]
  periods: FinPeriods
  data: dict[str, list[FinRecord]]


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
