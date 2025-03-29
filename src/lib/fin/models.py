import copy
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

  @field_validator("start_date", "end_date", mode="before")
  def validate_start_date(cls, value: int | Date) -> Date:
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(f"Invalid value: {value}")

    return value

  @field_serializer("start_date", "end_date")
  def serialize_date(self, date: Date):
    return int(date.strftime("%Y%m%d"))


class Instant(BaseModel, frozen=True):
  instant: Date

  @field_validator("instant", mode="before")
  def validate_date(cls, value: int) -> Date:
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(f"Invalid value: {value}")

    return value

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
  date: Date
  fiscal_period: FiscalPeriod
  fiscal_end: str
  url: list[str] | None = None
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

  @field_validator("periods", mode="before")
  @classmethod
  def validate_periods(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      try:
        parsed_value = json.loads(value)
        if not isinstance(parsed_value, dict):
          raise ValueError(
            f"{info.field_name} must be a dictionary. Invalid value: {value}"
          )
      except json.JSONDecodeError:
        raise ValueError(
          f"{info.field_name} must be a valid JSON dictionary string. Invalid value: {value}"
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

  @field_serializer("date")
  def serialize_date(self, date: Date):
    return int(date.strftime("%Y%m%d"))

  @field_serializer("currency")
  def serialize_currency(self, currency: set[str]):
    return sorted(currency)

  @field_serializer("periods")
  @classmethod
  def serialize_periods(cls, periods: FinPeriods):
    obj = {
      "d": [d.model_dump() for d in periods["d"]],
      "i": [i.model_dump() for i in periods["i"]],
    }
    return obj

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

  def dump_json_values(self) -> dict[str, int | str]:
    obj = self.model_dump()
    for key in ("url", "currency", "periods", "units", "data"):
      obj[key] = json.dumps(obj[key])

    return obj

  def merge(self, other: "FinStatement"):
    if not isinstance(other, FinStatement):
      return NotImplemented

    if other.date != self.date and other.fiscal_period != self.fiscal_period:
      raise ValueError("Cannot merge statements with different dates or periods")

    self._merge_urls(other)
    self.currency.update(other.currency)
    self._merge_data(other)
    self._sort_entries()

  def _merge_urls(self, other: "FinStatement") -> None:
    if other.url is None:
      return

    if self.url is None:
      self.url = []

    new_urls = set(other.url).difference(self.url)
    if new_urls:
      self.url.extend(other.url)

  def _merge_data(self, other: "FinStatement") -> None:
    diff_periods: dict[str, Instant | Duration] = {}
    diff_periods_lookup: dict[Instant | Duration, str] = {}
    period_index_count: dict[Literal["d", "i"], int] = {
      "d": len(self.periods["d"]),
      "i": len(self.periods["i"]),
    }

    old_periods: dict[Literal["d", "i"], set[Instant | Duration]] = {
      "d": set(self.periods["d"]),
      "i": set(self.periods["i"]),
    }

    diff_units: list[str] = []

    common_items = set(self.data.keys()).intersection(other.data.keys())
    for item in common_items:
      for entry in other.data[item]:
        self._process_entry(
          other,
          entry,
          item,
          item,
          old_periods,
          diff_periods,
          diff_periods_lookup,
          period_index_count,
          diff_units,
        )

    diff_items = set(other.data.keys()).difference(common_items)

    for item in diff_items:
      self.data[item] = []

      same_item = self._find_same_item(other, item)

      if same_item is not None:
        for entry in other.data[item]:
          self._process_entry(
            other,
            entry,
            same_item,
            item,
            old_periods,
            diff_periods,
            diff_periods_lookup,
            period_index_count,
            diff_units,
          )
      else:
        for entry in other.data[item]:
          self._process_entry(
            other,
            entry,
            item,
            item,
            old_periods,
            diff_periods,
            diff_periods_lookup,
            period_index_count,
            diff_units,
          )

    diff_periods = dict(sorted(diff_periods.items()))

    for k, v in diff_periods.items():
      period_type = cast(Literal["d", "i"], k[0])
      self.periods[period_type].append(v)

  def _process_entry(
    self,
    other: "FinStatement",
    entry: FinRecord,
    target_item: str,
    source_item: str,
    old_periods: dict[Literal["d", "i"], set[Instant | Duration]],
    diff_periods: dict[str, Instant | Duration],
    diff_periods_lookup: dict[Instant | Duration, str],
    period_index_count: dict[Literal["d", "i"], int],
    diff_units: list[str],
  ) -> None:
    period_type = cast(Literal["d", "i"], entry["period"][0])
    period_index = int(entry["period"][1:])
    period = other.periods[period_type][period_index]

    new_entry = copy.deepcopy(entry)
    new_entry = self._remap_units(other, new_entry, diff_units)

    if period in old_periods[period_type]:
      if target_item == source_item:
        self._merge_members(other, entry, target_item, diff_units)
        return

      new_key = f"{period_type}{self.periods[period_type].index(period)}"
      new_entry["period"] = new_key
      self.data[target_item].append(new_entry)
      return

    new_entry = self._remap_periods(
      new_entry,
      period,
      period_type,
      diff_periods,
      diff_periods_lookup,
      period_index_count,
    )
    self.data[target_item].append(new_entry)

    return

  def _find_same_item(self, other: "FinStatement", item: str) -> str | None:
    entries = other.data.get(item)
    if entries is None:
      return None

    values: dict[str, float] = {}

    for entry in entries:
      if "value" not in entry:
        continue

      period_type = cast(Literal["d", "i"], entry["period"][0])
      period_index = int(entry["period"][1:])
      period = other.periods[period_type][period_index]

      if period not in self.periods[period_type]:
        continue

      period_index = self.periods[period_type].index(period)
      new_key = f"{period_type}{period_index}"

      values[new_key] = entry["value"]

    if not values:
      return None

    for target_item in self.data:
      target_values = {
        entry["period"]: entry["value"]
        for entry in self.data[target_item]
        if "value" in entry
      }

      common_periods = sorted(set(values.keys()).intersection(target_values.keys()))
      if not common_periods:
        continue

      values_ = [values[p] for p in common_periods]
      target_values_ = [target_values[p] for p in common_periods]

      if values_ == target_values_:
        return target_item

    return None

  def _merge_members(
    self,
    other: "FinStatement",
    entry: FinRecord,
    item: str,
    diff_units: list[str],
  ) -> None:
    members = entry.get("members")
    if members is None:
      return

    members = copy.deepcopy(members)

    for target_entry in self.data[item]:
      if target_entry["period"] != entry["period"]:
        continue

      if "members" not in target_entry:
        for member in members.values():
          if "unit" not in member:
            continue

          member["unit"] = self._remap_single_unit(other, diff_units, member["unit"])

        target_entry["members"] = members
        return

      diff_members = set(members.keys()).difference(
        cast(dict[str, Member], target_entry["members"]).keys()
      )
      if not diff_members:
        return

      for m in diff_members:
        members[m]["unit"] = self._remap_single_unit(
          other, diff_units, members[m]["unit"]
        )

      cast(dict[str, Member], target_entry["members"]).update(members)

  def _remap_single_unit(
    self,
    other: "FinStatement",
    diff_units: list[str],
    unit_index: int,
  ) -> int:
    unit = other.units[unit_index]

    if unit in self.units:
      return self.units.index(unit)

    if unit not in diff_units:
      diff_units.append(unit)

    return len(self.units) + diff_units.index(unit)

  def _remap_units(
    self,
    other: "FinStatement",
    entry: FinRecord,
    diff_units: list[str],
  ) -> FinRecord:
    if "unit" in entry:
      entry["unit"] = self._remap_single_unit(other, diff_units, entry["unit"])

    members = entry.get("members")
    if members is None:
      return entry

    for member in members.values():
      if "unit" not in member:
        continue

      member["unit"] = self._remap_single_unit(other, diff_units, member["unit"])

    return entry

  def _remap_periods(
    self,
    entry: FinRecord,
    period: Instant | Duration,
    period_type: Literal["d", "i"],
    diff_periods: dict[str, Instant | Duration],
    diff_periods_lookup: dict[Instant | Duration, str],
    period_index_count: dict[Literal["d", "i"], int],
  ) -> FinRecord:
    new_key = diff_periods_lookup.get(period, "")
    if new_key not in diff_periods:
      new_key = f"{period_type}{period_index_count[period_type]}"
      period_index_count[period_type] += 1
      diff_periods[new_key] = period
      diff_periods_lookup[period] = new_key

      entry["period"] = new_key

    return entry

  def _sort_entries(self) -> None:
    sorted_periods = FinPeriods(
      d=sorted(
        self.periods["d"],
        key=lambda x: (x.start_date, x.end_date),
      ),
      i=sorted(self.periods["i"], key=lambda x: x.instant),
    )

    period_remap: dict[str, str] = {}
    for period_type in ("d", "i"):
      for old_index, p in enumerate(self.periods[period_type]):
        new_index = sorted_periods[period_type].index(p)
        period_remap[f"{period_type}{old_index}"] = f"{period_type}{new_index}"

    sorted_units = sorted(self.units)
    unit_remap = [sorted_units.index(unit) for unit in self.units]

    self.periods = sorted_periods
    self.units = sorted_units

    for item in self.data:
      for entry in self.data[item]:
        entry["period"] = period_remap[entry["period"]]

        if "unit" in entry:
          entry["unit"] = unit_remap[entry["unit"]]

        members = entry.get("members")
        if members is None:
          continue

        for m in members:
          if "unit" in members[m]:
            cast(dict[str, Member], entry["members"])[m]["unit"] = unit_remap[
              members[m]["unit"]
            ]

      self.data[item] = sorted(self.data[item], key=lambda x: x["period"])

    self.data = dict(sorted(self.data.items()))


class FinStatementFrame(DataFrameModel):
  date: Timestamp
  period: str = Field(isin={"FY", "Q1", "Q2", "Q3", "Q4"})
  fiscal_end: str
  url: list[str]
  currency: set[str]
  periods: FinPeriods
  units: list[str]
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
