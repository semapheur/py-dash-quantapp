import copy
from datetime import date as Date, datetime as dt
import math
import re
from typing import cast, Literal, TypedDict
import warnings

import orjson
from pandera import DataFrameModel, Field
from pandera.dtypes import Timestamp
from pandera.typing import Index
from pydantic import (
  BaseModel,
  NonNegativeInt,
  ValidationInfo,
  field_serializer,
  field_validator,
  model_validator,
  model_serializer,
)

from lib.utils.validate import normalize_nan

type Scope = Literal["annual", "quarterly"]
type Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
type Ttm = Literal["TTM1", "TTM2", "TTM3"]
type FiscalPeriod = Literal["FY"] | Quarter
type FinData = dict[str, dict[Duration | Instant, FinRecord]]
type FinDataIndexed = dict[str, dict[str, FinRecordIndexed]]


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


class ValueBase(BaseModel):
  value: int | float

  @field_validator("value", mode="before")
  @classmethod
  def validate_value(cls, value, info: ValidationInfo) -> int | float:
    if value is None:
      return float("nan")

    if isinstance(value, str):
      try:
        return float(value)
      except ValueError:
        raise ValueError(f"Invalid value: {value}")

    return value

  @field_serializer("value")
  def serialize_value(self, value: int | float):
    if isinstance(value, float) and math.isnan(value):
      return None

    return value


class Value(ValueBase):
  unit: str


class ValueIndexed(ValueBase):
  unit: NonNegativeInt


def serialize_date(date: Date) -> int:
  return int(date.strftime("%Y%m%d"))


class DurationSerialized(TypedDict):
  start_date: int
  end_date: int


class Duration(BaseModel, frozen=True):
  start_date: Date
  end_date: Date

  @property
  def months(self) -> int:
    whole_years = self.end_date.year - self.start_date.year
    remaining_months = self.end_date.month - self.start_date.month
    return whole_years * 12 + remaining_months

  def __lt__(self, other) -> bool:
    if isinstance(other, Instant):
      return self.start_date < other.instant

    if not isinstance(other, Duration):
      return NotImplemented

    if self.start_date == other.start_date:
      return self.end_date < other.end_date

    return self.start_date < other.start_date

  @field_validator("start_date", "end_date", mode="before")
  def validate_start_date(cls, value: int | Date) -> Date:
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(f"Invalid value: {value}")

    return value

  @field_serializer("start_date", "end_date")
  def serialize_date_fields(self, date: Date) -> int:
    return serialize_date(date)


class InstantSerialized(TypedDict):
  instant: int


class Instant(BaseModel, frozen=True):
  instant: Date

  def __lt__(self, other) -> bool:
    if isinstance(other, Duration):
      return self.instant < other.start_date

    if not isinstance(other, Instant):
      return NotImplemented

    return self.instant < other.instant

  @field_validator("instant", mode="before")
  def validate_date(cls, value: int) -> Date:
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(f"Invalid value: {value}")

    return value

  @field_serializer("instant")
  def serialize_date_field(self, date: Date) -> int:
    return serialize_date(date)


def get_date(period: Instant | Duration) -> Date:
  return period.end_date if isinstance(period, Duration) else period.instant


class FinPeriods(TypedDict):
  d: list[Duration]
  i: list[Instant]


class FinPeriodsSerialized(TypedDict):
  d: list[DurationSerialized]
  i: list[InstantSerialized]


class Member(Value):
  dim: str


class MemberIndexed(ValueIndexed):
  dim: NonNegativeInt


class FinRecord(Value):
  value: int | float | None = None
  unit: str | None = None
  members: dict[str, Member] | None = None

  @model_validator(mode="after")
  def at_least_one_field(cls, model):
    if model.value is None and model.members is None:
      raise ValueError("FinRecord must have either 'value' or 'members'.")
    return model


class FinRecordIndexed(ValueIndexed):
  members: dict[str, MemberIndexed] | None = None


class FinPeriodLookup:
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


class StringLookup:
  def __init__(self) -> None:
    self.strings: set[str] = set()

  def add_string(self, string: str):
    if not isinstance(string, str):
      raise ValueError(f"Invalid string: {string}")
    self.strings.add(string)

  def get_lookup(self) -> tuple[list[str], dict[str, int]]:
    sorted_strings = sorted(self.strings)
    reverse_lookup = {unit: i for i, unit in enumerate(sorted_strings)}

    return sorted_strings, reverse_lookup


def validate_json(expected_type: type, allow_none: bool):
  def inner(cls, value, info: ValidationInfo):
    if allow_none and value is None:
      return expected_type()

    if isinstance(value, str):
      if allow_none and value == "null":
        return expected_type()

      try:
        parsed_value = orjson.loads(value)
        if not isinstance(parsed_value, expected_type):
          raise ValueError(
            f"{info.field_name} must be a {expected_type.__name__}. Got: {value}"
          )
      except orjson.JSONDecodeError:
        raise ValueError(
          f"{info.field_name} must be a valid JSON {expected_type.__name__}. Got: {value}"
        )
      return parsed_value
    return value

  return inner


def validate_json_set(allow_none: bool):
  def inner(cls, value, info: ValidationInfo):
    if isinstance(value, str):
      if allow_none and value == "null":
        return set()

      try:
        parsed_value = set(orjson.loads(value))
      except orjson.JSONDecodeError:
        raise ValueError(
          f"{info.field_name} must be a valid JSON array string. Got: {value}"
        )
      return parsed_value
    return value

  return inner


def serialize_synonyms(
  synonyms: dict[str, set[str]] | None,
) -> dict[str, list[str]] | None:
  if not synonyms:
    return None

  return {k: sorted(v) for k, v in sorted(synonyms.items())}


def serialize_period_index(periods: set[Duration | Instant]):
  serialized_periods: FinPeriodsSerialized = {
    "d": [],
    "i": [],
  }
  period_lookup: dict[Duration | Instant, str] = {}
  for period in sorted(periods):
    if isinstance(period, Duration):
      period_lookup[period] = f"d{len(serialized_periods['d'])}"
      serialized_periods["d"].append(cast(DurationSerialized, period.model_dump()))
    else:
      period_lookup[period] = f"i{len(serialized_periods['i'])}"
      serialized_periods["i"].append(cast(InstantSerialized, period.model_dump()))

  return serialized_periods, period_lookup


def index_serialized_periods(periods) -> FinPeriods | None:
  if isinstance(periods, str):
    try:
      periods = orjson.loads(periods)
    except orjson.JSONDecodeError:
      raise ValueError(f"Invalid JSON string: {periods}")

  if not isinstance(periods, dict):
    return None

  if not set(periods.keys()).issubset({"d", "i"}):
    raise ValueError(f"Invalid periods: {periods}")

  duration_serialized = periods.get("d", [])
  if not isinstance(duration_serialized, list):
    raise ValueError(
      f"Expected durations to be a list. Got: {type(duration_serialized).__name__}"
    )

  durations = []
  for duration in duration_serialized:
    if not isinstance(duration, dict):
      raise ValueError(
        f"Expected duration to be a dict. Got: {type(duration).__name__}"
      )

    if not set(duration.keys()).issubset({"start_date", "end_date"}):
      raise ValueError(f"Invalid duration format: {duration}")

    durations.append(Duration(**duration))

  instant_serialized = periods.get("i", [])
  if not isinstance(instant_serialized, list):
    raise ValueError(
      f"Expected instant to be a list. Got: {type(instant_serialized).__name__}"
    )

  instants = []
  for instant in instant_serialized:
    if not isinstance(instant, dict):
      raise ValueError(f"Expected instant to be a dict. Got: {type(instant).__name__}")

    if not set(instant.keys()).issubset({"instant"}):
      raise ValueError(f"Invalid instant format: {instant}")

    instants.append(Instant(**instant))

  return FinPeriods(d=durations, i=instants)


class FinStatement(BaseModel):
  date: Date
  fiscal_period: FiscalPeriod
  fiscal_end: str
  sources: list[str]
  currencies: set[str]
  synonyms: dict[str, set[str]] = dict()
  periods: set[Duration | Instant]
  units: set[str]
  dimensions: set[str] = set()
  data: FinData

  @field_validator("date", mode="before")
  @classmethod
  def validate_date(cls, value, info: ValidationInfo):
    if isinstance(value, int):
      try:
        return dt.strptime(str(value), "%Y%m%d").date()
      except ValueError:
        raise ValueError(
          f"As integer '{info.field_name}' must be in YYYYMMDD format. Got: {value}"
        )

    return value

  @field_validator("fiscal_end", mode="before")
  @classmethod
  def validate_fiscal_end(cls, value, info: ValidationInfo):
    try:
      # Add a dummy year to parse the date
      _ = dt.strptime(f"2000-{value}", "%Y-%m-%d")
    except ValueError:
      raise ValueError(f"{info.field_name} must in MM-DD format. Got: {value}")

    return value

  @field_validator("sources", mode="before")
  @classmethod
  def validate_sources(cls, value, info: ValidationInfo):
    return validate_json(list, False)(cls, value, info)

  @field_validator("currencies", mode="before")
  @classmethod
  def validate_currency(cls, value, info: ValidationInfo):
    return validate_json_set(False)(cls, value, info)

  @field_validator("synonyms", mode="before")
  @classmethod
  def validate_synonyms(cls, value, info: ValidationInfo):
    return validate_json(dict, True)(cls, value, info)

  @field_validator("units", "periods", mode="before")
  @classmethod
  def validate_set_fields(cls, value, info: ValidationInfo):
    if isinstance(value, list):
      return set(value)

    return validate_json_set(False)(cls, value, info)

  @field_validator("dimensions", mode="before")
  @classmethod
  def validate_dimensions(cls, value, info: ValidationInfo):
    if isinstance(value, list):
      return set(value)

    return validate_json_set(True)(cls, value, info)

  @classmethod
  def _validate_period_key(cls, key: str, path: str) -> tuple[Literal["d", "i"], int]:
    if not re.fullmatch(r"[di]\d+", key):
      raise ValueError(f"{path} has invalid period key: {key}")

    return cast(Literal["d", "i"], key[0]), int(key[1:])

  @classmethod
  def _resolve_index(cls, index_list: list[str], index, path: str) -> str:
    if not isinstance(index, int):
      raise TypeError(f"{path} must be an integer index. Got: {type(index).__name__}")

    try:
      return index_list[index]
    except IndexError:
      raise ValueError(
        f"{path} index {index} out of range for list of length {len(index_list)}"
      )

  @model_validator(mode="before")
  @classmethod
  def validator(cls, data) -> dict:
    if not isinstance(data, dict):
      raise ValueError(f"{cls.__name__} must be a dictionary. Got: {type(data)}")

    required_fields = set(cls.model_fields.keys())
    missing = required_fields.difference(data.keys())
    if missing:
      raise ValueError(f"{cls.__name__} missing the following fields: {missing}")

    periods_index = index_serialized_periods(data["periods"])
    if periods_index is None:
      return data

    units = data["units"]
    dimensions = data["dimensions"]

    findata = data["data"]
    if not isinstance(findata, dict):
      raise ValueError(
        f"{cls.__name__} data must be a dictionary. Got: {type(findata)}"
      )

    resolved_findata = {}
    for item, records in findata.items():
      if not isinstance(records, dict):
        raise ValueError(
          f"{cls.__name__} data records must be a dictionary. Got: {type(data)}"
        )

      resolved_records = {}
      for period_key, record in records.items():
        if not isinstance(period_key, str):
          raise ValueError(
            f"{cls.__name__} data records keys must be strings. Got: {type(data)}"
          )

        period_type, period_index = cls._validate_period_key(period_key, "data")
        try:
          period: Duration | Instant = periods_index[period_type][period_index]
        except IndexError:
          raise ValueError(f"Invalid period key: {period_key}")

        unit_index = record.get("unit")
        resolved_record = {
          "value": record.get("value"),
          "unit": cls._resolve_index(
            units, unit_index, f"data['{item}']['{period_key}']['unit']"
          )
          if isinstance(unit_index, int)
          else None,
        }

        members = record.get("members")
        if members is not None:
          resolved_members = {}
          for member_key, member in members.items():
            if not isinstance(member_key, str):
              raise ValueError(f"Member key must be a string. Got: {type(member_key)}")

            if not isinstance(member, dict):
              raise ValueError(
                f"{cls.__name__} data records members values must be dictionaries. Got: {type(data)}"
              )

            resolved_members[member_key] = {
              "value": member.get("value"),
              "unit": cls._resolve_index(
                units,
                member.get("unit"),
                f"data['{item}']['{period_key}']['members']['{member_key}']['unit']",
              ),
              "dim": cls._resolve_index(
                dimensions,
                member.get("dim"),
                f"data['{item}']['{period_key}']['members']['{member_key}']['dim']",
              ),
            }

          resolved_record["members"] = resolved_members

        resolved_records[period] = resolved_record

      resolved_findata[item] = resolved_records

    data["data"] = resolved_findata
    data["periods"] = set(periods_index["d"] + periods_index["i"])
    return data

  @field_serializer("date")
  def serialize_date_field(self, date: Date) -> int:
    return serialize_date(date)

  @field_serializer("currencies")
  def serialize_set_fields(self, field_value: set[str]) -> list[str]:
    return sorted(field_value)

  @field_serializer("synonyms")
  def serialize_synonyms_field(
    self, synonyms: dict[str, set[str]] | None
  ) -> dict[str, list[str]] | None:
    return serialize_synonyms(synonyms)

  @field_serializer("periods", when_used="json")
  def serialize_periods(
    self, periods: set[Duration | Instant]
  ) -> list[Duration | Instant]:
    return sorted(periods)

  @field_serializer("dimensions")
  def serialize_dimensions(self, dimensions: set[str]) -> list[str] | None:
    if not dimensions:
      return None

    return sorted(dimensions)

  @field_serializer("units")
  def serialize_units(self, units: set[str]) -> list[str]:
    return sorted(units)

  @field_serializer("data")
  def serialize_data(self, data: FinData) -> dict:
    return {
      item: {
        "period": period.model_dump(),
        **record.model_dump(),
      }
      for item, records in data.items()
      for period, record in records.items()
    }

  @model_serializer(mode="plain")
  def serializer(self) -> dict:
    periods, period_lookup = serialize_period_index(self.periods)

    units = sorted(self.units)
    unit_lookup = {u: i for i, u in enumerate(units)}

    dimensions = sorted(self.dimensions)
    dim_lookup = {d: i for i, d in enumerate(dimensions)}

    reindexed_data: dict = {}
    for item, records in sorted(self.data.items()):
      reindexed_records: dict = {}
      for period, record in sorted(records.items()):
        serialized_record: dict = {}
        period_key = period_lookup[period]

        if record.value is not None:
          serialized_record["value"] = record.value

        if record.unit is not None:
          serialized_record["unit"] = unit_lookup[record.unit]

        members = record.members
        if members is not None:
          members_reindexed = {}
          for member_name, member in members.items():
            members_reindexed[member_name] = {
              "value": member.value,
              "unit": unit_lookup[member.unit],
              "dim": dim_lookup[member.dim],
            }
          serialized_record["members"] = members_reindexed

        reindexed_records[period_key] = serialized_record

      reindexed_data[item] = reindexed_records

    return {
      "date": serialize_date(self.date),
      "fiscal_period": self.fiscal_period,
      "fiscal_end": self.fiscal_end,
      "sources": self.sources,
      "currencies": sorted(self.currencies),
      "periods": periods,
      "units": units,
      "dimensions": dimensions,
      "synonyms": serialize_synonyms(self.synonyms),
      "data": reindexed_data,
    }

  def dump_json_fields(self) -> dict[str, int | str]:
    dump = self.model_dump()
    for k in (
      "sources",
      "currencies",
      "periods",
      "units",
      "dimensions",
      "synonyms",
      "data",
    ):
      dump[k] = orjson.dumps(dump[k]).decode("utf-8")

    return dump

  def merge(self, other: "FinStatement"):
    if not isinstance(other, FinStatement):
      return NotImplemented

    if other.date != self.date and other.fiscal_period != self.fiscal_period:
      raise ValueError("Cannot merge statements with different dates or fiscal periods")

    self._merge_sources(other)
    self.currencies.update(other.currencies)
    self._merge_data(other)

  def _merge_sources(self, other: "FinStatement") -> None:
    if other.sources is None:
      return

    if self.sources is None:
      self.sources = []

    new_urls = set(other.sources).difference(self.sources)
    if new_urls:
      self.sources.extend(other.sources)

  def _merge_data(self, other: "FinStatement") -> None:
    common_items = set(self.data.keys()).intersection(other.data.keys())
    for item in common_items:
      for period, record in other.data[item].items():
        self._process_record(
          record,
          period,
          item,
        )

    diff_items = set(other.data.keys()).difference(common_items)
    for item in diff_items:
      same_item = self._find_same_item(other, item)

      target_item = item if same_item is None else same_item
      if same_item is None:
        self.data[item] = dict()
      else:
        self.synonyms.setdefault(same_item, set()).add(item)

      for period, record in other.data[item].items():
        self._process_record(record, period, target_item)

  def _process_record(
    self,
    record: FinRecord,
    period: Duration | Instant,
    target_item: str,
  ) -> None:
    if period in self.data[target_item]:
      other_members = record.members
      if other_members is None:
        return

      self._merge_members(other_members, period, target_item)
      return

    if record.unit is not None:
      self.units.update(record.unit)
    self.data[target_item][period] = copy.deepcopy(record)

  def _find_same_item(self, other: "FinStatement", item: str) -> str | None:
    record = other.data.get(item)
    if record is None:
      return None

    check_values = {
      p: (normalize_nan(float(e.value)), e.unit)
      for p, e in record.items()
      if p in self.periods and e.value is not None
    }

    if not check_values:
      return None

    check_periods = set(check_values.keys())
    for target_item, target_record in self.data.items():
      target_values = {
        p: (normalize_nan(float(e.value)), e.unit)
        for p, e in target_record.items()
        if e.value is not None and p in check_periods
      }

      if not target_values:
        continue

      if target_values == check_values:
        return target_item

    return None

  def _merge_members(
    self,
    other_members: dict[str, Member],
    period: Duration | Instant,
    target_item: str,
  ) -> None:
    target_entry = self.data[target_item][period]
    if target_entry.members is None:
      target_entry.members = copy.deepcopy(other_members)
      self.units.update({m.unit for m in other_members.values()})
      self.dimensions.update({m.dim for m in other_members.values()})
      return

    common_members = set(target_entry.members.keys()).intersection(other_members.keys())
    for member in common_members:
      other_member = other_members[member]
      same_member = self._find_same_member(target_entry.members, other_member)
      if same_member is None:
        warnings.warn(f"Item {target_item} has ambiguous member {member}")
        continue

      if other_member.dim not in self.dimensions:
        synomym_dim = target_entry.members[same_member].dim
        self.synonyms.setdefault(synomym_dim, set()).add(other_member.dim)

    diff_members = set(other_members.keys()).difference(common_members)
    for member in diff_members:
      other_member = other_members[member]
      same_member = self._find_same_member(target_entry.members, other_member)
      if same_member is None:
        target_entry.members[member] = copy.deepcopy(other_member)
        self.units.add(other_member.unit)
        self.dimensions.add(other_member.dim)
        continue

      self.synonyms.setdefault(same_member, set()).add(member)

      if other_member.dim not in self.dimensions:
        synomym_dim = target_entry.members[same_member].dim
        self.synonyms.setdefault(synomym_dim, set()).add(other_member.dim)

  def _find_same_member(
    self, target_members: dict[str, Member], member: Member
  ) -> str | None:
    for k, v in target_members.items():
      if v.value != member.value:
        continue

      if v.unit != member.unit:
        warnings.warn(
          f"Member {k} has ambigous units {v.unit} (self) and {member.unit} (other) with same value {v.value}"
        )

      return k

    return None

  def fill_values(self) -> None:
    for records in self.data.values():
      unit_check: set[str] = set()
      value_by_dim: dict[str, float] = dict()
      for record in records.values():
        if record.value is not None:
          continue

        if record.members is None:
          continue

        for member in record.members.values():
          value_by_dim.setdefault(member.dim, 0)
          value_by_dim[member.dim] += member.value

          unit_check.add(member.unit)

        value_check = set(value_by_dim.values())
        if len(unit_check) != 1 or len(value_check) != 1:
          continue

        record.value = value_check.pop()
        record.unit = unit_check.pop()

  def to_json(self, filepath: str) -> None:
    if not filepath.endswith(".json"):
      filepath += ".json"

    dump = self.model_dump()
    serialized = orjson.dumps(dump)

    with open(filepath, "wb") as file:
      file.write(serialized)


class FinStatementFrame(DataFrameModel):
  date: Timestamp
  period: str = Field(isin={"FY", "Q1", "Q2", "Q3", "Q4"})
  fiscal_end: str
  sources: list[str]
  currencies: set[str]
  periods: FinPeriods
  units: list[str]
  dimensions: set[str]
  synonyms: dict[str, str]
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
