from calendar import monthrange
from datetime import datetime as dt, date as Date, time
from dateutil.relativedelta import relativedelta
import math
from typing import cast, Literal


def valid_date(date_text: str, date_format: str = "%Y-%m-%d") -> bool:
  try:
    dt.strptime(date_text, date_format)
    return True
  except ValueError:
    return False


def handle_date(date: dt | Date) -> dt:
  if isinstance(date, Date):
    date = dt.combine(date, time())

  return date


def month_difference(date1: dt | Date, date2: dt | Date) -> int:
  start_date, end_date = sorted([date1, date2])
  delta = relativedelta(end_date, start_date)
  months = delta.years * 12 + delta.months
  days_in_month = monthrange(end_date.year, end_date.month)[1]
  partial = round(delta.days / days_in_month) if days_in_month else 0

  return months + partial


type Quarter = Literal["Q1", "Q2", "Q3", "Q4"]


def fiscal_quarter(date: dt, fiscal_month: int, fiscal_day: int) -> Quarter:
  condition = date.month < fiscal_month or (
    date.month == fiscal_month and date.day <= fiscal_day
  )
  fiscal_year = date.year - 1 if condition else date.year
  fiscal_start = dt(fiscal_year, fiscal_month, fiscal_day)
  months = month_difference(date, fiscal_start)

  return cast(Quarter, f"Q{math.ceil(months / 3)}")


def fiscal_quarter_monthly(month: int, fiscal_end_month: int | None = None) -> int:
  if fiscal_end_month is not None:
    month = 12 - ((fiscal_end_month - month) % 12)

  return ((month - 1) // 3) + 1


def month_end(year: int, month: int, unleap=True) -> int:
  import calendar

  if unleap and month == 2 and calendar.isleap(year):
    return 28

  return calendar.monthrange(year, month)[1]


def exclusive_end_date(start_date: Date, end_date: Date) -> Date:
  months = month_difference(start_date, end_date)
  delta = relativedelta(years=1) if months == 12 else relativedelta(months=months)
  expected_xbrl_end = start_date + delta

  if end_date != expected_xbrl_end:
    return end_date + relativedelta(days=1)

  return end_date
