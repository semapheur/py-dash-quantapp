from datetime import datetime as dt, date as Date
import json
import re
from typing import Literal

import httpx
import pandas as pd
from parsel import Selector

from lib.const import HEADERS
from lib.fin.models import FinStatement, FinRecord, Duration, Instant
from lib.fin.statement import upsert_statements
from lib.utils import pascal_case, month_difference


def get_company_slugs():
  def fetch(url: str, params: dict[str, str]):
    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS, params=params)
      return response.json()

  def parse_companies(data: list[dict]):
    result: list[dict] = []

    for company in data:
      url_name = company["displayName"].lower().replace(" ", "-")
      postal_code = company["postalAddress"]["postPlace"].lower()

      industries: list[dict[str, str]] = company["proffIndustries"]
      industry = (
        industries[0]["name"].lower().replace(" ", "-").replace("--", "-")
        if len(industries) > 0
        else "-"
      )

      id = company["organisationNumber"]

      result.append(
        {
          "organization_number": id,
          "slug": f"{url_name}/{postal_code}/{industry}/{id}",
          "status": company["status"]["status"],
        }
      )

    return result

  url = "https://proff.no/_next/data/5Bols_bq5UW_kWGbmTA4H/segmentation.json"
  params = {
    "companyType": "ASA",
    "mainUnit": "true",
    "page": "1",
  }

  parse = fetch(url, params)

  result = parse_companies(parse["pageProps"]["companies"])

  pages = parse["pageProps"]["pagination"]["numberOfAvailablePages"]

  for page in range(2, pages):
    params["page"] = page
    parse = fetch(url, params)

    result.extend(parse_companies(parse["pageProps"]["companies"]))

  return pd.DataFrame(result)


def get_financials(slug: str):
  def fetch_data(slug: str) -> dict:
    url = f"https://proff.no/regnskap/{slug}"

    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS)
      selector = Selector(response.text)

    script = selector.xpath("//*[@id='__NEXT_DATA__']/text()").get()
    data = json.loads(script)
    return data

  def fix_account_name(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("%", "prosent")
    return pascal_case(text)

  def parse_period(
    start_date: Date, end_date: Date, period: Literal["duration", "instant"]
  ) -> Instant | Duration:
    if period == "instant":
      return Instant(instant=start_date)

    if period == "duration":
      months = month_difference(start_date, end_date)
      return Duration(start_date=start_date, end_date=date, months=months)

  def parse_statement(statement: dict) -> dict[str, FinRecord]:
    start_date = statement["periodStart"]
    end_date = statement["periodEnd"]
    currency = statement["currency"]

    result: dict[str, FinRecord] = {}

    for account in statement["accounts"]:
      meta = account_lex[account["code"]]
      unit = meta["unit"]
      period: Literal["duration", "instant"] = meta["period"]

      result[meta["label"]] = [
        FinRecord(
          value=account["amount"],
          unit=currency if unit == "currency" else unit,
          period=parse_period(start_date, end_date, period),
        )
      ]

    return result

  data = fetch_data(slug)

  # account_map = data["props"]["i18n"]["initialStore"]["no"]["common"]["AccountingFigures"]["figures"]["NO"]
  # account_map = {k: fix_account_name(v) for k, v in account_map.items()}

  with open("lex/proff_accounts.json") as f:
    account_lex = json.load(f)

  statements: list[dict] = data["props"]["pageProps"]["company"]["companyAccounts"]

  records: list[FinStatement] = []
  for statement in statements:
    currency = set(statement["currency"])
    date = dt.strptime(statement["periodEnd"], "%Y-%m-%d").date()

    records.append(
      FinStatement(
        url=f"https://proff.no/selskap/{slug}",
        date=date,
        scope="annual",
        fiscal_period="FY",
        fiscal_end=f"{date.month}-{date.day}",
        currency=currency,
        data=parse_statement(statement),
      )
    )

  upsert_statements("statements.db", "statements", records)
