import asyncio
import re
from typing import Literal

import aiometer
import httpx
import numpy as np
import pandas as pd
import polars as pl
import pycountry
from pydantic import BaseModel, Field

from lib.const import HEADERS
from lib.utils.string import replace_all

# Morningstar API docs: https://developer.morningstar.com/direct-web-services/documentation/api-reference/screener/regulatory-screener#data-points


class ApiParams(BaseModel):
  page: int = Field(default=1, gt=1)
  pageSize: int = Field(default=1, gt=1)
  sortOrder: str = "Name asc"
  outputType: str = "json"
  version: int = 1
  languageId: str = "en-US"
  currencyId: str = "NOK"
  filters: str = ""
  filterDataPoints: str = ""
  term: str = ""
  securityDataPoints: str = ""
  universeIds: str = ""
  subUniverseId: str = ""


async def fetch_api(
  params: ApiParams, timeout: float | int | httpx.Timeout | None = None
) -> dict:
  url = "https://tools.morningstar.co.uk/api/rest.svc/dr6pz9spfi/security/screener"
  client_timeout = (
    httpx.Timeout(timeout) if isinstance(timeout, (float, int)) else timeout
  )

  async with httpx.AsyncClient(timeout=client_timeout) as client:
    response = await client.get(
      url, headers=HEADERS, params=params.model_dump(exclude_none=True)
    )
    response.raise_for_status()
    return response.json()


def camel_to_snake(name: str) -> str:
  pattern = re.compile(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])")
  return pattern.sub("_", name).lower()


async def fetch_currency(id: str) -> str:
  params = ApiParams(term=id, securityDataPoints="Currency")

  parse = await fetch_api(params)
  return parse["rows"][0].get("Currency", "USD")


async def get_tickers(
  security: Literal["stock", "etf", "index", "fund", "fund_category", "ose"],
) -> pl.LazyFrame:
  rename = {
    "IsPrimary": "primary",
    "SecId": "security_id",
    "EquityCompanyId": "company_id",
    "ExchangeId": "mic",
  }
  fields = {
    "stock": (
      "SecId",
      "isin",
      "IsPrimary",
      "ExchangeId",
      "Currency",
      "Ticker",
      "IPODate",
      "EquityCompanyId",
      "Name",
      # "LegalName",
      "Domicile",
      "SectorName",
      "IndustryName",
      "ClosePrice",
    ),
    "etf": (
      "isin",
      "SecId",
      "Ticker",
      "Name",
      "mic",
      "Currency",
      "CategoryName",
      "ClosePrice",
    ),
    "index": ("SecId", "Name", "Currency"),
    "fund": (
      "SecId",
      "fundId",
      "isin",
      "PrimaryBenchmarkId",
      "PrimaryBenchmarkName",
      "LegalName",
      "Currency",
      "CategoryId",
      "CategoryName",
      "FundInceptionDate",
      "AdministratorCompanyId",
      "AdministratorCompanyName",
      "AdvisorCompanyId",
      "AdvisorCompanyName",
      "BrandingCompanyId",
      "BrandingCompanyName",
      "CustodianCompanyId",
      "CustodianCompanyName",
      "Domicile",
      "GlobalCategoryId",
      "GlobalCategoryName",
    ),
    "func_category": ("name", "id"),
  }
  universe = {
    "stock": "E0WWE$$ALL",
    "etf": "ETEXG$XOSE|ETEUR$$ALL",
    "index": "IXMSX$$ALL",
    "fund": "FONOR$$ALL",
    "fund_category": "FONOR$$ALL",
  }

  def parse_data(
    data: dict,
    security: str,
  ) -> list[dict[str, str | float]]:
    if security == "fund_category":
      return data["filters"][0][0]["CategoryId"]

    return data["rows"]

  async def fetch_data(params: ApiParams, pages: int) -> list[dict | BaseException]:
    timeout = httpx.Timeout(connect=60.0, read=120.0, write=60.0, pool=60.0)

    tasks = [
      asyncio.create_task(fetch_api(params.model_copy(update={"page": p}), timeout))
      for p in range(2, pages + 1)
    ]
    data = await asyncio.gather(*tasks, return_exceptions=True)
    return data

  async def fallback_price_walk(
    params: ApiParams, bound: int, price: float
  ) -> list[dict[str, str | float]]:
    params.page = 1
    params.filters = f"ClosePrice:LTE:{price}"

    scrap: list[dict[str, str | float]] = []
    while price > 0 and len(scrap) < bound:
      data = await fetch_api(params, 60)
      scrap.extend(parse_data(data, security))
      price = float(scrap[-1].get("ClosePrice", 0.0))
      params.filters = f"ClosePrice:LTE:{price}"

    return scrap

  page_size = 50000
  sort_order = "ClosePrice desc" if "ClosePrice" in fields[security] else "Name asc"
  params = ApiParams(
    pageSize=page_size,
    sortOrder=sort_order,
    securityDataPoints="|".join(fields[security]),
    universeIds=universe[security],
    filterDataPoints="CategoryId" if security == "fund_category" else "",
  )

  init = await fetch_api(params, 60)
  total: int = init["total"]
  scrap = parse_data(init, security)

  if total > page_size:
    pages = int(np.ceil(total / page_size))
    data = await fetch_data(params, pages)

    error = False
    for d in data:
      if isinstance(d, Exception):
        error = True
        break

      scrap.extend(parse_data(d, security))

    if error:
      price = float(scrap[-1].get("ClosePrice", 0.0))
      if price > 0.0:
        print("Using fallback")
        temp = await fallback_price_walk(params, total - len(scrap), price)
        scrap.extend(temp)

  lf = pl.LazyFrame(scrap)

  if "ClosePrice" in lf.collect_schema().names():
    lf = lf.filter(pl.col("ClosePrice") > 0.0).drop("ClosePrice")

  lf = lf.rename(rename)

  name_exclude = {"Name", "LegalName"}
  lf = lf.rename(
    {
      c: c.removesuffix("Name")
      for c in lf.collect_schema().names()
      if c.endswith("Name") and c not in name_exclude
    },
  )
  lf = lf.rename({c: camel_to_snake(c) for c in lf.collect_schema().names()})

  if security == "stock":
    pattern = r"^EX(\$+|TP\$+)"
    lf = lf.with_columns(pl.col("mic").str.replace(pattern, "").replace("LTS", "XLON"))

  return lf.unique()


async def fund_data():
  equity_style = (
    "NA",
    "Value-Big Cap",
    "Mix-Big Cap",
    "Growth-Big Cap",
    "Value-Mid Cap",
    "Mix-Mid Cap",
    "Growth-Mid Cap",
    "Value-Small Cap",
    "Mix-Small Cap",
    "Growth-Small Cap",
  )
  bond_style = (
    "NA",
    "Hi CQ-Lo IS",
    "Hi CQ-Mi IS",
    "Hi CQ-Hi IS",
    "Mi CQ-Lo IS",
    "Mi CQ-Mi IS",
    "Mi CQ-Hi IS",
    "Lo CQ-Lo IS",
    "Lo CQ-Mi IS",
    "Lo CQ-Hi IS",
  )
  fields = {
    "SecId": "security_id",
    "PriceCurrency": "currency",
    "EquityStyleBox": "equity_style",
    "BondStyleBox": "bond_style",
    "AverageCreditQualityCode": "average_credit_quality",
    "StarRatingM255": "rating",
    "SustainabilityRank": "sustainaility",
    "GBRReturnM0": "return_ty",
    "GBRReturnM12": "return_1y",
    "GBRReturnM36": "return_3y",
    "GBRReturnM60": "return_5y",
    "GBRReturnM120": "return_10y",
    "Yield_M12": "yield_1y",
    "AlphaM12": "alpha_1y",
    "AlphaM36": "alpha_3y",
    "AlphaM60": "alpha_5y",
    "AlphaM120": "alpha_10y",
    "BetaM12": "beta_1y",
    "BetaM36": "beta_3y",
    "BetaM60": "beta_5y",
    "BetaM120": "beta_10y",
    "R2M12": "r2_1y",
    "R2M36": "r2_3y",
    "R2M60": "r2_5y",
    "R2M120": "r2_10y",
    "StandardDeviationM12": "standard_deviation_1y",
    "StandardDeviationM36": "standard_deviation_3y",
    "StandardDeviationM60": "standard_deviation_5y",
    "StandardDeviationM120": "standard_deviation_10y",
    "SharpeM12": "sharpe_1y",
    "SharpeM36": "sharpe_3y",
    "SharpeM60": "sharpe_5y",
    "SharpeM120": "sharpe_10y",
    "SortinoM12": "sortino_1y",
    "SortinoM36": "sortino_3y",
    "SortinoM60": "sortino_5y",
    "SortinoM120": "sortino_10y",
    "InformationRatioM12": "information_ratio_1y",
    "InformationRatioM36": "information_ratio_3y",
    "InformationRatioM60": "information_ratio_5y",
    "InformationRatioM120": "information_ratio_10y",
    "PERatio": "pe_ratio",
    "PBRatio": "pb_ratio",
    "SRRI": "srri",
    "ExpenseRatio": "expense_ratio",
    "InitialPurchase": "initial_purchase",
    "ActualManagementFee": "actual_management_fee",
    "InvestmentManagementFeePDS": "investment_management_fee",
    "ManagementFee": "management_fee",
    "MaximumExitFee": "maximum_exit_fee",
    "MaxRedemptionFee": "max_redemption_fee",
    "PerformanceFeeActual": "performance_fee",
    "TransactionFeeActual": "transaction_fee",
  }
  params = ApiParams(
    pageSize=50000,
    sortOrder="LegalName asc",
    universeIds="FONOR$$ALL",
    securityDataPoints="|".join(fields.keys()),
  )

  parse = await fetch_api(params)
  data = parse["rows"]
  df = pd.DataFrame.from_records(data)
  df.rename(columns=fields, inplace=True)
  df["equity_style"] = df["equity_style"].apply(lambda x: equity_style[x])
  df["bond_style"] = df["bond_style"].apply(lambda x: bond_style[x])
  return df


def index_data(scope: Literal["country", "region", "sector-us", "sector-global"]):
  scope_list = {
    "country": "countryReturnData",
    "region": "regionReturnData",
    "sector-us": "sectorReturnDataUS",
    "sector-global": "sectorReturnDataGlobal",
  }
  params = {
    "top": "!MSTAR",
    "clientId": "undefined",
    "benchmarkId": "category",
    "version": "3.35.0",
  }
  url = (
    f"https://api.morningstar.com/sal-service/v1/index/valuation/{scope_list[scope]}"
  )

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS, params=params)
    parse = rs.json()

  scrap = []
  for i in parse["gmbValuationDataList"]:
    scrap.append(
      {
        "id": i.get("performanceId"),
        "ticker": i.get("ticker"),
        "name": i.get("indexName"),
        "pfv_ratio:mr": i.get("pfvMostRecent"),
        "pfv_ratio:3m": i.get("pfvMost3M"),
        "pfv_ratio:1y": i.get("pfvMost1Y"),
        "pfv_ratio:5y": i.get("pfvMost5Y"),
        "pfv_ratio:10y": i.get("pfvMost10Y"),
      }
    )

  df = pd.DataFrame.from_records(scrap)

  def alpha3(s):
    pattern = r"^Morningstar [a-zA-Z.]+\s?[a-zA-Z]* Inde"
    match = re.search(pattern, s)

    country = match.group()
    replace = {
      "Morningstar ": "",
      " Inde": "",
      " Market": "",
      "Czech Republic": "Czechia",
      "Korea": "Korea, Republic of",
      "Russia": "Russian Federation",
      "Taiwan": "Taiwan, Province of China",
      "UK": "United Kingdom",
      "U.S.": "United States",
    }
    country = replace_all(country, replace)
    alpha3 = pycountry.countries.get(name=country).alpha_3

    return alpha3

  df["country"] = df["name"].apply(alpha3)

  return df
