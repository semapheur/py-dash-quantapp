import hashlib
import json
import re

from pandera.typing import DataFrame
import pycountry

from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks
from lib.mic import get_mics


def hash_companies(companies: list[list[str]], hash_length=10) -> dict[str, list[str]]:
  result: dict[str, list[str]] = {}
  hashes: set[str] = set()

  def generate_hash(company: str, suffix=""):
    base = company + suffix
    return hashlib.sha256(base.encode()).hexdigest()[:hash_length]

  for company in companies:
    name = min(company, key=len)
    hash_value = generate_hash(name)
    suffix = 0

    while hash_value in result:
      suffix += 1
      hash_value = generate_hash(name, str(suffix))

    hashes.add(hash_value)
    result[hash_value] = company

  return result


def find_index(nested_list: list[list[str]], query: str) -> int:
  for i, sublist in enumerate(nested_list):
    if query in sublist:
      return i

  return -1


async def seed_stock_tickers():
  blacklist = ["cedear", r"class \w"]
  pattern = r"(?!^)\b(?:{})\b".format("|".join(blacklist))

  def get_primary_tickers(group: DataFrame) -> str:
    domicile = group["domicile"].iloc[0]

    if domicile == "BM":
      primary_securities = group.loc[group["primary"], "security_id"].tolist()

    else:
      mask = (group["primary"]) & (group["country"] == domicile)
      primary_securities = group.loc[mask, "security_id"].tolist()

    return json.dumps(primary_securities)

  tickers = await get_tickers("stock")

  exchanges = (
    tickers.groupby("mic")
    .agg(
      {
        "currency": "first",
      }
    )
    .reset_index()
  )
  mics = get_mics()
  mics_columns = [
    "mic",
    "market_name",
    "lei",
    "country",
    "city",
    "url",
    "creation_date",
  ]
  exchanges = exchanges.merge(mics[mics_columns], on="mic", how="left")
  exchanges["city"] = exchanges["city"].str.capitalize()
  exchanges["url"] = exchanges["url"].str.lower()

  tickers.loc[:, "domicile"] = tickers["domicile"].apply(
    lambda x: pycountry.countries.get(alpha_3=x).alpha_2
  )

  companies = tickers.groupby("company_id").agg(
    {
      "name": lambda x: re.sub(pattern, "", min(x, key=len), flags=re.I).strip(),
      "domicile": "first",
      "sector": "first",
      "industry": "first",
    }
  )

  tickers = tickers.merge(exchanges[["mic", "country"]], on="mic", how="left")
  companies["primary_security"] = tickers.groupby("company_id").apply(
    get_primary_tickers
  )

  tickers = tickers[
    tickers.columns.difference(
      ["primary", "currency", "country", "domicile", "sector", "industry"]
    )
  ]

  insert_sqlite(tickers, "ticker.db", "stock", "replace", False)
  insert_sqlite(companies, "ticker.db", "company", "replace", True)
  insert_sqlite(exchanges, "ticker.db", "exchange", "replace", False)


async def seed_ciks():
  ciks = await get_ciks()

  insert_sqlite(ciks, "ticker.db", "edgar", "replace", False)
