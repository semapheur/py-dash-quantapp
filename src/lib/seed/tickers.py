import hashlib
import json
from numpy import nan
import re

from pandera.typing import DataFrame
import pycountry

from lib.db.lite import insert_sqlite, read_sqlite
from lib.brreg.parse import get_company_ids
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks
from lib.mic import get_mics
from lib.fuzz import fuzzy_merge


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
  blacklist = ["cedear", r"class \w", "ordinary shares", "shs"]
  pattern = r"(?!^)\b(?:{})\b".format("|".join(blacklist))

  def get_primary_currency(group: DataFrame) -> str:
    domicile = group["domicile"].iloc[0]

    if domicile not in group["country"].tolist():
      currencies = group.loc[group["primary"], "currency"]

    else:
      mask = (group["primary"]) & (group["country"] == domicile)
      currencies = group.loc[mask, "currency"]

    currency = currencies.unique()
    return currency[0] if len(currency) == 1 else nan

  def get_primary_tickers(group: DataFrame) -> str:
    domicile = group["domicile"].iloc[0]

    if domicile not in group["country"].tolist():
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
        "currency": lambda x: x.value_counts().index[0],
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
  exchanges["city"] = exchanges["city"].str.title()
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
  companies["currency"] = tickers.groupby("company_id").apply(get_primary_currency)

  tickers = tickers[
    tickers.columns.difference(["primary", "country", "domicile", "sector", "industry"])
  ]

  insert_sqlite(tickers, "ticker.db", "stock", "replace", False)
  insert_sqlite(companies, "ticker.db", "company", "replace", True)
  insert_sqlite(exchanges, "ticker.db", "exchange", "replace", False)


async def seed_funds():
  funds = await get_tickers("fund")

  attributes = {
    "category",
    "asset_class",
    "administrator_company",
    "advisor_company",
    "branding_company",
    "custodian_company",
    "primary_benchmark",
  }

  for a in attributes:
    a_id = f"{a}_id"
    group = funds[[a_id, a]].groupby(a_id).first()
    insert_sqlite(group, "fund.db", a, "replace", True)

  cols = funds.columns.difference(attributes)
  funds = funds[cols]

  insert_sqlite(funds, "fund.db", "fund", "replace", False)


async def seed_ciks():
  ciks = await get_ciks()

  insert_sqlite(ciks, "ticker.db", "edgar", "replace", False)


def seed_brreg_ids():
  brreg = get_company_ids()
  brreg.loc[:, "name"] = brreg["name"].str.lower()

  query = "SELECT company_id, LOWER(name) as name FROM company WHERE domicile = 'NO'"
  df = read_sqlite("ticker.db", query)

  df["name"] = df["name"].str.replace("ordinary shares", "").str.strip()

  brreg = brreg.merge(df, on="name", how="left")

  brreg_rest = brreg.loc[brreg["company_id"].isnull(), ["brreg_id", "name"]]

  matched_names = brreg.loc[~brreg["company_id"].isnull(), "name"]
  df_rest = df[~df["name"].isin(matched_names)]

  fuzzy_df = fuzzy_merge(brreg_rest, df_rest, on="name", threshold=90).drop_duplicates()

  brreg.set_index("brreg_id", inplace=True)
  fuzzy_df.set_index("brreg_id", inplace=True)

  brreg["company_id"] = brreg["company_id"].fillna(fuzzy_df["company_id"])

  brreg = brreg.merge(fuzzy_df[["name_match"]], on="brreg_id", how="left")

  brreg.to_csv("brreg.csv")
