from contextlib import closing
import hashlib
import json
import re
import sqlite3

import orjson
from pandera.typing import DataFrame
import polars as pl
import pycountry

from lib.db.lite import (
  insert_sqlite,
  polars_to_sqlite,
  read_sqlite,
  sqlite_path,
  sqlite_vacuum,
)
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


async def seed_tickers():
  await seed_stock_tickers()
  await seed_ciks()
  map_cik_to_company()
  sqlite_vacuum("ticker.db")


async def seed_stock_tickers():
  def clean_company_name(name: list[str]) -> str:
    blacklist = ["cedear", r"class \w", "ordinary shares", "shs"]
    pattern = re.compile(r"(?!^)\b(?:{})\b".format("|".join(blacklist)), flags=re.I)
    cleaned = [pattern.sub("", x).strip() for x in name]
    return min(cleaned, key=len)

  def primary_securities(rows: pl.Series) -> str:
    primary_mask = rows.struct.field("primary")
    countries = rows.struct.field("country")
    security_ids = rows.struct.field("security_id")
    domiciles = rows.struct.field("domicile")

    domicile = domiciles[0]
    domicile_exists = countries.is_in([domicile]).any()
    mask = primary_mask & (countries == domicile) if domicile_exists else primary_mask

    primary_securities = security_ids.filter(mask).to_list()
    return orjson.dumps(primary_securities).decode("utf-8")

  def primary_currency(rows: pl.Series) -> str | None:
    primary_mask = rows.struct.field("primary")
    countries = rows.struct.field("country")
    currencies = rows.struct.field("currency")
    domiciles = rows.struct.field("domicile")

    domicile = domiciles[0]
    domicile_exists = countries.is_in([domicile]).any()
    mask = primary_mask & (countries == domicile) if domicile_exists else primary_mask

    currency = currencies.filter(mask).unique()
    return currency[0] if currency.len() == 1 else None

  tickers_lf = await get_tickers("stock")

  exchanges_lf = tickers_lf.group_by("mic").agg(
    pl.col("currency").mode().first().alias("currency")
  )

  mics_columns = [
    "mic",
    "market_name",
    "lei",
    "country",
    "city",
    "url",
    "creation_date",
  ]
  mics_lf = get_mics().select(mics_columns)
  exchanges_lf = (
    exchanges_lf.join(mics_lf, on="mic", how="left")
    .with_columns(
      pl.col("city").str.to_titlecase(),
      pl.col("url").str.to_lowercase(),
      pl.col("lei").replace("", None),
    )
    .sort("mic")
  )

  tickers_lf = tickers_lf.with_columns(
    pl.col("domicile").map_elements(
      lambda x: pycountry.countries.get(alpha_3=x).alpha_2, return_dtype=pl.String
    )
  ).join(exchanges_lf.select(["mic", "country"]), on="mic", how="left")

  companies_lf = (
    tickers_lf.group_by("company_id")
    .agg(
      pl.col("name")
      .map_elements(clean_company_name, return_dtype=pl.String)
      .alias("name"),
      pl.col("domicile").first().alias("domicile"),
      pl.col("sector").first().alias("sector"),
      pl.col("industry").first().alias("industry"),
      pl.struct(["primary", "country", "security_id", "domicile"])
      .map_elements(primary_securities, return_dtype=pl.String)
      .alias("primary_security"),
      pl.struct(["primary", "country", "currency", "domicile"])
      .map_elements(primary_currency, return_dtype=pl.String)
      .alias("currency"),
      pl.lit(None).alias("lei"),
    )
    .sort("company_id")
  )

  tickers_lf = tickers_lf.drop(
    ["primary", "country", "domicile", "sector", "industry"]
  ).sort("security_id")

  polars_to_sqlite(
    tickers_lf.collect(), "ticker.db", "stock", "replace", ("security_id",)
  )
  polars_to_sqlite(
    companies_lf.collect(), "ticker.db", "company", "replace", ("company_id",)
  )
  polars_to_sqlite(exchanges_lf.collect(), "ticker.db", "exchange", "replace", ("mic",))


async def seed_stock_tickers_pandas():
  blacklist = ["cedear", r"class \w", "ordinary shares", "shs"]
  pattern = r"(?!^)\b(?:{})\b".format("|".join(blacklist))

  def get_primary_currency(group: DataFrame) -> str | None:
    domicile = group["domicile"].iloc[0]

    if domicile not in group["country"].tolist():
      currencies = group.loc[group["primary"], "currency"]

    else:
      mask = (group["primary"]) & (group["country"] == domicile)
      currencies = group.loc[mask, "currency"]

    currency = currencies.unique()
    return currency[0] if len(currency) == 1 else None

  def get_primary_tickers(group: DataFrame) -> str:
    domicile = group["domicile"].iloc[0]

    if domicile not in group["country"].tolist():
      primary_securities = group.loc[group["primary"], "security_id"].tolist()

    else:
      mask = (group["primary"]) & (group["country"] == domicile)
      primary_securities = group.loc[mask, "security_id"].tolist()

    return json.dumps(primary_securities)

  tickers_lf = await get_tickers("stock")
  tickers = tickers_lf.collect().to_pandas()

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
    },
    include_groups=False,
  )

  tickers = tickers.merge(exchanges[["mic", "country"]], on="mic", how="left")
  companies["primary_security"] = tickers.groupby("company_id").apply(
    get_primary_tickers
  )
  companies["currency"] = tickers.groupby("company_id").apply(get_primary_currency)
  companies["lei"] = None

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

  polars_to_sqlite(ciks, "ticker.db", "edgar", "replace", ("cik",))


def map_cik_to_company_():
  required_tables = {"stock", "exchange", "edgar"}
  query = """
    BEGIN TRANSACTION;

    CREATE TABLE edgar_updated AS
    SELECT
      ed.*,
      matched.company_id
    FROM edgar ed
    LEFT JOIN (
      SELECT DISTINCT
        ed_inner.cik AS cik,
        s.company_id AS company_id
      FROM stock s
      JOIN exchange ex ON s.mic = ex.mic
      JOIN edgar ed_inner
        ON s.ticker IN (
          SELECT value FROM json_each(ed_inner.tickers)
        )
      WHERE ex.country = 'US'
    ) AS matched ON ed.cik = matched.cik;
    
    DROP TABLE edgar;
    ALTER TABLE edgar_updated RENAME TO edgar;
    COMMIT;
  """

  with closing(sqlite3.connect(sqlite_path("ticker.db"))) as conn:
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    missing_tables = required_tables - existing_tables
    if missing_tables:
      raise RuntimeError(f"Missing tables: {', '.join(missing_tables)}")
    conn.executescript(query)


def map_cik_to_company():
  required_tables = {"stock", "exchange", "edgar"}

  query = """
    BEGIN TRANSACTION;
    
    UPDATE edgar 
    SET company_id = (
      SELECT s.company_id
      FROM stock s
      JOIN exchange ex ON s.mic = ex.mic
      WHERE ex.country = 'US'
        AND EXISTS (
          SELECT 1 
          FROM json_each(edgar.tickers) je
          WHERE je.value = s.ticker
        )
      LIMIT 1
    )
    WHERE EXISTS (
      SELECT 1
      FROM stock s
      JOIN exchange ex ON s.mic = ex.mic
      WHERE ex.country = 'US'
        AND EXISTS (
          SELECT 1 
          FROM json_each(edgar.tickers) je
          WHERE je.value = s.ticker
        )
    );
    
    COMMIT;
    """

  with sqlite3.connect(sqlite_path("ticker.db")) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    missing_tables = required_tables - existing_tables
    if missing_tables:
      raise RuntimeError(f"Missing tables: {', '.join(missing_tables)}")

    # Add company_id column if it doesn't exist
    cursor.execute("PRAGMA table_info(edgar)")
    columns = [column[1] for column in cursor.fetchall()]
    if "company_id" not in columns:
      cursor.execute("ALTER TABLE edgar ADD COLUMN company_id TEXT")

    conn.executescript(query)


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
