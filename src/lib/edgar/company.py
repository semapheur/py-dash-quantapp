import asyncio
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import date as Date
import re
from typing import cast, Literal

import httpx
from lxml import etree
import pandas as pd
import pandera as pa
from pandera.typing import Index, Series
from pandera.dtypes import Timestamp
import polars as pl
from pydantic import BaseModel

# Local
from lib.const import HEADERS
from lib.edgar.models import Recent, CompanyInfo
from lib.edgar.parse import (
  parse_xbrl_url,
  parse_xbrl_urls,
  parse_statements,
  parse_taxonomy,
)
from lib.fin.models import FinStatement
from lib.scrap import fetch_json

FIELDS = {
  "id": "TEXT",
  "date": "TEXT",
  "scope": "TEXT",
  "period": "TEXT",
  "fiscal_end": "TEXT",
  "currency": "JSON",
  "data": "JSON",
}


class FilingsJSON(BaseModel):
  accessionNumber: list[str]
  filingDate: list[str]
  reportDate: list[str]
  acceptanceDateTime: list[str]
  act: list[str]
  form: list[str]
  fileNumber: list[str]
  filmNumber: list[str]
  items: list[str]
  size: list[int]
  isXBRL: list[Literal[0, 1]]
  isInlineXBRL: list[Literal[0, 1]]
  primaryDocument: list[str]
  primaryDocDescription: list[str]


class FilingsFrame(pa.DataFrameModel):
  id: Index[str]
  date: Series[Timestamp]
  form: Series[str]
  primary_document: Series[str]
  is_xbrl: Series[Literal[0, 1]]


@dataclass(slots=True)
class Company:
  cik: int

  def _padded_cik(self):
    return str(self.cik).zfill(10)

  async def info(self) -> CompanyInfo:
    url = f"https://data.sec.gov/submissions/CIK{self._padded_cik()}.json"
    parse = await fetch_json(url)
    return CompanyInfo(**parse)

  async def filings(
    self,
    forms: list[str] | None = None,
    date: dt | None = None,
    filter_xbrl: bool = False,
  ) -> pl.LazyFrame:
    async def fetch_files(file_name: str) -> Recent:
      url = f"https://data.sec.gov/submissions/{file_name}"
      parse = await fetch_json(url)
      return Recent(**parse)

    def json_to_lf(filings: Recent) -> pl.LazyFrame:
      lf = pl.LazyFrame(
        {
          "id": filings["accessionNumber"],
          "date": filings["reportDate"],
          "form": filings["form"],
          "primary_document": filings["primaryDocument"],
          "is_xbrl": filings["isXBRL"],
        }
      )
      lf = lf.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
        pl.col("is_xbrl").cast(pl.Boolean),
      )
      return lf

    info = await self.info()
    lfs: list[pl.LazyFrame] = [json_to_lf(info.filings["recent"])]

    files = info.filings.get("files", [])
    if files:
      tasks = [fetch_files(f["name"]) for f in info.filings["files"]]
      additional_filings = await asyncio.gather(*tasks)
      lfs.extend([json_to_lf(f) for f in additional_filings])

    lf = (
      cast(pl.LazyFrame, pl.concat(lfs, how="vertical", rechunk=False))
      .unique(subset="id")
      .sort("date")
    )

    if forms:
      lf = lf.filter(pl.col("form").is_in(forms))

    if date is not None:
      lf = lf.filter(pl.col("date") >= pl.lit(date.date()).cast(pl.Date))

    if filter_xbrl:
      lf = lf.filter(pl.col("is_xbrl"))

    return lf

  async def xbrls(self, date: dt | None = None) -> pl.DataFrame:
    filings_lf = await self.filings(["10-K", "10-Q", "20-F", "40-F"], date, True)

    df = filings_lf.sort("date", descending=True).limit(1).collect()

    if cast(Date, df["date"].max()) < dt(2020, 12, 31).date():
      raise Exception("Not possible to find XBRL names")

    prefix = "https://www.sec.gov/Archives/edgar/data/"

    last_filing = df["primary_document"][0]
    pattern = r"([a-z]+)-?\d{8}"
    ticker = cast(re.Match[str], re.search(pattern, last_filing)).group(1)

    filings_lf = (
      filings_lf.with_columns(
        [
          # Base XBRL path
          (
            pl.lit(prefix)
            + pl.lit(str(self.cik))
            + pl.lit("/")
            + pl.col("id").str.replace_all("-", "")
            + pl.lit("/")
          ).alias("xbrl_base"),
          # Modern XBRL mask
          (
            (
              (pl.col("date") >= dt(2020, 7, 1).date())
              & pl.col("form").is_in(["10-K", "10-Q"])
            )
            | ((pl.col("date") > dt(2020, 12, 31).date()) & (pl.col("form") == "20-F"))
          ).alias("is_modern"),
        ]
      )
      .with_columns(
        [
          # Build XBRL URLs based on mask
          pl.when(pl.col("is_modern"))
          .then(
            pl.col("xbrl_base")
            + pl.col("primary_document").str.replace(".htm", "_htm.xml")
          )
          .otherwise(
            pl.col("xbrl_base")
            + pl.lit(ticker + "-")
            + pl.col("date").dt.strftime("%Y%m%d")
            + pl.lit(".xml")
          )
          .alias("xbrl")
        ]
      )
      .select(["id", "xbrl"])
    )

    return filings_lf.collect()

  async def xbrl_urls(self, date: dt | None = None) -> pl.DataFrame:
    try:
      urls = await self.xbrls()

    except Exception:
      filings = await self.filings(["10-Q", "10-K"], date, True)
      doc_ids: list[str] = filings.select("id").collect().get_column("id").to_list()
      urls = await parse_xbrl_urls(self.cik, doc_ids, "htm")

    return urls

  async def get_financials(self, date: dt | None = None) -> list[FinStatement]:
    xbrls = await self.xbrl_urls(date)
    urls = xbrls.select("xbrl").get_column("xbrl").to_list()

    return await parse_statements(urls)

  async def get_calc_template(self, doc_id) -> dict:
    url = await parse_xbrl_url(self.cik, doc_id)
    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS)
      root: etree._Element = etree.fromstring(response.content)

    namespaces = root.nsmap
    xlink = namespaces["xlink"]
    url_pattern = r"https?://www\..+/"
    el_pattern = r"(?<=_)[A-Z][A-Za-z]+(?=_)"

    calc = dict()
    for sheet in root.findall(".//link:calculationLink", namespaces=namespaces):
      temp: dict[str, dict] = dict()
      for el in sheet.findall(".//link:calculationArc"):
        parent = re.search(el_pattern, el.attrib[f"{{{xlink}}}from"]).group()
        child = re.search(el_pattern, el.attrib[f"{{{xlink}}}to"]).group()

        if parent not in temp:
          temp[parent] = {}

        temp[parent].update({child: float(el.attrib["weight"])})

      label = re.sub(url_pattern, "", sheet.attrib[f"{{{xlink}}}role"])
      calc[label] = temp

    return calc

  async def get_taxonomy(self) -> pd.DataFrame:
    async def fetch() -> pd.DataFrame:
      filings = await self.filings(["10-K", "10-Q"])
      docs: list[str] = filings.select("id").collect().get_column("id").to_list()
      tasks: list[asyncio.Task] = []
      for doc in docs:
        url = await parse_xbrl_url(self.cik, doc, "cal")
        if not url:
          continue
        tasks.append(asyncio.create_task(parse_taxonomy(url)))

      frames: list[pl.DataFrame] = await asyncio.gather(*tasks)
      df = pl.concat(frames, how="vertical", rechunk=True).unique(subset="item")
      return df

    result = await fetch()
    return result
