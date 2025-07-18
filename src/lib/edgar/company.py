import asyncio
from dataclasses import dataclass
from datetime import datetime as dt
import re
from typing import cast, Literal
import xml.etree.ElementTree as et

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series
from pandera.dtypes import Timestamp
from pydantic import BaseModel
import httpx

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

  def padded_cik(self):
    return str(self.cik).zfill(10)

  def info(self) -> CompanyInfo:
    url = f"https://data.sec.gov/submissions/CIK{self.padded_cik()}.json"
    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS)
      parse = CompanyInfo(**response.json())

    return parse

  def filings(
    self,
    forms: list[str] | None = None,
    date: dt | None = None,
    filter_xbrl: bool = False,
  ) -> DataFrame[FilingsFrame]:
    def fetch_files(file_name: str) -> Recent:
      url = f"https://data.sec.gov/submissions/{file_name}"
      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS)
        parse = Recent(**rs.json())

      return parse

    def json_to_df(filings: Recent) -> DataFrame[FilingsFrame]:
      data = {
        "id": filings.accessionNumber,
        "date": filings.reportDate,
        "form": filings.form,
        "primary_document": filings.primaryDocument,
        "is_xbrl": filings.isXBRL,
      }
      df = pd.DataFrame(data)
      df.set_index("id", inplace=True)
      df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
      return cast(DataFrame[FilingsFrame], df)

    dfs: list[DataFrame[FilingsFrame]] = []

    info = self.info()

    recent = info.filings.recent
    dfs.append(json_to_df(recent))

    for f in info.filings.files:
      recent = fetch_files(f.name)
      dfs.append(json_to_df(recent))

    if len(dfs) == 1:
      df = dfs.pop()

    else:
      df = cast(DataFrame[FilingsFrame], pd.concat(dfs))
      df.drop_duplicates(inplace=True)
      df.sort_values("date", ascending=True, inplace=True)

    if forms:
      df = cast(DataFrame[FilingsFrame], df.loc[df["form"].isin(forms)])

    if date:
      df = cast(DataFrame[FilingsFrame], df.loc[df["date"] >= date])

    if filter_xbrl:
      df = cast(DataFrame[FilingsFrame], df.loc[df["is_xbrl"].astype(bool)])

    return df

  def xbrls(self, date: dt | None = None) -> Series[str]:
    filings = self.filings(["10-K", "10-Q", "20-F", "40-F"], date, True)

    if filings["date"].max() < dt(2020, 12, 31):
      raise Exception("Not possible to find XBRL names")

    prefix = "https://www.sec.gov/Archives/edgar/data/"

    filings.sort_values("date", ascending=False, inplace=True)
    filings.reset_index(inplace=True)

    last_filing: str = filings["primary_document"].iloc[0]
    pattern = r"([a-z]+)-?\d{8}"
    ticker = cast(re.Match[str], re.search(pattern, last_filing)).group(1)

    filings["xbrl"] = (
      prefix + str(self.cik) + "/" + filings["id"].str.replace("-", "") + "/"
    )
    mask = (filings["date"] >= dt(2020, 7, 1)) & (
      filings["form"].isin(["10-K", "10-Q"])
    ) | ((filings["date"] > dt(2020, 12, 31)) & (filings["form"] == "20-F"))
    filings.loc[~mask, "xbrl"] += (
      ticker + "-" + filings["date"].dt.strftime("%Y%m%d") + ".xml"
    )
    filings.loc[mask, "xbrl"] += filings.loc[mask, "primary_document"].str.replace(
      ".htm", "_htm.xml"
    )
    filings.set_index("id", inplace=True)
    return cast(Series[str], filings["xbrl"])

  async def xbrl_urls(self, date: dt | None = None) -> Series[str]:
    try:
      urls = self.xbrls()

    except Exception:
      filings = self.filings(["10-Q", "10-K"], date, True)
      urls = await parse_xbrl_urls(self.cik, filings.index.to_list(), "htm")

    return urls

  async def get_financials(self, date: dt | None = None) -> list[FinStatement]:
    xbrls = await self.xbrl_urls(date)

    return await parse_statements(xbrls.tolist())

  async def get_calc_template(self, doc_id):
    ns = "http://www.w3.org/1999/xlink"

    url = await parse_xbrl_url(self.cik, doc_id)
    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS)
      root = et.fromstring(rs.content)

    url_pattern = r"https?://www\..+/"
    el_pattern = r"(?<=_)[A-Z][A-Za-z]+(?=_)"

    calc = dict()
    for sheet in root.findall(".//{*}calculationLink"):
      temp = dict()
      for el in sheet.findall(".//{*}calculationArc"):
        parent = re.search(el_pattern, el.attrib[f"{{{ns}}}from"]).group()
        child = re.search(el_pattern, el.attrib[f"{{{ns}}}to"]).group()

        if parent not in temp:
          temp[parent] = {}

        temp[parent].update({child: float(el.attrib["weight"])})

      label = re.sub(url_pattern, "", sheet.attrib[f"{{{ns}}}role"])
      calc[label] = temp

    return calc

  async def get_taxonomy(self) -> pd.DataFrame:
    async def fetch() -> pd.DataFrame:
      docs = self.filings(["10-K", "10-Q"]).index
      tasks: list[asyncio.Task] = []
      for doc in docs:
        url = await parse_xbrl_url(self.cik, doc, "cal")
        if not url:
          continue
        tasks.append(asyncio.create_task(parse_taxonomy(url)))

      dfs: list[pd.DataFrame] = await asyncio.gather(*tasks)
      df = pd.concat(dfs)
      df = df.loc[~df.index.duplicated()]
      return df

    result = await fetch()
    return result
