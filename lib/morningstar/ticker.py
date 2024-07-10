# import asyncio
import asyncio
from dataclasses import dataclass
from datetime import date as Date, datetime as dt
import re
from typing import cast, Literal, Optional

import httpx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from parsel import Selector

from lib.const import HEADERS
from lib.utils import replace_all
from lib.fin.models import Quote
from lib.morningstar.models import Ohlcv, Close, Document, StockDocuments
from lib.morningstar.fetch import fetch_currency


SCREENER_API = (
  "https://tools.morningstar.co.uk/api/rest.svc/dr6pz9spfi/security/screener"
)
#'https://lt.morningstar.com/api/rest.svc/klr5zyak8x/security/screener'

FACTOR = {"tusener": 1e3, "millioner": 1e6, "milliarder": 1e9}


@dataclass(slots=True)
class Security:
  id: str
  currency: Optional[str] = None


class Stock(Security):
  async def ohlcv(
    self,
    start_date: Date | dt = Date(1950, 1, 1),
    end_date: Optional[Date | dt] = None,
  ) -> DataFrame[Quote]:
    params = {
      "id": f"{self.id}]3]0]",
      "currencyId": self.currency,
      "idtype": "Morningstar",
      "frequency": "daily",
      "startDate": start_date.strftime("%Y-%m-%d"),
      "endDate": "" if end_date is None else start_date.strftime("%Y-%m-%d"),
      "outputType": "COMPACTJSON",
      "applyTrackRecordExtension": "true",
      "performanceType": "",
    }
    url = "https://tools.morningstar.no/api/rest.svc/timeseries_ohlcv/dr6pz9spfi"

    async with httpx.AsyncClient() as client:
      response = await client.get(url, headers=HEADERS, params=params)
      if response.status_code != 200:
        raise httpx.RequestError(f"Error fetching OHLCV for {self.id}: {response}")
      parse: list[list[float | int]] = response.json()

    scrap: list[Ohlcv] = []
    for d in parse:
      scrap.append(
        Ohlcv(
          date=cast(int, d[0]),
          open=d[1],
          high=d[2],
          low=d[3],
          close=d[4],
          volume=d[5],
        )
      )

    df = pd.DataFrame.from_records(scrap)
    if df.empty:
      raise ValueError(
        f"Could not retrive quotes in {self.currency} for {self.id} from Morningstar!"
      )

    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df.set_index("date", inplace=True)
    return cast(DataFrame[Quote], df)

  def financials(self) -> pd.DataFrame | None:
    def parse_sheet(sheet: Literal["is", "bs", "cf"]):
      params = {
        "tab": "10",
        "vw": sheet,
        "SecurityToken": f"{self.id}]3]0]E0WWE$$ALL",
        "Id": f"{self.id}",
        "ClientFund": "0",
        "CurrencyId": self.currency,
      }
      url = "https://tools.morningstar.no/no/stockreport/default.aspx"

      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        dom = Selector(rs.text)

      table = dom.xpath('//table[@class="right years5"]')[0]
      if table is None:
        return None

      disclaimer = cast(
        str,
        dom.xpath('//div[@id="FinancialDisclaimer"]//p[1]/text()')[0].get(),
      )

      factor = FACTOR[
        cast(re.Match, re.search(r"(?<=^Tall i )\w+(?=\.)", disclaimer)).group()
      ]

      tr = cast(list[Selector], dom.xpath('//table[@class="right years5"]//tr'))
      rpl = {" ": "", ",": "."}

      data: dict[str, list[float]] = {}
      dates: list[dt] = []
      headers = cast(list[Selector], tr[0].xpath("th"))[1:]
      for h in headers:
        dates.append(dt(int(h.xpath("text()").get()), 12, 31))

      for row in tr[1:]:
        if row.xpath("@class")[0].get() == "majorGrouping":
          continue

        item = row.xpath("th/text()")[0].get()
        data[item] = []

        cols = row.xpath("td")
        for col in cols:
          if col.xpath("text()").get() == "-":
            data[item].append(np.nan)
          else:
            data[item].append(
              float(replace_all(col.xpath("text()").get(), rpl)) * factor
            )

      df = pd.DataFrame.from_dict(data, orient="columns")
      df.index = pd.Index(data=dates, name="date")

      return df

    sheets: list[pd.DataFrame] = []

    for s in ("is", "bs", "cf"):
      sheets.append(parse_sheet(cast(Literal["is", "bs", "cf"], s)))

    financials = pd.concat(sheets)
    return financials

  def documents(self) -> DataFrame[StockDocuments]:
    replacements = {"http:": "https:", "amp;": "", "?": "/?"}

    p = 0
    params = {
      "id": f"{self.id}]3]0]E0WWE`$`$ALL",
      "currencyId": "NOK",
      "languageId": "nb-NO",
      "pageSize": "1000",
      "moduleId": "59",
      "documentCategory": "financials",
      "pageNumber": f"{p}",
    }
    url = "https://tools.morningstar.no/api/rest.svc/security_documents/dr6pz9spfi"

    scrap: list[Document] = []
    while True:
      with httpx.Client() as client:
        response = client.get(url, headers=HEADERS, params=params)
        soup = Selector(response.text)

      docs = soup.xpath("//document")
      if not docs:
        break

      for d in docs:
        attr = d.xpath("attributes/text()").get()
        rep_type = "N/A" if attr is None else attr.strip()

        scrap.append(
          Document(
            date=cast(str, d.xpath("effectivedate/text()").get()),
            doc_type=rep_type,
            language=cast(str, d.xpath("languageid/text()").get()).strip(),
            doc_format=cast(str, d.xpath("format/text()").get()),
            url=replace_all(
              cast(str, d.xpath("downloadurl/text()").get()), replacements
            ),
          )
        )

      p += 1
      params["pageNumber"] = str(p)

    df = pd.DataFrame.from_records(scrap)
    df.drop_duplicates(ignore_index=True, inplace=True)

    return cast(DataFrame[StockDocuments], df)


class Fund(Security):
  def price(
    self, start_date: Date | dt = Date(1970, 1, 1), end_date: Optional[Date | dt] = None
  ) -> DataFrame[Quote]:
    params = {
      "id": f"{self.id}]2]1]",
      "currencyId": self.currency,
      "idtype": "Morningstar",
      "frequency": "daily",
      "startDate": start_date.strftime("%Y-%m-%d"),
      "endDate": "" if end_date is None else start_date.strftime("%Y-%m-%d"),
      "outputType": "COMPACTJSON",
      "applyTrackRecordExtension": "true",
      "priceType": "",
    }
    url = "https://tools.morningstar.no/api/rest.svc/timeseries_price/dr6pz9spfi"

    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS, params=params)
      if response.status_code != 200:
        raise httpx.RequestError("Could not parse json!")
      parse: list[list[float | int]] = response.json()

    scrap: list[Close] = []
    for d in parse:
      scrap.append(Close(date=cast(int, d[0]), close=d[4]))

    df = pd.DataFrame.from_records(scrap)
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df.set_index("date", inplace=True)
    return cast(DataFrame[Quote], df)

  def fund_details(self) -> pd.DataFrame:
    url = "https://www.morningstar.no/no/funds/snapshot/snapshot.aspx?id=" + self.id
    with httpx.Client() as client:
      response = client.get(url)
      dom = Selector(response.text)

    replacements = {",": ".", "%": "", "n/a": "0", "1\xa0\n            NOK": "0"}

    # Fees
    fee_table = dom.xpath(
      '//table[@class="snapshotTextColor snapshotTextFontStyle snapshotTable overviewKeyStatsTable"]'
    )
    buy_fee_text = cast(str, fee_table[0].xpath("(.//tr)[7]//td[2]/text()").get())
    ann_fee_text = cast(str, fee_table[0].xpath("(.//tr)[8]//td[2]/text()").get())
    buy_fee_text = re.sub(r"^-$", "0", buy_fee_text)
    ann_fee_text = re.sub(r"^-$", "0", ann_fee_text)
    buy_fee = float(replace_all(buy_fee_text, replacements))
    ann_fee = float(replace_all(ann_fee_text, replacements))

    # Reference index
    index_table = dom.xpath(
      '//table[@class="snapshotTextColor snapshotTextFontStyle snapshotTable overviewBenchmarkTable2Cols"]'
    )
    vals = index_table[0].xpath("(.//tr)[3]//td")
    index = cast(str, vals[0].xpath("text()").get())
    if vals[1].xpath("text()").get() != "-":
      index += "\n" + cast(str, vals[1].xpath("text()").get())

    # Alfa/beta (best fit)
    tab = url + "&tab=2"
    with httpx.Client() as client:
      response = client.get(tab)
      dom = Selector(response.text)

    perf_tbl = dom.xpath(
      '//table[@class="snapshotTextColor snapshotTextFontStyle snapshotTable ratingMptStatsTable"]'
    )
    beta_text = cast(str, perf_tbl[0].xpath("(.//tr)[3]//td[2]/text()").get())
    alfa_text = cast(str, perf_tbl[0].xpath("(.//tr)[4]//td[2]/text()").get())
    beta_text = re.sub(r"^-$", "0", beta_text)
    alfa_text = re.sub(r"^-$", "0", alfa_text)
    beta = float(replace_all(beta_text, replacements))
    alfa = float(replace_all(alfa_text, replacements))

    # Portfolio
    tab = url + "&tab=3"
    with httpx.Client() as client:
      response = client.get(tab)
      dom = Selector(response.text)

    style_table = dom.xpath(
      '//table[@class="snapshotTextColor snapshotTextFontStyle snapshotTable portfolioEquityStyleTable"]'
    )
    if style_table:
      ratios = []
      rows = style_table[0].xpath(".//tr")[1:6]
      for row in rows:
        portfolio = float(
          replace_all(cast(str, row.xpath("td[2]/text()").get()), replacements)
        )
        relative = float(
          replace_all(cast(str, row.xpath("td[3]/text()").get()), replacements)
        )
        ratios.extend([portfolio, relative])
      rows = style_table[0].xpath(".//tr")[7:]
      for row in rows:
        portfolio = float(
          replace_all(cast(str, row.xpath("td[2]/text()").get()), replacements)
        )
        relative = float(
          replace_all(cast(str, row.xpath("td[3]/text()").get()), replacements)
        )
        ratios.extend([portfolio, relative])
    else:
      ratios = [0] * 20

    # Append to scrap matrix
    scrap = [index, alfa, beta]
    scrap.extend(ratios)
    scrap.extend([buy_fee, ann_fee])

    cols = (
      "benchmark_index",
      "alfa_best_fit",
      "beta_best_fit",
      "price_to_earnings_ratio",
      "price_to_earnings_ratio_relative",
      "price_to_book_value_ratio",
      "price_to_book_value_ratio_relative",
      "price_to_sales_ratio",
      "price_to_sales_ratio_relative",
      "price_to_free_cashflow_ratio",
      "price_to_free_cashflow_ratio_relative",
      "dyf",
      "dyf_relative",
      "estimated_revenue_growth",
      "estimated_revenue_growth_relative",
      "revenue_growth",
      "revenue_growth_relative",
      "free_cashflow_growth",
      "free_cashflow_growth_relative",
      "book_value_growth",
      "book_value_growth_relative",
      "commission",
      "annal_commission",
    )

    df = pd.DataFrame(scrap, columns=cols)
    return df


class Etf(Security):
  def price(
    self, start_date: Date | dt = Date(1970, 1, 1), end_date: Optional[Date | dt] = None
  ) -> DataFrame[Quote]:
    params = {
      "id": f"{self.id}]22]1]",
      "currencyId": self.currency,
      "idtype": "Morningstar",
      "frequency": "daily",
      "startDate": start_date.strftime("%Y-%m-%d"),
      "endDate": "" if end_date is None else start_date.strftime("%Y-%m-%d"),
      "outputType": "COMPACTJSON",
      "applyTrackRecordExtension": "true",
      "performanceType": "",
    }
    url = "https://tools.morningstar.no/api/rest.svc/timeseries_price/dr6pz9spfi"

    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS, params=params)
      if rs.status_code != 200:
        raise httpx.RequestError("Could not parse json!")
      parse: list[list[float | int]] = rs.json()

    scrap: list[Close] = []
    for d in parse:
      scrap.append(Close(date=cast(int, d[0]), close=d[4]))

    df = pd.DataFrame.from_records(scrap)
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df.set_index("date", inplace=True)
    return cast(DataFrame[Quote], df)


async def batch_ohlcv(
  ids: list[str],
  start_date: Date | dt = Date(1950, 1, 1),
  end_date: Optional[Date | dt] = None,
  currencies: Optional[str | list[str]] = None,
) -> DataFrame:
  if isinstance(currencies, str):
    currencies = [currencies] * len(ids)
  elif currencies is None:
    currency_tasks = [asyncio.create_task(fetch_currency(id)) for id in ids]
    currencies = await asyncio.gather(*currency_tasks)

  ohlcv_tasks = [
    asyncio.create_task(Stock(id, currency).ohlcv(start_date, end_date))
    for (id, currency) in zip(ids, currencies)
  ]
  result = await asyncio.gather(*ohlcv_tasks)

  for i, id in enumerate(ids):
    multi_columns = pd.MultiIndex.from_product([[id], result[i].columns])
    result[i].columns = multi_columns

  return cast(DataFrame, pd.concat(result, axis=1))
