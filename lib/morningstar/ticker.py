# import asyncio
import asyncio
from dataclasses import dataclass
from datetime import date as Date, datetime as dt
import re
from typing import cast, Literal, Optional

import bs4 as bs
import httpx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from lib.const import HEADERS
from lib.utils import replace_all
from lib.fin.models import Quote
from lib.morningstar.models import Ohlcv, Close, Document, StockDocuments
from lib.morningstar.fetch import fetch_currency


SCREENER_API = (
  'https://tools.morningstar.co.uk/api/rest.svc/dr6pz9spfi/security/screener'
)
#'https://lt.morningstar.com/api/rest.svc/klr5zyak8x/security/screener'

FACTOR = {'tusener': 1e3, 'millioner': 1e6, 'milliarder': 1e9}


@dataclass(slots=True)
class Security:
  id: str
  currency: Optional[str] = 'USD'


class Stock(Security):
  async def ohlcv(
    self,
    start_date: Date | dt = Date(1950, 1, 1),
    end_date: Optional[Date | dt] = None,
  ) -> DataFrame[Quote]:
    params = {
      'id': f'{self.id}]3]0]',
      'currencyId': self.currency,
      'idtype': 'Morningstar',
      'frequency': 'daily',
      'startDate': start_date.strftime('%Y-%m-%d'),
      'endDate': '' if end_date is None else start_date.strftime('%Y-%m-%d'),
      'outputType': 'COMPACTJSON',
      'applyTrackRecordExtension': 'true',
      'performanceType': '',
    }
    url = 'https://tools.morningstar.no/api/rest.svc/timeseries_ohlcv/dr6pz9spfi'

    async with httpx.AsyncClient() as client:
      response = await client.get(url, headers=HEADERS, params=params)
      if response.status_code != 200:
        raise httpx.RequestError(f'Error fetching OHLCV for {self.id}: {response}')
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
        f'Could not retrive quotes in {self.currency} for {self.id} from Morningstar!'
      )

    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    return cast(DataFrame[Quote], df)

  def get_currency(self):
    self.currency = fetch_currency(self.id)

  def financials(self) -> pd.DataFrame | None:
    def parse_sheet(sheet: Literal['is', 'bs', 'cf']):
      params = {
        'tab': '10',
        'vw': sheet,
        'SecurityToken': f'{self.id}]3]0]E0WWE$$ALL',
        'Id': f'{self.id}',
        'ClientFund': '0',
        'CurrencyId': self.currency,
      }
      url = 'https://tools.morningstar.no/no/stockreport/default.aspx'

      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        soup = bs.BeautifulSoup(rs.text, 'lxml')

      table = soup.find('table', {'class': 'right years5'})
      if table is None:
        return None

      disclaimer = cast(
        bs.Tag, cast(bs.Tag, soup.find('div', {'id': 'FinancialDisclaimer'})).find('p')
      ).text

      factor = FACTOR[
        cast(re.Match, re.search(r'(?<=^Tall i )\w+(?=\.)', disclaimer)).group()
      ]
      # currency = cast(
      #  re.Match, re.search(r'(?<=\bValuta er )\w{3}', disclaimer)
      # ).group()

      tr = cast(bs.ResultSet[bs.Tag], cast(bs.Tag, table).find_all('tr'))
      rpl = {' ': '', ',': '.'}

      data: dict[str, list[float]] = {}
      dates: list[dt] = []
      headers = cast(bs.ResultSet[bs.Tag], cast(bs.Tag, tr[0]).find_all('th'))[1:]
      for h in headers:
        dates.append(dt(year=int(h.text), month=12, day=31))

      for row in tr[1:]:
        if row.get('class', '') == 'majorGrouping':
          continue

        item = cast(bs.Tag, row.find('th')).text
        data[item] = []

        cols = cast(bs.ResultSet[bs.Tag], row.find_all('td'))
        for col in cols:
          if col.text == '-':
            data[item].append(np.nan)
          else:
            data[item].append(float(replace_all(col.text, rpl)) * factor)

      df = pd.DataFrame.from_dict(data, orient='columns')
      df.index = pd.Index(data=dates, name='date')

      return df

    sheets: list[pd.DataFrame] = []

    for s in ('is', 'bs', 'cf'):
      sheets.append(parse_sheet(cast(Literal['is', 'bs', 'cf'], s)))

    financials = pd.concat(sheets)
    return financials

  def documents(self) -> DataFrame[StockDocuments]:
    replacements = {'http:': 'https:', 'amp;': '', '?': '/?'}

    p = 0
    params = {
      'id': f'{self.id}]3]0]E0WWE`$`$ALL',
      'currencyId': 'NOK',
      'languageId': 'nb-NO',
      'pageSize': '1000',
      'moduleId': '59',
      'documentCategory': 'financials',
      'pageNumber': f'{p}',
    }
    url = 'https://tools.morningstar.no/api/rest.svc/security_documents/dr6pz9spfi'
    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS, params=params)
      soup = bs.BeautifulSoup(rs.text, 'lxml')

    pages = int(
      cast(str, cast(bs.Tag, soup.find('documents')).get('totalnumberofpages', '-1'))
    )

    scrap: list[Document] = []
    while p <= pages:
      docs = cast(bs.ResultSet[bs.Tag], soup.find_all('document'))
      for d in docs:
        attr = d.find('attributes')
        rep_type = 'N/A' if attr is None else attr.text.replace('\n', '')

        scrap.append(
          Document(
            date=cast(bs.Tag, d.find('effectivedate')).text,
            doc_type=rep_type,
            language=cast(bs.Tag, d.find('languageid')).text.replace('\n', ''),
            doc_format=cast(bs.Tag, d.find('format')).text,
            url=replace_all(cast(bs.Tag, d.find('downloadurl')).text, replacements),
          )
        )

      p += 1
      params['pageNumber'] = f'{p}'

      with httpx.Client() as client:
        rs = client.get(url, headers=HEADERS, params=params)
        soup = bs.BeautifulSoup(rs.text, 'lxml')

    df = pd.DataFrame.from_records(scrap)
    df.drop_duplicates(ignore_index=True, inplace=True)

    return cast(DataFrame[StockDocuments], df)


class Fund(Security):
  def price(
    self, start_date: Date | dt = Date(1970, 1, 1), end_date: Optional[Date | dt] = None
  ) -> DataFrame[Quote]:
    params = {
      'id': f'{self.id}]2]1]',
      'currencyId': self.currency,
      'idtype': 'Morningstar',
      'frequency': 'daily',
      'startDate': start_date.strftime('%Y-%m-%d'),
      'endDate': '' if end_date is None else start_date.strftime('%Y-%m-%d'),
      'outputType': 'COMPACTJSON',
      'applyTrackRecordExtension': 'true',
      'priceType': '',
    }
    url = 'https://tools.morningstar.no/api/rest.svc/timeseries_price/dr6pz9spfi'

    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS, params=params)
      if rs.status_code != 200:
        raise httpx.RequestError('Could not parse json!')
      parse: list[list[float | int]] = rs.json()

    scrap: list[Close] = []
    for d in parse:
      scrap.append(Close(date=cast(int, d[0]), close=d[4]))

    df = pd.DataFrame.from_records(scrap)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    return cast(DataFrame[Quote], df)

  def fund_details(self) -> pd.DataFrame:
    scrap = []

    # String replacements
    replacements = {',': '.', '%': '', 'n/a': '0', '1\xa0\n            NOK': '0'}

    url = 'https://www.morningstar.no/no/funds/snapshot/snapshot.aspx?id=' + self.id
    with httpx.Client() as client:
      rs = client.get(url)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    # Fees
    table_class = 'snapshotTextColor snapshotTextFontStyle snapshotTable'
    fee_table = cast(
      bs.Tag, soup.find('table', {'class': f'{table_class} overviewKeyStatsTable'})
    )
    buy_fee_text = cast(
      bs.ResultSet[bs.Tag],
      cast(bs.ResultSet[bs.Tag], fee_table.find_all('tr'))[7].find_all('td'),
    )[2].text
    ann_fee_text = cast(
      bs.ResultSet[bs.Tag],
      cast(bs.ResultSet[bs.Tag], fee_table.find_all('tr'))[8].find_all('td'),
    )[2].text
    buy_fee_text = re.sub(r'^-$', '0', buy_fee_text)
    ann_fee_text = re.sub(r'^-$', '0', ann_fee_text)
    buy_fee = float(replace_all(buy_fee_text, replacements))
    ann_fee = float(replace_all(ann_fee_text, replacements))

    # Reference index
    index_table = cast(
      bs.Tag,
      soup.find('table', {'class': f'{table_class} overviewBenchmarkTable2Cols'}),
    )
    vals = cast(bs.ResultSet[bs.Tag], index_table.find_all('tr'))[3].find_all('td')
    index = cast(bs.ResultSet[bs.Tag], vals)[0].text
    if vals[1].text != '-':
      index += '\n' + vals[1].text

    # Alfa/beta (best fit)
    tab = url + '&tab=2'
    with httpx.Client() as client:
      rs = client.get(tab)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    perf_tbl = cast(
      bs.Tag, soup.find('table', {'class': f'{table_class} ratingMptStatsTable'})
    )
    beta_text = cast(
      bs.ResultSet[bs.Tag],
      cast(bs.ResultSet[bs.Tag], perf_tbl.find_all('tr'))[3].find_all('td'),
    )[2].text
    alfa_text = cast(
      bs.ResultSet[bs.Tag],
      cast(bs.ResultSet[bs.Tag], perf_tbl.find_all('tr'))[4].find_all('td'),
    )[2].text
    beta_text = re.sub(r'^-$', '0', beta_text)
    alfa_text = re.sub(r'^-$', '0', alfa_text)
    beta = float(replace_all(beta_text, replacements))
    alfa = float(replace_all(alfa_text, replacements))

    # Portfolio
    tab = url + '&tab=3'
    with httpx.Client() as client:
      rs = client.get(tab)
      soup = bs.BeautifulSoup(rs.text, 'lxml')
    # Check if equity/bond fund

    attrs = {'class': f'{table_class} portfolioEquityStyleTable'}
    style_table = cast(
      bs.ResultSet[bs.Tag],
      cast(bs.Tag, soup.find('table', attrs)).find_all('table', attrs),
    )
    if style_table:
      ratios = []
      for row in cast(bs.ResultSet[bs.Tag], style_table[2].find_all('tr'))[1:6]:
        portfolio = float(
          replace_all(
            cast(bs.ResultSet[bs.Tag], row.find_all('td'))[1].text, replacements
          )
        )
        relative = float(
          replace_all(
            cast(bs.ResultSet[bs.Tag], row.find_all('td'))[2].text, replacements
          )
        )
        ratios.extend([portfolio, relative])
      for row in cast(bs.ResultSet[bs.Tag], style_table[2].find_all('tr'))[7:]:
        portfolio = float(
          replace_all(
            cast(bs.ResultSet[bs.Tag], row.find_all('td'))[1].text, replacements
          )
        )
        relative = float(
          replace_all(
            cast(bs.ResultSet[bs.Tag], row.find_all('td'))[2].text, replacements
          )
        )
        ratios.extend([portfolio, relative])
    else:
      ratios = [0] * 20

    # Append to scrap matrix
    scrap.extend([index, alfa, beta])
    scrap.extend(ratios)
    scrap.extend([buy_fee, ann_fee])

    cols = (
      'benchmark_index',
      'alfa_best_fit',
      'beta_best_fit',
      'price_to_earnings_ratio',
      'price_to_earnings_ratio_relative',
      'price_to_book_value_ratio',
      'price_to_book_value_ratio_relative',
      'price_to_sales_ratio',
      'price_to_sales_ratio_relative',
      'price_to_free_cashflow_ratio',
      'price_to_free_cashflow_ratio_relative',
      'dyf',
      'dyf_relative',
      'estimated_revenue_growth',
      'estimated_revenue_growth_relative',
      'revenue_growth',
      'revenue_growth_relative',
      'free_cashflow_growth',
      'free_cashflow_growth_relative',
      'book_value_growth',
      'book_value_growth_relative',
      'commission',
      'annal_commission',
    )

    df = pd.DataFrame(scrap, columns=cols)
    return df


class Etf(Security):
  def price(
    self, start_date: Date | dt = Date(1970, 1, 1), end_date: Optional[Date | dt] = None
  ) -> DataFrame[Quote]:
    params = {
      'id': f'{self.id}]22]1]',
      'currencyId': self.currency,
      'idtype': 'Morningstar',
      'frequency': 'daily',
      'startDate': start_date.strftime('%Y-%m-%d'),
      'endDate': '' if end_date is None else start_date.strftime('%Y-%m-%d'),
      'outputType': 'COMPACTJSON',
      'applyTrackRecordExtension': 'true',
      'performanceType': '',
    }
    url = 'https://tools.morningstar.no/api/rest.svc/timeseries_price/dr6pz9spfi'

    with httpx.Client() as client:
      rs = client.get(url, headers=HEADERS, params=params)
      if rs.status_code != 200:
        raise httpx.RequestError('Could not parse json!')
      parse: list[list[float | int]] = rs.json()

    scrap: list[Close] = []
    for d in parse:
      scrap.append(Close(date=cast(int, d[0]), close=d[4]))

    df = pd.DataFrame.from_records(scrap)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    return cast(DataFrame[Quote], df)


async def batch_ohlcv(
  ids: list[str],
  start_date: Date | dt = Date(1950, 1, 1),
  end_date: Optional[Date | dt] = None,
  currency: Optional[str] = None,
) -> DataFrame:
  if currency is None:
    currency = 'USD'

  tasks = [
    asyncio.create_task(Stock(id, currency).ohlcv(start_date, end_date)) for id in ids
  ]
  result = await asyncio.gather(*tasks)

  for i, id in enumerate(ids):
    multi_columns = pd.MultiIndex.from_product([[id], result[i].columns])
    result[i].columns = multi_columns

  return cast(DataFrame, pd.concat(result, axis=1))
