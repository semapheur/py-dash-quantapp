from datetime import datetime as dt
from datetime import timezone as tz
from dateutil.relativedelta import relativedelta

import httpx
import numpy as np
import pandas as pd
from parsel import Selector

from lib.const import HEADERS


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


def get_income_and_balance(slug: str):
  url = f"https://proff.no/regnskap/{slug}"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    selector = Selector(response.text)

  period_table = selector.css('table[aria-label="AccountingTable dates"]')


class Ticker:
  def __init__(self, ticker, link):
    self._ticker = ticker
    self._link = link

  def financials(self):
    def scrapTbl(tbl, nanAppend, cashFlow=False):
      tr0 = 1 if cashFlow else 2

      scrap = {}
      for row in tbl.findAll("tr")[tr0:]:
        if "graphic-row" not in row.get("class", []):
          scrapRow = []
          key = row.find("th").text.replace("\n", "")

          for col in row.findAll("td")[:-1]:
            if col.text == "-":
              entry = np.nan
            else:
              entry = float(col.text.replace("\xa0", "")) * 1e3

            scrapRow.append(entry)

          if nanAppend > 0:
            for n in range(nanAppend):
              scrapRow.append(np.nan)

          scrap.update({key: scrapRow})

      return scrap

    url = [
      f"https://proff.no/regnskap/{self._link}",  # Income and balance sheet
      f"https://proff.no/nokkeltall/{self._link}",  # Cash sheet
    ]

    soup = []
    for u in url:
      with requests.Session() as s:
        rc = s.get(u, headers=self._headers)
        soup.append(bs.BeautifulSoup(rc.text, "lxml"))

    tbls1 = soup[0].findAll("table", {"class": "account-table years-account-table"})

    tbls2 = soup[1].findAll("table", {"class": "account-table years-account-table"})

    # Check for corporate statement
    if len(tbls1) == 8:
      tblIter = tbls1[5:]

      if len(tbls2) == 4:
        tbl = tbls2[3]

    else:
      tblIter = tbls1[1:4]
      tbl = tbls2[1]

    # Dates
    dates1 = []
    dates2 = []

    for d in tblIter[0].findAll("tr")[0].findAll("th")[1:-1]:
      date = dt(year=int(d.text), month=12, day=31)
      dates1.append(date)

    for d in tbl.findAll("tr")[0].findAll("th")[1:-1]:
      date = dt(year=int(d.text), month=12, day=31)
      dates2.append(date)

    # Currency
    currency = []

    for c in tblIter[0].findAll("tr")[1].findAll("td")[:-1]:
      currency.append(c.text)

    if len(dates1) > len(dates2):
      cols = dates1
      nans = (0, len(dates1) - len(dates2))

    elif len(dates1) < len(dates2):
      cols = dates2
      nans = (len(dates2) - len(dates1), 0)
      # Extend currencies
      currency += [currency[-1]] * (len(dates2) - len(dates1))

    else:
      cols = dates1
      nans = (0, 0)

    # Scrap
    parse = {}

    # Executive salary, and income & balance sheet
    for t in tblIter:
      parse.update(scrapTbl(t, nans[0]))

    # Cash sheet
    parse.update(scrapTbl(tbl, nans[1], True))

    df = pd.DataFrame.from_dict(parse, orient="index", columns=cols)
    df = df.T
    df.index.rename("date", inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # Rename columns
    df.rename(columns=renamer(), inplace=True)  # Rename duplicate columns
    rnm = finItemRenameDict("Proff")  # Rename dictionary
    df.rename(columns=rnm, inplace=True)

    # Remove excess columns
    rmv = df.filter(regex="_1").columns
    # df = df[df.columns.difference(rmv)]
    df.drop(rmv, axis=1, inplace=True)
    df.drop(["Lønn", "Leder annen godtgjørelse"], axis=1, inplace=True)

    # Add currencies
    df["currency"] = currency[::-1]

    # Check for other currencies than NOK
    forex = df["currency"].str.contains("NOK")

    # Convert foreign currency values to NOK
    if not forex.all():
      df["exchange"] = np.nan
      df["exRate"] = 1

      df.loc[~forex, "exchange"] = df.loc[~forex, "currency"].apply(
        lambda x: "NOK=X" if x == "USD" else x + "NOK=X"
      )

      startDate = df.index[0] + relativedelta(years=-1)
      ex = df["exchange"].dropna().unique()
      exRates = yf.download(
        tickers=ex.tolist(),
        start=startDate.strftime("%Y-%m-%d"),
        end=dt.now().strftime("%Y-%m-%d"),
        group_by="column",
        auto_adjust=True,
        threads=True,
      )["Close"]

      exRates = exRates.fillna(method="ffill").resample("D").ffill()

      if len(ex) == 1:
        df.loc[~forex, "exRate"] = exRates.reindex(index=df.index[~forex])

      else:
        for er in exRates.columns:
          mask = df["exchange"] == er
          df.loc[mask, "exRate"] = exRates.loc[df.index[mask], er]

      # Convert forex to NOK
      for c in df.select_dtypes(include="number").columns[:-1]:
        df[c] *= df["exRate"]

      # Remove excess columns
      df.drop(["currency", "exchange", "exRate"], axis=1, inplace=True)

    # df.fillna(0, inplace=True)
    for c in ["div", "da", "imp", "invnt", "stInv", "rcv"]:
      df[c].fillna(0, inplace=True)

    df["totDbt"] = df["stDbt"] + df["ltDbt"]
    df["opEx"] = df["rvn"] - df["opInc"]
    df["ebitda"] = df["opInc"] + df["invInc"] + df["da"]
    df["taxRate"] = df["taxEx"] / df["ebt"]

    # Interest coverage ratio
    df["intCvg"] = df["opInc"] / df["intEx"]

    df["capEx"] = df["ppe"].diff() + df["da"]

    df["wrkCap"] = (
      df["totCrtAst"].rolling(2, min_periods=0).mean()
      - df["totCrtLiab"].rolling(2, min_periods=0).mean()
    )

    if df["chgWrkCap"].isnull().all():
      wc = df["wrkCap"].diff()

    else:
      wc = df["chgWrkCap"]

    df["freeCfFirm"] = (
      df["netInc"] + df["da"] + df["intEx"] * (1 - df["taxRate"]) - wc - df["capEx"]
    )

    df["freeCf"] = (
      df["freeCfFirm"] + df["totDbt"].diff() - df["intEx"] * (1 - df["taxRate"])
    )

    return df
