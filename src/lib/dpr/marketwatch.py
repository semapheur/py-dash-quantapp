# Data analysis
import numpy as np
import pandas as pd

# Date
from datetime import datetime as dt
from datetime import timezone as tz
from dateutil.relativedelta import relativedelta
from calendar import monthrange

# Web scrapping
import requests
import bs4 as bs
import json

# Utils
import re
from tqdm import tqdm
import pycountry

# Local
from lib.finlib import finItemRenameDict


class Ticker:
  # Class variable
  _headers = headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.5",
    "Dylan2010.EntitlementToken": "cecc4267a0194af89ca343805a3e57af",
    "Origin": "https://www.marketwatch.com",
    "DNT": "1",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "Sec-GPC": "1",
  }

  def __init__(self, ticker, exchange, country):
    self._ticker = ticker
    self._exchange = exchange

    if len(country) > 2:
      country = pycountry.countries.get(country=country).alpha_2

    self._country = country
    self._fiscalMonth = self.fiscalYearEnd()

  def fiscalYearEnd(self):
    params = (("mod", "mw_quote_tab"),)

    url = (
      "https://www.marketwatch.com/investing/stock/"
      f"{self._ticker}/company-profile?countrycode={self._country}"
    )
    with requests.Session() as s:
      rq = s.get(url, headers=self._headers, params=params)
      soup = bs.BeautifulSoup(rq.text, "lxml")

    span = soup.find("small", string="Fiscal Year-end")
    if span is not None:
      span = span.find_next_sibling("span").text
      if span == "N/A":
        month = 12
      else:
        month = int(span.split("/")[0])

    else:
      month = 12

    return month

  def financials(self):
    def cellToNumerical(cellVal):
      pattern = (r"(?<=^\().*(?=\)$)", r"[BKMT]$")
      suffix = ""

      if (cellVal == "-") or (cellVal == "N/A"):
        return np.nan

      elif cellVal[-1] == "%":
        return float(cellVal[:-1]) / 100

      m = re.search(pattern[0], cellVal)
      if m:  # Negative figure
        cellVal = f"-{m.group()}"

      # Extract suffix
      if re.search(pattern[1], cellVal):
        suffix = cellVal[-1]
        cellVal = cellVal[:-1]

      # Cast to float
      cellVal = float(cellVal)

      # Multiply by factor suffix
      if suffix == "K":
        cellVal *= 1e3
      elif suffix == "M":
        cellVal *= 1e6
      elif suffix == "B":
        cellVal *= 1e9
      elif suffix == "T":
        cellVal *= 1e12

      return cellVal

    def parseTbl(tbl, period):
      pattern = (
        r"(^Return On )|"
        r"("
        r"((?<= / )(Sales|Total (Assets|Deposits)|Interest Income))|"
        r"\b((Gr(wo|ow)th( Rate)?)|Margin|Turnover|Ratio|Yield)"
        r")$"
      )

      dates = []
      tr = tbl.findAll("tr")
      # Dates
      invalidDates = []
      for i, th in enumerate(tr[0].findAll("th")[1:-1]):
        if period == "a":
          year = int(th.find("div").text)
          day = monthrange(year, self._fiscalMonth)[1]
          dates.append(dt(year, self._fiscalMonth, day))
        elif period == "q":
          date = dt.strptime(th.find("div").text, "%d-%b-%Y")
          if date.year == 1:
            invalidDates.append(i)

          dates.append(date)

      if invalidDates:
        if len(dates) == 1:
          return None

        for i in invalidDates[::-1]:
          if i < (len(dates) - 1):
            date = dates[i + 1] + relativedelta(months=-3)

            day = monthrange(date.year, date.month)[1]
            dates[i] = dt(date.year, date.month, day)

      # Sheet
      data = {}
      for r in tr[1:]:
        td = r.findAll("td")
        itm = td[0].find("div").text

        if re.search(pattern, itm) is None:
          data[itm] = []

          if len(td[1:-1]) > 0:
            tds = td[1:-1]
          else:
            tds = td[1:]

          for d in tds:
            span = d.find("div").find("span")
            if span is not None:
              num = span.text
            else:
              num = "-"

            data[itm].append(cellToNumerical(num.replace(",", "")))

      try:
        df = pd.DataFrame.from_dict(data, orient="columns")
        df.index = dates

      except:
        ln = 0
        for a in data.values():
          if len(a) > ln:
            ln = len(a)

          if len(a) < ln:
            a.extend([np.nan] * (ln - len(a)))

        df = pd.DataFrame.from_dict(data, orient="columns")
        df.index = dates

      return df

    interval = {"a": "", "q": "/quarter"}

    if self._country != "US":
      params = (("countrycode", self._country),)
    else:
      params = ()

    dfs = []
    for k, v in interval.items():
      dfSheets = []
      for sheet in ["income", "balance-sheet", "cash-flow"]:
        url = (
          "https://www.marketwatch.com/investing/stock/"
          f"{self._ticker}/financials/{sheet}{v}"
        )
        with requests.Session() as s:
          rq = s.get(url, headers=self._headers, params=params, stream=True)
          soup = bs.BeautifulSoup(rq.text, "lxml")

        tblCls = "table table--overflow align--right"
        tbls = soup.findAll("table", {"class": tblCls})
        if tbls:
          for tbl in tbls:
            dfSheets.append(parseTbl(tbl, k))

      if dfSheets:
        dfSheets = [i for i in dfSheets if i is not None]
        temp = pd.concat(dfSheets, axis=1)
        temp = temp.loc[:, ~temp.columns.duplicated()]
        temp["period"] = k
        dfs.append(temp)

    if not dfs:
      return None

    df = pd.concat(dfs, axis=0)
    # Drop duplicate columns
    # df = df.loc[:,~df.columns.duplicated()]
    # Rename columns
    rnm = finItemRenameDict("Marketwatch")
    df.rename(columns=rnm, inplace=True)
    df.set_index("period", append=True, inplace=True)
    df.index.names = ["date", "period"]

    # Additional items
    cols = df.columns
    if "rvn" not in cols:
      df["rvn"] = df["grsIntInc"] + df["noIntInc"] + df["opInc"]

    if "opEx" not in cols:
      df["opEx"] = df["othOpEx"]

    if "rvnEx" not in cols:
      df["rvnEx"] = df["rvn"] - df["opEx"] + df["opInc"]

    if "intEx" not in cols:
      df["intEx"] = df["othIntEx"]

    if "ebit" not in cols:
      df["ebit"] = df["netInc"] + df["intEx"] + df["taxEx"]
      # df['ebit'] = df['rvn'] - df['rvnEx'] - df['opEx']

    if "ebitda" not in cols:
      if "da" not in cols:
        if "loanLossPrv" in cols:
          df["da"] = df["loanLossPrv"]

        elif "othOpEx" in cols:
          df["da"] = df["othOpEx"]

      df["ebitda"] = df["ebit"] + df["da"]
      # df['ebit'] = df['rvn'] - df['rvnEx'] - df['opEx']

    if "acntRcv" not in cols:
      if "intRcv" in cols:
        df["acntRcv"] = df["intRcv"]

      elif "isrcPrmRcv" in cols:
        df["acntRcv"] = df["isrcPrmRcv"]

      else:
        df["acntRcv"] = np.nan

    if "stInv" not in cols:
      itm = [
        "trdAcntSecInv",
        "bnkRsrvSldSecPrchInv",
        "othSecInv",
        "prfEqtInv",
        "cmnEqtInv",
      ]
      df["stInv"] = 0
      for i in itm:
        if i in cols:
          df["stInv"] += df[i].fillna(0)

      mask = df["stInv"] == 0
      df.loc[mask, "stInv"] = np.nan

    if "ltInv" not in cols:
      df["ltInv"] = df["totInv"] - df["stInv"]

    if "cceStInv" not in cols:
      df["cceStInv"] = df["cce"] + df["stInv"]

    df["intCvg"] = df["ebit"] / df["intEx"]
    df["taxRate"] = df["taxEx"] / df["ebt"]
    df["totDbt"] = df["stDbt"] + df["ltDbt"]
    df["tgbEqt"] = df["totEqt"] - df["prfEqt"] - df["itgbAst"]
    if "gw" in cols:
      df["tgbEqt"] -= df["gw"]

    if "totCrtAst" not in cols:
      df["totCrtAst"] = df["cceStInv"]

      itm = ["acnRcv", "intRcv", "trdAcntSet", "othSec", "cnsmLoan", "isrcPrmRcv"]
      for i in itm:
        if i in cols:
          df["totCrtAst"] += df[i]

      df["totNoCrtAst"] = df["totAst"] - df["totCrtAst"]

    if "totCrtLbt" not in cols:
      df["totCrtLbt"] = df["stDbt"]

      itm = ["dmndDps", "isrcClmLbt"]
      for i in itm:
        if i in cols:
          df["totCrtLbt"] += df[i]

      df["totNoCrtLbt"] = df["totLbt"] - df["totCrtLbt"]

    df["wrkCap"] = np.nan
    for p in df.index.get_level_values("period").unique():
      msk = (slice(None), p)
      df.loc[msk, "wrkCap"] = (
        df.loc[msk, "totCrtAst"].rolling(2, min_periods=0).mean()
        - df.loc[msk, "totCrtLbt"].rolling(2, min_periods=0).mean()
      )

    if "dvd" not in cols:
      df["dvd"] = np.nan

    if "chgWrkCap" not in cols:
      df["chgWrkCap"] = np.nan
      for p in df.index.get_level_values("period").unique():
        msk = (slice(None), p)
        df.loc[msk, "chgWrkCap"] = df.loc[msk, "wrkCap"].diff()

    if "capEx" not in cols:
      # Capital expenditures
      df["capEx"] = df["ppe"].diff() + df["da"]

    if "opCf" not in cols:
      df["opCf"] = np.nan

    if "freeCf" not in cols:
      df["freeCfFirm"] = (
        df["netInc"]
        + df["da"]
        + df["intEx"] * (1 - df["taxRate"])
        - df["chgWrkCap"]
        - df["capEx"]
      )

      df["freeCf"] = (
        df["freeCfFirm"] + df["totDbt"].diff() - df["intEx"] * (1 - df["taxRate"])
      )

    return df

  def ohlcv(self, startDate="", period="all"):
    if period == "all":
      timeArg = '"TimeFrame": "all"'

    else:
      if isinstance(startDate, dt):
        startStamp = int(startDate.replace(tzinfo=tz.utc).timestamp())

      elif isinstance(startDate, str):
        if startDate:
          startDate = dt.strptime(startDate, "%Y-%m-%d")
          startStamp = int(startDate.replace(tzinfo=tz.utc).timestamp())

      endStamp = int(dt.now().replace(tzinfo=tz.utc).timestamp()) * 1000
      endStamp += 3600 * 24

      timeArg = f'"TimeFrame":"","StartDate":{startStamp},"EndDate":{endStamp}'

    args = (
      f'{{"Step":"P1D",{timeArg},'
      '"EntitlementToken":"cecc4267a0194af89ca343805a3e57af",'
      '"IncludeMockTick":true,"FilterNullSlots":false,'
      '"FilterClosedPoints":true,"IncludeClosedSlots":false,'
      '"IncludeOfficialClose":true,"InjectOpen":false,'
      '"ShowPreMarket":false,"ShowAfterHours":false,"UseExtendedTimeFrame":true,'
      '"WantPriorClose":false,"IncludeCurrentQuotes":false,'
      '"ResetTodaysAfterHoursPercentChange":false,'
      f'"Series":[{{"Key":"STOCK/{self._country}/{self._exchange}/{self._ticker}",'
      '"Dialect":"Charting","Kind":"Ticker",'
      '"SeriesId":"s1","DataTypes":["Open","High","Low","Last"],'
      '"Indicators":[{"Parameters":[],"Kind":"Volume","SeriesId":"i3"}]}]}'
    )
    params = (
      ("json", args),
      ("ckey", "cecc4267a0"),
    )
    url = "https://api-secure.wsj.net/api/michelangelo/timeseries/history"

    with requests.Session() as s:
      rq = s.get(url, headers=self._headers, params=params)
      parse = json.loads(rq.text)

    ix = parse["TimeInfo"]["Ticks"]

    ohlc = parse["Series"][0]["DataPoints"]
    volume = parse["Series"][1]["DataPoints"]
    volume = [itm for sublist in volume for itm in sublist]

    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "adjClose"])
    df["volume"] = volume
    df.index = pd.to_datetime(ix, unit="ms").floor(
      "D"
    )  # Convert index from unix to date
    # df.index.rename('date', inplace=True)
    df.index.names = ["date"]
    df.dropna(inplace=True)
    return df


def batchOhlcv(tickers, startDate="", period=None):
  if isinstance(tickers, str):
    ticker = Ticker(tickers)
    return Ticker.ohlcv(startDate, period)

  elif isinstance(tickers, list):
    dfs = []
    for t in tickers:
      ticker = Ticker(t)

      df = ticker.ohlcv(startDate, period)

      cols = pd.MultiIndex.from_product([[t], [c for c in df.columns]])
      df.columns = cols

      dfs.append(df)

    return pd.concat(dfs, axis=1)
  else:
    return None


def getTickers():
  noXchg = {
    # Stocks with missing exchange
    ("Alpha Healthcare Acquisition Corp. III Cl A", "ALPA"): "XNAS",
    ("Abri SPAC I Inc.", "ASPA"): "XNAS",
    ("ACON S2 Acquisition Corp. Wt", "STWOW"): "XNAS",
    ("AGNC Investment Corp. 6.125% Cum. Redeem. Pfd. Series F", "AGNCP"): "XNAS",
    ("Agrico Acquisition Corp. Cl A", "RICO"): "XNAS",
    ("Allegro MicroSystems Inc.", "ALGM"): "XNAS",
    (
      "Arch Capital Group Ltd. Dep. Pfd. (Rep. 1/1000th 4.550% Pfd. Series G)",
      "ACGLN",
    ): "XNAS",
    ("Artelo Biosciences Inc. Wt", "ATLEW"): "OOTC",
    ("ARYA Sciences Acquisition Corp. V Cl A", "ARYE"): "XNAS",
    ("Atlantic Street Acquisition Corp. Wt", "ASAQ.WT"): "XNYS",
    ("Atlas Financial Holdings Inc.", "AFHIF"): "OOTC",
    ("Blue Safari Group Acquisition Corp. Cl A", "BSGA"): "XNAS",
    ("Burford Capital Ltd.", "BUR"): "XNYS",
    ("Burgundy Technology Acquisition Corp. Wt", "BTAQW"): "XNAS",
    ("CBL & Associates Properties Inc.", "CBLAQ"): "OOTC",
    ("CM Life Sciences III Inc.", "CMLTU"): "XNAS",
    ("Coliseum Acquisition Corp.", "MITAU"): "XNAS",
    (
      "ConnectOne Bancorp Inc. Dep. Pfd. (Rep. 1/40th Non-Cum. Perp. Pfd. Series A)",
      "CNOBP",
    ): "XNAS",
    ("Corner Growth Acquisition Corp. Cl A", "COOL"): "XNAS",
    ("Dune Acquisition Corp. Cl A", "DUNE"): "XNAS",
    ("Emmis Communications Corp.", "EMMS"): "OOTC",
    ("Good Works II Acquisition Corp. Cl A", "GWII"): "XNAS",
    ("GreenPower Motor Co. Inc.", "GP"): "XNAS",
    ("Greenrose Acquisition Corp. Wt", "GNRSW"): "OOTC",
    ("Hancock Whitney Corp. 6.25% Sub. Notes due 2060", "HWCPZ"): "XNAS",
    ("Healthcare Services Acquisition Corp. Cl A", "HCAR"): "XNAS",
    ("Horizonte Minerals PLC", "HZMMF"): "OOTC",
    ("iHeartMedia Inc. Cl B", "IHRTB"): "OOTC",
    ("Interpace Biosciences Inc.", "IDXG"): "OOTC",
    ("LATAM Airlines Group S.A. ADR", "LTMAQ"): "OOTC",
    ("Leisure Acquisition Corp. Wt", "LACQW"): "XNAS",
    ("Lionheart Acquisition Corp. II Wt", "LCAPW"): "XNAS",
    ("Mallinckrodt PLC", "MNKKQ"): "OOTC",
    ("Medalist Diversified REIT Inc. Cum. Redeem. Pfd. Series A", "MDRRP"): "XNAS",
    ("Obsidian Energy Ltd.", "OBELF"): "OOTC",
    ("OceanTech Acquisitions I Corp. Cl A", "OTEC"): "XNAS",
    ("Oxford Lane Capital Corp. 6.75% Notes due 2031", "OXLCL"): "XNAS",
    ("Paringa Resources Ltd. ADR", "PNRLY"): "OOTC",
    ("PropTech Investment Corp. II Cl A", "PTIC"): "XNAS",
    ("Pucara Gold Ltd.", "PCRAF"): "OOTC",
    ("Roth CH Acquisition IV Co.", "ROCG"): "XNAS",
    ("Savannah Resources PLC", "SAVNF"): "OOTC",
    ("Shelter Acquisition Corp. I Cl A", "SHQA"): "XNAS",
    ("StoneBridge Acquisition Corp. Cl A", "APAC"): "XNAS",
    ("Tandy Leather Factory Inc.", "TLFA"): "OOTC",
    ("USHG Acquisition Corp.", "HUGS.UT"): "XNYS",
    ("Verano Holdings Corp.", "VRNOF"): "OOTC",
    ("VIQ Solutions Inc.", "VQS"): "XNAS",
    ("Westell Technologies Inc.", "WSTL"): "OOTC",
    ("Wins Finance Holdings Inc.", "WINSF"): "OOTC",
    ("Zion Oil & Gas Inc. Wt", "ZNOGW"): "OOTC",
  }

  def parseTbl(tbl):
    scrap = []
    pattern = (r"(?<=\().+(?=\)$)", r"(?<=\?countryCode=)[A-Z]{2}(?=&amp;iso=[A-Z]+$)?")
    for tr in tbl.findAll("tr")[1:]:
      td = tr.findAll("td")

      ticker = td[0].find("a").find("small").text
      name = td[0].find("a").text.replace(ticker, "").rstrip()
      country = re.search(pattern[1], td[0].find("a").get("href")).group()

      if country == "US":
        xchg = noXchg.get((name, ticker[1:-1]), td[1].text)
      else:
        xchg = td[1].text

      scrap.append(
        {
          "ticker": ticker[1:-1],
          "name": name,
          "sector": td[2].text,
          "exchange": xchg,
          "country": country,
        }
      )

    return scrap

  # Get href for countries countries
  url = "https://www.marketwatch.com/tools/markets"

  with requests.Session() as s:
    rq = s.get(url, headers=Ticker._headers)
    parse = bs.BeautifulSoup(rq.text, "lxml")

  ulCountries = parse.find("ul", {"class": "list-unstyled"})

  links = []
  for li in ulCountries.findAll("li"):
    # if 'break' not in li['class']:
    if li.find("a") is not None:
      links.append(li.find("a").get("href"))

  # Get tickers
  scrap = []
  for l in tqdm(links):
    url = f"https://www.marketwatch.com{l}"

    with requests.Session() as s:
      rq = s.get(url, headers=Ticker._headers)
      soup = bs.BeautifulSoup(rq.text, "lxml")

    # Scrap first page
    tbl = soup.find("table", {"class": "table table-condensed"})
    scrap += parseTbl(tbl)

    # Find pagination
    pg = soup.find("ul", {"class": "pagination"})
    if pg is not None:
      numPages = pg.findAll("li")

      lastPage = numPages[-2].find("a").text

      if "-" in lastPage:
        lastPage = lastPage.split("-")[-1]

      lastPage = int(lastPage)
      pages = [str(i) for i in range(2, lastPage + 1)]

      for p in pages:
        url = f"https://www.marketwatch.com{l}/{p}"

        with requests.Session() as s:
          rc = s.get(url, headers=Ticker._headers)
          soup = bs.BeautifulSoup(rc.text, "lxml")

        tbl = soup.find("table", {"class": "table table-condensed"})

        # scrap.extend(parse(tbl, k))
        scrap += parseTbl(tbl)

  df = pd.DataFrame.from_records(scrap)
  df.drop_duplicates(inplace=True)

  # Remove warrants and notes and pre-IPO stocks
  mask = df["ticker"].str.contains(r"\.W(I|T)[ABCDRS]?$", regex=True) | (
    df["exchange"] == "IPO"
  )
  df = df[~mask]

  # Add security type
  df["type"] = "stock"
  patterns = {
    "fund": r"((?<!reit)|(?<!estate)|(?<!property)|(?<!exchange traded)).+\bfundo?\b",
    "etf": r"(\b(et(c|f|n|p)(s)?)\b)|(\bxtb\b)|(\bexch(ange)? traded\b)",
    "bond": (
      r"(\s\d(\.\d+)?%\s)|"
      r"(t-bill snr)|((\bsnr )?\b(bnd|bds|bonds)\b)|"
      r"(\bsub\.? )?\bno?te?s( due/exp(iry)?)?(?! asa)"
    ),
    "warrant": r"(\bwt|warr(na|an)ts?)\b(?!.+\b(co(rp)?|l(imi)?te?d|asa|ag|plc|spa)\.?\b)",
    "reit": (
      r"(\breit\b)|(\b(estate|property)\b.+\b(fund|trust)\b)|"
      r"(\bfii\b|\bfdo( de)? inv imob\b)|"
      r"(\bfundos?( de)? in(f|v)est(imento)? (?=imob?i?li(a|á)?ri(a|o)\b))"
    ),
  }
  for k, v in patterns.items():
    mask = df["name"].str.contains(v, regex=True, flags=re.I)
    df.loc[mask, "type"] = k
  # Remove bonds, warrants and REITs
  df = df[~df["type"].isin(["bond", "warrant", "reit"])]

  # Trim tickers
  df["tickerTrim"] = df["ticker"].str.lower()
  patterns = {
    "XIST": r"\.e$",
    "XTAE": r"\.(m|l)$",
  }
  for k, v in patterns.items():
    mask = df["exchange"] == k
    df.loc[mask, "tickerTrim"] = df.loc[mask, "tickerTrim"].str.replace(
      v, "", regex=True
    )

  # Remove suffixes and special characters
  pattern = (
    r"(?<=\.p)r(?=\.?[a-z]$)|"  # .PRx
    r"(?<=\.(r|u|w))t(?=[a-z]?$)|"  # .RT/UT/WT
    r"((\.|/)?[inpx][047]{4,5}$)|"
    r"(\s|\.|_|-|/|\')"
  )
  df["tickerTrim"] = df["tickerTrim"].str.replace(pattern, "", regex=True)

  # Change exchange of XLON tickers starting with 0
  mask = (df["exchange"] == "XLON") & df["ticker"].str.contains(r"^0")
  df.loc[mask, "exchange"] = "LTS"

  return df


# import sqlalchemy as sqla
# if __name__ == '__main__':
#    df = getTickers()
#    engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\ticker.db')
#
#    src = 'marketwatch'
#    for t in df['type'].unique(): # Store stock and etf tickers separately
#        mask = df['type'] == t
#        df.loc[mask, df.columns != 'type'].to_sql(
#            f'{src}{t.capitalize()}', con=engine, index=False, if_exists='replace')
