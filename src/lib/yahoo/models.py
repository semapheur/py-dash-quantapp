from typing import Literal, TypedDict

type QuotePeriod = Literal[
  "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]

type QuoteInterval = Literal[
  "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]


class Period(TypedDict):
  timezone: str
  end: int
  start: int
  gmtoffset: int


class CurrentTradingPeriod(TypedDict):
  pre: Period
  regular: Period
  post: Period


class Meta(TypedDict):
  currency: str
  symbol: str
  exchangeName: str
  fullExchangeName: str
  instrumentType: str
  firstTradeDate: int
  regularMarketTime: int
  hasPrePostMarketData: bool
  gmtoffset: int
  timezone: str
  exchangeTimezoneName: str
  regularMarketPrice: float
  fiftyTwoWeekHigh: float
  fiftyTwoWeekLow: float
  regularMarketDayHigh: float
  regularMarketDayLow: float
  regularMarketVolume: int
  longName: str
  shortName: str
  chartPreviousClose: float
  priceHint: int
  currentTradingPeriod: CurrentTradingPeriod
  dataGranularity: str
  range: str
  validRanges: list[str]


class Quote(TypedDict):
  open: list[float]
  high: list[float]
  low: list[float]
  volume: list[int]
  close: list[float]


class AdjcloseItem(TypedDict):
  adjclose: list[float]


class Indicators(TypedDict):
  quote: list[Quote]
  adjclose: list[AdjcloseItem]


class QuoteDividend(TypedDict):
  amount: float
  date: int


class QuoteSplit(TypedDict):
  date: int
  numerator: int
  denominator: int
  splitRatio: str


class QuoteEvents(TypedDict):
  dividends: dict[str, QuoteDividend]
  splits: dict[str, QuoteSplit]


class QuoteData(TypedDict, total=False):
  meta: Meta
  timestamp: list[int]
  events: QuoteEvents | None
  indicators: Indicators


class Chart(TypedDict):
  result: list[QuoteData]
  error: None


class PriceHistory(TypedDict):
  chart: Chart


class ItemMeta(TypedDict):
  symbol: list[str]
  type: list[str]


class Value(TypedDict):
  raw: int
  fmt: str


class ItemRecord(TypedDict):
  dataId: int
  asOfDate: str
  periodType: Literal["TTM", "12M", "3M"]
  currencyCode: str
  reportedValue: Value


type Item = list[dict[str, list[int] | ItemMeta | list[ItemRecord]]]
