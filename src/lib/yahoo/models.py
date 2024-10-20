from typing import Literal, TypeAlias, TypedDict

QuotePeriod: TypeAlias = Literal[
  '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
]

QuoteInterval: TypeAlias = Literal[
  '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
]


class Period(TypedDict):
  timezone: str
  start: int
  end: int
  gmtoffset: int


class TradingPeriod(TypedDict):
  pre: Period
  regular: Period
  post: Period


class QuoteMeta(TypedDict):
  currency: str
  symbol: str
  exchangeName: str
  instrument_type: str
  firstTradeDate: int
  regularMarketTime: int
  gmtoffset: int
  timezone: str
  exchangeTimezoneName: str
  regularMarketPrice: float
  chartPreviousClose: float
  priceHint: int
  currentTradingPeriod: TradingPeriod
  dataGranularity: QuoteInterval
  validRanges: list[QuoteInterval]


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


class Quote(TypedDict):
  high: list[float]
  open: list[float]
  low: list[float]
  close: list[float]
  volume: list[int]


class QuoteAdjustedClose(TypedDict):
  adjclose: list[float]


class QuoteIndicators(TypedDict):
  quote: list[Quote]
  adjclose: list[QuoteAdjustedClose]


class QuoteData(TypedDict):
  meta: QuoteMeta
  timestamp: list[int]
  events: QuoteEvents
  indicators: QuoteIndicators


class ItemMeta(TypedDict):
  symbol: list[str]
  type: list[str]


class Value(TypedDict):
  raw: int
  fmt: str


class ItemRecord(TypedDict):
  dataId: int
  asOfDate: str
  periodType: Literal['TTM', '12M', '3M']
  currencyCode: str
  reportedValue: Value


Item: TypeAlias = list[dict[str, list[int] | ItemMeta | list[ItemRecord]]]
