from typing import Literal, TypeAlias, TypedDict

QuotePeriod: TypeAlias = Literal[
  '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
]

QuoteInterval: TypeAlias = Literal[
  '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
]


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
