from typing import Annotated, TypedDict
from pandera import DataFrameModel


class Close(TypedDict):
  date: Annotated[int, "unix in milliseconds"]
  close: float | int


class Ohlcv(Close):
  open: float | int
  high: float | int
  low: float | int
  volume: float | int


class Document(TypedDict):
  date: str
  doc_id: str
  doc_type: str
  language: str
  doc_format: str


class EquityDocuments(DataFrameModel):
  date: str
  doc_id: str
  doc_type: str
  language: str
  doc_format: str
