from typing_extensions import TypedDict

from pydantic import BaseModel
from pandera import DataFrameModel, Field


class Recent(TypedDict):
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
  isXBRL: list[int]
  isInlineXBRL: list[int]
  primaryDocument: list[str]
  primaryDocDescription: list[str]


class File(TypedDict):
  name: str
  filingCount: int
  filingFrom: str
  filingTo: str


class Filings(TypedDict):
  recent: Recent
  files: list[File]


class Mailing(TypedDict):
  street1: str
  street2: str | None = None
  city: str
  stateOrCountry: str
  zipCode: str
  stateOrCountryDescription: str


class Business(TypedDict):
  street1: str
  street2: str | None = None
  city: str
  stateOrCountry: str
  zipCode: str
  stateOrCountryDescription: str


class Addresses(TypedDict):
  mailing: Mailing
  business: Business


class FormerName(BaseModel):
  name: str
  from_: str = Field(alias="from")
  to: str


class CompanyInfo(BaseModel):
  cik: str
  entityType: str
  sic: str
  sicDescription: str
  insiderTransactionForOwnerExists: int
  insiderTransactionForIssuerExists: int
  name: str
  tickers: list[str]
  exchanges: list[str]
  ein: str
  description: str
  website: str
  investorWebsite: str
  category: str
  fiscalYearEnd: str | None
  stateOfIncorporation: str
  stateOfIncorporationDescription: str
  addresses: Addresses
  phone: str
  flags: str
  formerNames: list[FormerName]
  filings: Filings


class CikEntry(TypedDict):
  cik_str: int
  ticker: str
  title: str


class CikFrame(DataFrameModel):
  cik: int
  name: str
  tickers: str
