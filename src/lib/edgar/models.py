from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel
from pandera import DataFrameModel, Field


class Recent(BaseModel):
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


class File(BaseModel):
  name: str
  filingCount: int
  filingFrom: str
  filingTo: str


class Filings(BaseModel):
  recent: Recent
  files: list[File]


class Mailing(BaseModel):
  street1: str
  street2: Optional[str] = None
  city: str
  stateOrCountry: str
  zipCode: str
  stateOrCountryDescription: str


class Business(BaseModel):
  street1: str
  street2: Optional[str] = None
  city: str
  stateOrCountry: str
  zipCode: str
  stateOrCountryDescription: str


class Addresses(BaseModel):
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
