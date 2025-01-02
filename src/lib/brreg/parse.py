from datetime import datetime as dt
from typing import cast, Literal
import xml.etree.ElementTree as et

import httpx

from lib.const import HEADERS
from lib.utils import month_difference


def parse_period(type: Literal["instant", "interval"], start_date: str, end_date: str):
  if type == "instant":
    return {"instant": end_date}

  elif type == "interval":
    months = month_difference(
      dt.strptime(start_date, "%Y-%m-%d"), dt.strptime(end_date, "%Y-%m-%d")
    )
    return {"start_date": start_date, "end_date": end_date, "months": months}

  return None


def parse_financial_statements(
  organization_number: str, year: int, type: Literal["SELSKAP", "KONSERN"] = "SELSKAP"
):
  url = f"https://data.brreg.no/regnskapsregisteret/regnskap/{organization_number}?%C3%A5r={year}&regnskapstype={type}"

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    xml = rs.content

  root = et.fromstring(xml)

  start_date = cast(
    str, cast(et.Element, root.find(".//regnskapsperiode/fraDato")).text
  )
  end_date = cast(str, cast(et.Element, root.find(".//regnskapsperiode/tilDato")).text)
  currency = cast(str, cast(et.Element, root.find(".//valuta")).text)

  sections: dict[Literal["instant", "interval"], list[str]] = {
    "instant": ["egenkapitalGjeld", "eiendeler"],
    "interval": ["resultatregnskapResultat"],
  }

  data: dict = {}

  section_el = root.find(".//egenkapitalGjeld")

  for k, v in sections.items():
    for section in v:
      section_el = cast(et.Element, root.find(f".//{section}"))

      for el in section_el.iter():
        if el.text is None:
          continue

        data[el.tag] = [
          {
            "value": float(el.text),
            "period": parse_period(k, start_date, end_date),
            "unit": currency,
          }
        ]

    return xml
