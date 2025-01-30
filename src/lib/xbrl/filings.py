import httpx

from lib.const import HEADERS


def get_filings(country: str, page_size: int = 10000):
  url = (
    "https://filings.xbrl.org/api/filings?include=entity,language"
    f'&filter=[{{"name"%3A"country"%2C"op"%3A"eq"%2C"val"%3A"{country}"}}]'
    f"&sort=-date_added&page[size]={page_size}&page[number]=1&_=1738274381858"
  )

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    parse = rs.json()

  return parse
