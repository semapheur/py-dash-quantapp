from typing import cast

import httpx

from lib.const import HEADERS


async def lei_by_isin(isin: str) -> str | None:
  url = "https://api.gleif.org/api/v1/lei-records"
  params = {
    "filter[isin]": isin,
  }

  async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
      print(response.text)
      return None

    parse = response.json()

  data = parse["data"]
  if not data:
    return None

  if len(data) > 1:
    raise ValueError(f"Multiple LEI records found for ISIN: {isin}")

  return cast(str, data[0]["id"])


def fuzzysearch_lei(search: str, fulltext: bool = False) -> str | dict[str, str] | None:
  url = "https://api.gleif.org/api/v1/fuzzycompletions"
  params = {
    "field": "entity.legalName" if fulltext else "fulltext",
    "q": search,
  }

  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS, params=params)
    parse = rs.json()

  print(parse)
  data = parse["data"]
  if not data:
    return None

  if len(data) == 1:
    hit = data[0]
    if "relationships" not in hit or hit["type"] != "fuzzycompletions":
      return None

    return hit["relationships"]["lei-records"]["data"]["id"]

  result: dict[str, str] = {}
  for hits in parse["data"]:
    if "relationships" not in hits or hits["type"] != "fuzzycompletions":
      continue

    name = hits["attributes"]["value"]
    lei = hits["relationships"]["lei-records"]["data"]["id"]
    result[name] = lei

  if not result:
    return None

  return result
