import asyncio
from typing import cast

import bs4 as bs
import httpx
import numpy as np
import requests

from lib.const import HEADERS


def free_proxy_list() -> list[str]:
  url = 'https://free-proxy-list.net/#'

  with httpx.Client() as client:
    rs = client.get(url)
    soup = bs.BeautifulSoup(rs.text, 'lxml')

  proxies = cast(
    bs.Tag, cast(bs.Tag, soup.find('div', {'id': 'raw'})).find('textarea')
  ).text
  proxy_list = proxies.split('\n')[3:]
  return list(filter(None, proxy_list))


def proxylist_geonode(limit: int = 500) -> list[str]:
  url = 'https://proxylist.geonode.com/api/proxy-list'
  params = {
    'limit': str(limit),
    'page': '1',
    'sort_by': 'lastChecked',
    'sort_type': 'desc',
  }
  with requests.Session() as client:
    rs = client.get(url, params=params, headers=HEADERS)
    parse = rs.json()

  proxies = [''] * limit
  for i, proxy in enumerate(parse['data']):
    proxies[i] = proxy['ip']

  return proxies


async def check_proxies(proxies: list[str]) -> list[str]:
  url = 'https://httpbin.org/ip'  # 'https://ipinfo.io/json'

  async def fetch(proxy: str) -> bool:
    proxies = {'http://': f'http://{proxy}', 'https://': f'https://{proxy}'}
    client = httpx.AsyncClient(proxies=proxies)

    try:
      rs = await client.get(url, headers=HEADERS)
      return rs.status_code == 200
    except Exception:
      return False
    finally:
      await client.aclose()

  tasks = [asyncio.create_task(fetch(p)) for p in proxies]
  result = await asyncio.gather(*tasks)

  result = np.array(proxies)[np.array(result)]
  return result
