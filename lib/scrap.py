import numpy as np

import asyncio
import httpx
import requests
import bs4 as bs

from lib.const import HEADERS

def free_proxy_list():
  url = 'https://free-proxy-list.net/#'

  with httpx.Client() as client:
    rs = client.get(url)
    soup = bs.BeautifulSoup(rs.text, 'lxml')
  
  proxies = soup.find('div', {'id': 'raw'}).find('textarea').text
  proxies = proxies.split('\n')[3:]
  return list(filter(None, proxies))

def proxylist_geonode(limit:int=500) -> list:
  url = 'https://proxylist.geonode.com/api/proxy-list'
  params = {
    'limit': limit,
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

def check_proxy(proxy:str) -> bool:
  url = 'https://httpbin.org/ip'
  proxies = {'http': proxy, 'https': proxy}

  try:
    rs = requests.get(url, headers=HEADERS)
    return rs.status_code == 200
  except:
    return False

async def check_proxies(proxies:list[str]) -> list[str]:
  url = 'https://httpbin.org/ip' # 'https://ipinfo.io/json'

  async def fetch(proxy: str):
    proxies = {'http://': f'http://{proxy}', 'https://': f'https://{proxy}'}
    client = httpx.AsyncClient()

    try:
      rs = await client.get(url, headers=HEADERS, proxies=proxies)
      return rs.status_code == 200
    except:
      return False
    finally:
      await client.aclose()

  tasks = [asyncio.create_task(fetch(p)) for p in proxies]
  result = await asyncio.gather(*tasks)

  result = np.array(proxies)[np.array(result)]
  return result