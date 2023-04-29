import numpy as np

import asyncio
import httpx
import requests
import bs4 as bs

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0',
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.5',
  #'Accept-Encoding': 'gzip, deflate, br',
}

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

async def check_proxies(proxies:list):
  url = 'https://httpbin.org/ip' # 'https://ipinfo.io/json'

  async def fetch(proxy):
    proxies = {'http://': f'http://{proxy}', 'https://': f'https://{proxy}'}
    client = httpx.AsyncClient(proxies=proxies, headers=HEADERS)

    try:
      rs = await client.get(url, timeout=2)
      return rs.status_code == 200
    except:
      print(proxy)
      return False
    finally:
      await client.aclose()

  tasks = [fetch(p) for p in proxies]
  result = await asyncio.gather(*tasks)

  result = np.array(proxies)[np.array(result)]
  return result

if __name__ == '__main__':
  proxies = proxylist_geonode()
  tst = asyncio.run(check_proxies(proxies))
  print(len(tst))