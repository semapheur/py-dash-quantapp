import asyncio
from io import BytesIO
from pathlib import Path
import random
from typing import cast, Literal

import httpx
import numpy as np
from parsel import Selector
import requests
from rnet import Client, Impersonate
from tqdm import tqdm

from lib.const import HEADERS


def get_chrome_versions(
  platform: Literal["android", "chromeos", "linux", "mac", "win"], number=10
) -> list[str]:
  url = f"https://versionhistory.googleapis.com/v1/chrome/platforms/{platform}/channels/stable/versions"
  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    parse = response.json()

  number = min(number, len(parse["versions"]))
  return [version["version"] for version in parse["versions"][:number]]


def get_firefox_versions(number=10) -> list[str]:
  url = "https://product-details.mozilla.org/1.0/firefox_history_major_releases.json"
  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    parse = response.json()

  number = min(number, len(parse))
  return list(parse.keys())[-number:]


def get_opera_versions(number=10) -> list[str]:
  def version_key(s):
    return [int(n) for n in s.split(".")]

  url = "https://get.opera.com/pub/opera/desktop/"
  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    dom = Selector(response.text)

  versions = dom.xpath("//a[@href]/@href").getall()[1:]
  versions = [version[:-1] for version in versions]
  versions = sorted(versions, key=version_key, reverse=True)

  number = min(number, len(versions))
  return versions[:number]


def generate_user_agents(num_agents=10):
  def random_safari_version() -> str:
    safari_versions = {
      "16": (0, 6),
      "17": (0, 5),
    }
    major = random.choice(list(safari_versions.keys()))
    minor = random.randint(*safari_versions[major])

    return f"{major}.{minor}"

  browsers = ["Chrome", "Firefox", "Safari", "Opera"]
  operating_systems = [
    "Windows NT 10.0",
    "Macintosh; Intel Mac OS X 10_15_7",
    "X11; Linux x86_64",
  ]

  chrome_versions = get_chrome_versions("win", 10)
  firefox_versions = get_firefox_versions(10)
  opera_versions = get_opera_versions(10)

  user_agents = set()

  while len(user_agents) < num_agents:
    browser = random.choice(browsers)
    os = random.choice(operating_systems)

    if browser == "Chrome":
      chrome = random.choice(chrome_versions)
      user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome} Safari/537.36"
    elif browser == "Firefox":
      firefox = random.choice(firefox_versions)
      user_agent = f"Mozilla/5.0 ({os}; rv:{firefox}) Gecko/20100101 Firefox/{firefox}"
    elif browser == "Safari":
      safari = random_safari_version()
      user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{safari} Safari/605.1.15"
    elif browser == "Opera":
      opera = random.choice(opera_versions)
      chrome = random.choice(chrome_versions)
      user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome} Safari/537.36 OPR/{opera}"

    user_agents.add(user_agent)

  return user_agents


def free_proxy_list() -> list[str]:
  url = "https://free-proxy-list.net/#"

  with httpx.Client() as client:
    response = client.get(url, headers=HEADERS)
    dom = Selector(response.text)

  proxy_table = dom.xpath('.//table[@class="table table-striped table-bordered"]//tr')
  proxies = []
  for row in proxy_table:
    ip = row.xpath(".//td[1]/text()").get()
    port = row.xpath(".//td[2]/text()").get()
    if ip is None or port is None:
      continue

    https = row.xpath(".//td[7]/text()").get()
    prefix = "https://" if https == "yes" else "http://"
    proxies.append(f"{prefix}{ip}:{port}")

  return list(filter(None, proxies))


def proxylist_geonode(limit: int = 500) -> list[str]:
  url = "https://proxylist.geonode.com/api/proxy-list"
  params = {
    "limit": str(limit),
    "page": "1",
    "sort_by": "lastChecked",
    "sort_type": "desc",
  }
  with requests.Session() as client:
    rs = client.get(url, params=params, headers=HEADERS)
    parse = rs.json()

  proxies = [""] * limit
  for i, proxy in enumerate(parse["data"]):
    proxies[i] = proxy["ip"]

  return proxies


async def check_proxies(proxies: list[str]) -> list[bool]:
  url = "https://httpbin.org/ip"  # 'https://ipinfo.io/json'

  async def fetch(proxy: str) -> bool:
    proxy_mounts = {
      "http://": httpx.HTTPTransport(proxy=f"{proxy}"),
      "https://": httpx.HTTPTransport(proxy=f"{proxy}"),
    }

    async with httpx.AsyncClient(mounts=proxy_mounts) as client:
      try:
        response = await client.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.status_code == 200
      except Exception:
        return False

  tasks = [asyncio.create_task(fetch(p)) for p in proxies]
  result = await asyncio.gather(*tasks)

  result = np.array(proxies)[np.array(result)]
  return result


async def fetch_json(url: str, params: dict[str, str] | None = None, **kwargs) -> dict:
  if kwargs.get("impersonate") is None:
    kwargs["impersonate"] = Impersonate.Chrome137

  client = Client(**kwargs)

  if params is not None:
    url += f"?{'&'.join([f'{key}={value}' for key, value in params.items()])}"

  response = await client.get(url)
  return await response.json()


def download_file(url: str, file_path: str | Path):
  with open(file_path, "wb") as file:
    with httpx.stream("GET", url=url, headers=HEADERS) as response:
      total = int(response.headers.get("content-length", 0))
      if response.status_code != 200:
        raise Exception(f"Download failed! Response headers: {response.headers}")

      with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B") as progress:
        bytes_downloaded = response.num_bytes_downloaded
        for chunk in response.iter_bytes():
          file.write(chunk)
          progress.update(response.num_bytes_downloaded - bytes_downloaded)
          bytes_downloaded = response.num_bytes_downloaded


async def download_file_memory(url: str) -> BytesIO:
  async with httpx.AsyncClient() as client:
    response = await client.get(url)
    if response.status_code == 200:
      return BytesIO(response.content)
    else:
      raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
