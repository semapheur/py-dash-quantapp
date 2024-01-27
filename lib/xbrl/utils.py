import re
from typing import cast
import xml.etree.ElementTree as et

import bs4 as bs
import httpx
import numpy as np
import pandas as pd

from lib.db.lite import insert_sqlite, read_sqlite
from lib.fin.taxonomy import extract_items

NAMESPACE = {
  'link': 'http://www.xbrl.org/2003/linkbase',
  'xbrli': 'http://www.xbrl.org/2003/instance',
  'xlink': 'http://www.w3.org/1999/xlink',
  'xs': 'http://www.w3.org/2001/XMLSchema',
}


def xbrl_namespaces(dom: bs.BeautifulSoup) -> dict:
  pattern = r'(?<=^xmlns:)[a-z\-]+$'
  namespaces = {}
  for ns, url in cast(bs.Tag, dom.find('xbrl')).attrs.items():
    if match := re.search(pattern, ns):
      namespaces[match.group()] = url

  return namespaces


def gaap_items(year: int) -> pd.DataFrame:
  def parse_type(text: str) -> str:
    return text.replace('ItemType', '').split(':')[-1]

  url = f'https://xbrl.fasb.org/us-gaap/{year}/elts/us-gaap-{year}.xsd'

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  data: list[dict[str, str | None]] = []

  for item in root.findall('.//xs:element', namespaces=NAMESPACE):
    data.append(
      {
        'name': item.attrib['name'],
        'type': parse_type(item.attrib['type']),
        'period': item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}periodType'),
        'balance': item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}balance'),
      }
    )

  df = pd.DataFrame(data)
  return df


def gaap_labels(year: int) -> pd.DataFrame:
  url = f'https://xbrl.fasb.org/us-gaap/{year}/elts/us-gaap-lab-{year}.xml'

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  data: list[dict[str, str]] = []
  for item in root.findall('.//link:label', namespaces=NAMESPACE):
    data.append(
      {
        'name': item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split('_')[-1],
        'label': cast(str, item.text),
      }
    )

  df = pd.DataFrame(data)
  return df


def gaap_description(year: int) -> pd.DataFrame:
  url = f'https://xbrl.fasb.org/us-gaap/{year}/elts/us-gaap-doc-{year}.xml'

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  data: list[dict[str, str]] = []

  for item in root.findall('.//link:label', namespaces=NAMESPACE):
    data.append(
      {
        'name': item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split('_')[-1],
        'description': cast(str, item.text),
      }
    )

  df = pd.DataFrame(data)
  return df


def gaap_calculation_url(year: int) -> list[str]:
  url = f'https://xbrl.fasb.org/us-gaap/{year}/stm/'

  with httpx.Client() as client:
    rs = client.get(url)
    dom = bs.BeautifulSoup(rs.text, 'lxml')

  pattern = rf'-cal-{year}.xml$'

  urls: list[str] = []
  for a in dom.find_all('a', href=True):
    slug = a.get('href')
    if re.search(pattern, slug):
      urls.append(f'https://xbrl.fasb.org/us-gaap/{year}/stm/{slug}')

  return urls


def parse_gaap_calculation(url: str) -> dict[str, dict[str, dict[str, float]]]:
  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for calc in root.findall('.//link:calculationArc', namespaces=NAMESPACE):
    parent = calc.attrib[f'{{{NAMESPACE["xlink"]}}}from'].split('_')[-1]

    item = calc.attrib[f'{{{NAMESPACE["xlink"]}}}to'].split('_')[-1]
    schema.setdefault(parent, {})[item] = {
      'order': float(calc.attrib['order']),
      'weight': float(calc.attrib['weight']),
    }

  return schema


def gaap_calculation(year: int) -> pd.DataFrame:
  def calculation_text(calc: dict[str, dict[str, float]]) -> str:
    def parse_weight(weight: float) -> str:
      result = '- ' if weight < 0 else '+ '
      result += '' if (norm := np.abs(weight)) == 1.0 else f'{norm}*'

      return result

    text: list[str] = []

    for k, v in calc.items():
      text.append(f'{parse_weight(v["weight"])}{k}')

    text[0] = text[0].replace('+', '')

    return ' '.join(text).strip()

  urls = gaap_calculation_url(year)

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for url in urls:
    schema.update(parse_gaap_calculation(url))

  data = [{'name': k, 'calculation': calculation_text(v)} for k, v in schema.items()]

  df = pd.DataFrame(data)
  return df


def gaap_taxonomy(year: int):
  items = gaap_items(year)
  labels = gaap_labels(year)
  description = gaap_description(year)
  calculation = gaap_calculation(year)

  items = items.merge(labels, how='left', on='name')
  items = items.merge(description, how='left', on='name')
  items = items.merge(calculation, how='left', on='name')
  items.drop_duplicates(inplace=True)
  items.sort_values('name', inplace=True)
  insert_sqlite(items, 'taxonomy.db', 'gaap', 'replace', False)
  return items


def gaap_network() -> list[dict[str, str]]:
  query = 'SELECT DISTINCT name, calculation FROM gaap WHERE type = "monetary" AND calculation IS NOT NULL'
  df = read_sqlite('taxonomy.db', query)

  data: list[dict[str, str]] = []

  for node, link_text in zip(df['name'], df['calculation']):
    links = extract_items(link_text)

    for link in links:
      data.append({'from': node, 'to': link})

  return data
