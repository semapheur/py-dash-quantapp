import re
import xml.etree.ElementTree as et

import bs4 as bs
import httpx
import pandas as pd

from lib.db.lite import insert_sqlite

def gaap_items(year: int = 2023) -> pd.DataFrame:

  def parse_type(text: str):
    return text \
      .replace('ItemType', '') \
      .split(':')[1]
  
  url = f'https://xbrl.fasb.org/us-gaap/{year}/elts/us-gaap-{year}.xsd'

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  data: list[dict[str, str]] = []

  namespace = {
    'xbrli': 'http://www.xbrl.org/2003/instance',
    'xs': 'http://www.w3.org/2001/XMLSchema',
  }
  for item in root.findall('.//xs:element', namespaces=namespace):
    data.append({
      'name': item.get('name'),
      'type': parse_type(item.get('type')),
      'period': item.get(f'{{{namespace["xbrli"]}}}periodType', ''),
      'balance': item.get(f'{{{namespace["xbrli"]}}}balance', '')
    })
  
  df = pd.DataFrame(data)
  return df

def gaap_description(year: int) -> pd.DataFrame:

  def parse_name(text: str):
    return text.replace('lab_', '')
  
  url = f'https://xbrl.fasb.org/us-gaap/{year}/elts/us-gaap-doc-{year}.xml'

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  data: list[dict[str, str]] = []
  namespace = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink'
  }
  for item in root.findall('.//link:label', namespaces=namespace):
    data.append({
      'name': parse_name(item.get(f'{{{namespace["xlink"]}}}label', '')),
      'description': item.text
    })

  df = pd.DataFrame(data)
  return df
    
def gaap_taxonomy(year: int) -> pd.DataFrame:
  items = gaap_items(year)
  description = gaap_description(year)

  result = items.merge(description, how='left', on='name')
  result.sort_values('name', inplace=True)
  insert_sqlite(result, 'taxonomy', 'gaap', 'replace')

def gaap_calculation_url(year: int = 2023) -> list[str]:

  url = f'https://xbrl.fasb.org/us-gaap/{year}/stm/'

  with httpx.Client() as client:
    rs = client.get(url)
    dom = bs.BeautifulSoup(rs.text, 'lxml')

  pattern = rf'^.+-cal-{year}.xml$'

  urls = []
  for a in dom.find_all('a', href=True):
    match = re.search(pattern, a.get('href'))
    if match:
      urls.append(match.group)

  return urls

def gaap_calculation(url: str) -> pd.DataFrame:
  namespace = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink'
  }

  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  sheet = root.find('.//link:calculationLink', namespaces=namespace) \
    .get(f'{{{namespace["xlink"]}}}role') \
    .split('/')[-1]

  data = {}
  for calc in root.findall('.//link:calculationArc', namespaces=namespace):
    parent = calc.get(f'{{{namespace["xlink"]}}}from')

    item = calc.get(f'{{{namespace["xlink"]}}}to')
    schema = {
      item: {
        'order': float(calc.get('order')),
        'weight': float(calc.get('weight'))
      }
    }
    
    if parent not in data:
      data[parent] = schema
    
    else:
      data[parent].update(schema)

  return data

  
'''
  taxonomy = []
  for sheet in root.findall('.//link:calculationLink', namespaces=namespace):
    sheet_label = re.sub(url_pattern, '', sheet.attrib[f'{{{namespace["xlink"]}}}role'])
    sheet_label = rename_sheet(sheet_label)

    for el in sheet.findall('.//link:calculationArc', namespaces=namespace):
      taxonomy.append({
        'sheet': sheet_label,
        'gaap': re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}to']).group(),
        'parent': re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}from']).group(),
      })
  
  df = pd.DataFrame.from_records(taxonomy)
  df.set_index('item', inplace=True)
  df.drop_duplicates(inplace=True)
  return df
'''