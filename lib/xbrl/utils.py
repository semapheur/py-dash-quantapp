import ast
from datetime import date
import re
from typing import cast, Annotated, Literal, Optional, TypedDict
import xml.etree.ElementTree as et

import bs4 as bs
import httpx
import numpy as np
import pandas as pd
from pandera import DataFrameModel
from pandera.typing import DataFrame, Series

from lib.db.lite import insert_sqlite, read_sqlite

NAMESPACE = {
  'link': 'http://www.xbrl.org/2003/linkbase',
  'xbrli': 'http://www.xbrl.org/2003/instance',
  'xlink': 'http://www.w3.org/1999/xlink',
  'xs': 'http://www.w3.org/2001/XMLSchema',
}

IRFS = (
  #'http://xbrl.ifrs.org/taxonomy/2014-03-05/IFRST_2014-03-05.zip'
  'http://xbrl.ifrs.org/taxonomy/2015-03-11/IFRST_2015-03-11.zip',
  'http://xbrl.ifrs.org/taxonomy/2016-03-31/IFRST_2016-03-31.zip',
  'http://xbrl.ifrs.org/taxonomy/2017-03-09/IFRST_2017-03-09.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2018-03-16.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2019-03-27.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2020-03-16.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2021-03-24.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRSAT-2022-03-24.zip',
  'https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRSAT_2023_03_23.zip',
)


class GaapTaxonomy(DataFrameModel):
  year: int
  name: str
  type: str
  period: Literal['duration', 'instant']
  label: str
  description: str
  calculation: str
  deprecated: str


class XbrlElement(TypedDict):
  name: str
  type: str
  period: Optional[Literal['duration', 'instant']]
  balance: Optional[Literal['credit', 'debit']]


class XbrlLabel(TypedDict):
  name: str
  label: str
  deprecated: Optional[int]


def xbrl_namespaces(dom: bs.BeautifulSoup) -> dict:
  pattern = r'(?<=^xmlns:)[a-z\-]+$'
  namespaces = {}
  for ns, url in cast(bs.Tag, dom.find('xbrl')).attrs.items():
    if match := re.search(pattern, ns):
      namespaces[match.group()] = url

  return namespaces


def xbrl_items(tree: et.Element) -> DataFrame:
  def parse_type(text: str) -> str:
    return text.replace('ItemType', '').split(':')[-1]

  data: list[XbrlElement] = []
  for item in tree.findall('.//xs:element', namespaces=NAMESPACE):
    data.append(
      XbrlElement(
        name=item.attrib['name'],
        type=parse_type(item.attrib['type']),
        period=cast(
          Literal['duration', 'instant'] | None,
          item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}periodType'),
        ),
        balance=cast(
          Literal['credit', 'debit'] | None,
          item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}balance'),
        ),
      )
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def xbrl_labels(tree: et.Element) -> DataFrame:
  pattern = r' \(Deprecated (?P<year>\d{4})(-\d{2}-\d{2})?\)$'

  data: list[XbrlLabel] = []
  for item in tree.findall('.//link:label', namespaces=NAMESPACE):
    deprecated: None | int = None
    label = cast(str, item.text)
    if (m := re.search(pattern, label)) is not None:
      label = re.sub(pattern, '', label)
      deprecated = int(m.group('year'))

    data.append(
      XbrlLabel(
        name=item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split('_')[-1],
        label=label,
        deprecated=deprecated,
      )
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def xbrl_description(tree: et.Element) -> DataFrame:
  data: list[dict[str, str]] = []

  pattern = r'^(?:\d{4}-\d{2}-\d{2}|\d{4}(?: New Element)?|' r'\[\d{4}-\d{2}\] \{.+\})$'

  for item in tree.findall('.//link:label', namespaces=NAMESPACE):
    description = cast(str, item.text)
    if re.match(pattern, description) is not None:
      continue

    data.append(
      {
        'name': item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split('_')[-1],
        'description': description,
      }
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def gaap_items(year: Annotated[int, '>=2011']) -> DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-{year}{"-01-31" if year < 2022 else ""}.xsd'
  )
  with httpx.Client() as client:
    rs = client.get(url)
    tree = et.fromstring(rs.content)

  return xbrl_items(tree)


def gaap_labels(year: Annotated[int, '>=2011']) -> DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-lab-{year}{"-01-31" if year < 2022 else ""}.xml'
  )
  with httpx.Client() as client:
    rs = client.get(url)
    tree = et.fromstring(rs.content)

  return xbrl_labels(tree)


def gaap_description(year: Annotated[int, '>=2011']) -> pd.DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-doc-{year}{"-01-31" if year < 2022 else ""}.xml'
  )

  with httpx.Client() as client:
    rs = client.get(url)
    tree = et.fromstring(rs.content)

  return xbrl_description(tree)


def parse_gaap_calculation_urls(
  year: Annotated[int, '>=2011'], suffix: Literal['stm', 'dis']
) -> list[str]:
  url = f'https://xbrl.fasb.org/us-gaap/{year}/{suffix}/'

  with httpx.Client() as client:
    rs = client.get(url)
    dom = bs.BeautifulSoup(rs.text, 'lxml')

  pattern = rf'-cal-{year}{"-01-31" if year < 2022 else ""}.xml$'

  urls: list[str] = []
  for a in dom.find_all('a', href=True):
    slug = a.get('href')
    if re.search(pattern, slug):
      urls.append(f'https://xbrl.fasb.org/us-gaap/{year}/{suffix}/{slug}')

  return urls


def gaap_calculation_urls(year: Annotated[int, '>=2011']) -> list[str]:
  urls: list[str] = []
  for suffix in ('stm', 'dis'):
    urls.extend(parse_gaap_calculation_urls(year, cast(Literal['stm', 'dis'], suffix)))

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

  for key, value in schema.items():
    schema[key] = {
      subkey: subdict
      for subkey, subdict in sorted(value.items(), key=lambda i: i[1]['order'])
    }

  return schema


def gaap_calculation(year: Annotated[int, '>=2011']) -> pd.DataFrame:
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

  urls = gaap_calculation_urls(year)

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for url in urls:
    schema.update(parse_gaap_calculation(url))

  data = [{'name': k, 'calculation': calculation_text(v)} for k, v in schema.items()]

  df = pd.DataFrame(data)
  df.drop_duplicates(inplace=True)
  return df


def gaap_taxonomy(year: Annotated[int, '>=2011']) -> DataFrame[GaapTaxonomy]:
  items = gaap_items(year)
  labels = gaap_labels(year)
  description = gaap_description(year)
  calculation = gaap_calculation(year)

  items = cast(DataFrame, items.merge(labels, how='left', on='name'))
  items = cast(DataFrame, items.merge(description, how='left', on='name'))
  items = cast(DataFrame, items.merge(calculation, how='left', on='name'))
  items.drop_duplicates(inplace=True)

  duplicates = items.loc[
    items.duplicated(subset=['name', 'type'], keep=False), :
  ].copy()
  drop_rows: list[int] = []
  for name in duplicates['name'].unique():
    df = duplicates.loc[duplicates['name'] == name, :].copy()
    drop_rows.append(df['label'].str.len().idxmin())

  items.drop(index=drop_rows, inplace=True)
  items.sort_values('name', inplace=True)
  items.insert(0, 'year', year, True)

  return cast(DataFrame[GaapTaxonomy], items)


def seed_gaap_taxonomy(end_year: Optional[int] = None):
  if end_year is None:
    end_year = date.today().year

  dfs = [gaap_taxonomy(y) for y in range(2011, end_year)]
  items = pd.concat(dfs, axis=0, ignore_index=True)

  items.drop_duplicates(
    subset=['name', 'label', 'description', 'calculation'], inplace=True
  )

  deprecated = items.loc[items['deprecated'].notna(), :].copy()
  deprecated.set_index(['name', 'type', 'deprecated'], inplace=True)

  for ix in deprecated.index.unique():
    mask = (items['name'] == ix[0]) & (items['type'] == ix[1])
    cast(Series, items.loc[mask, 'deprecated']).fillna(ix[2], inplace=True)

  duplicates = items.loc[
    items.duplicated(subset=['name', 'type', 'description'], keep=False), :
  ].copy()
  drop_rows: list[int] = []
  for name in duplicates['name'].unique():
    df = duplicates.loc[duplicates['name'] == name, :].copy()
    drop_rows.append(df['label'].str.len().idxmin())

  items.sort_values(['name', 'year'], inplace=True)
  items.drop(index=drop_rows, inplace=True)
  insert_sqlite(items, 'taxonomy.db', 'gaap', 'replace', False)


class CalcVisitor(ast.NodeVisitor):
  def __init__(self) -> None:
    self.links: dict[str, Literal['#FF0000', '#008000']] = {}

  def reset_links(self):
    self.links = {}

  def visit_UnaryOp(self, node):
    sign = '#FF0000' if isinstance(node.op, ast.USub) else '#008000'

    if isinstance(node.operand, ast.Name):
      self.links[node.operand.id] = sign

    self.generic_visit(node)

  def visit_BinOp(self, node):
    sign = '#FF0000' if isinstance(node.op, ast.Sub) else '#008000'

    if isinstance(node.left, ast.Name):
      self.links[node.left.id] = '#008000'

    if isinstance(node.right, ast.Name):
      self.links[node.right.id] = sign

    self.generic_visit(node)


def gaap_network() -> tuple[list[dict[str, int | str]], list[dict[str, str]]] | None:
  query = 'SELECT DISTINCT name, calculation FROM gaap WHERE type = "monetary"'
  df = read_sqlite('taxonomy.db', query)
  if df is None:
    return None

  node_id: set[str] = set()
  edges: list[dict[str, str]] = []

  for node, link_text in zip(df['name'], df['calculation']):
    if link_text is None:
      continue

    visitor = CalcVisitor()
    visitor.visit(ast.parse(link_text))
    links = visitor.links
    node_id.add(node)

    for link, color in links.items():
      node_id.add(link)

      edges.append({'from': node, 'to': link, 'color': color})

  nodes: list[dict[str, int | str]] = [{'id': i, 'label': i} for i in node_id]

  return nodes, edges
