import ast
from datetime import date
from pathlib import Path
import re
from typing import cast, Annotated, Literal, Optional, TypedDict
import xml.etree.ElementTree as et
import zipfile

import httpx
import numpy as np
import pandas as pd
from pandera import DataFrameModel
from pandera.typing import DataFrame, Series
from parsel import Selector

from lib.db.lite import insert_sqlite, read_sqlite
from lib.utils import download_file

NAMESPACE = {
  "link": "http://www.xbrl.org/2003/linkbase",
  "xbrli": "http://www.xbrl.org/2003/instance",
  "xlink": "http://www.w3.org/1999/xlink",
  "xs": "http://www.w3.org/2001/XMLSchema",
}

IFRS_ = (
  #'http://xbrl.ifrs.org/taxonomy/2014-03-05/IFRST_2014-03-05.zip'
  "http://xbrl.ifrs.org/taxonomy/2015-03-11/IFRST_2015-03-11.zip",
  "http://xbrl.ifrs.org/taxonomy/2016-03-31/IFRST_2016-03-31.zip",
  "http://xbrl.ifrs.org/taxonomy/2017-03-09/IFRST_2017-03-09.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2018-03-16.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2019-03-27.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2020-03-16.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRST_2021-03-24.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRSAT-2022-03-24.zip",
  "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/IFRSAT_2023_03_23.zip",
)

IFRS = {
  2015: "2015-03-11",
  2016: "2016-03-31",
  2017: "2017-03-09",
  2018: "2018-03-16",
  2019: "2019-03-27",
  2020: "2020-03-16",
  2021: "2021-03-24",
  2022: "2022-03-24",
  2023: "2023-03-23",
}


class CalcVisitor(ast.NodeVisitor):
  def __init__(self) -> None:
    self.links: dict[str, Literal["#FF0000", "#008000"]] = {}

  def reset_links(self):
    self.links = {}

  def visit_UnaryOp(self, node):
    sign = "#FF0000" if isinstance(node.op, ast.USub) else "#008000"

    if isinstance(node.operand, ast.Name):
      self.links[node.operand.id] = sign

    self.generic_visit(node)

  def visit_BinOp(self, node):
    sign = "#FF0000" if isinstance(node.op, ast.Sub) else "#008000"

    if isinstance(node.left, ast.Name):
      self.links[node.left.id] = "#008000"

    if isinstance(node.right, ast.Name):
      self.links[node.right.id] = sign

    self.generic_visit(node)


class Taxonomy(DataFrameModel):
  year: int
  name: str
  type: str
  period: Literal["duration", "instant"]
  label: str
  description: str
  calculation: str
  deprecated: str


class XbrlElement(TypedDict):
  name: str
  type: str
  period: Optional[Literal["duration", "instant"]]
  balance: Optional[Literal["credit", "debit"]]


class XbrlLabel(TypedDict):
  name: str
  label: Optional[str]
  deprecated: Optional[int]


def xbrl_namespaces(dom: Selector) -> dict:
  # pattern = r'(?<=^xmlns:)[a-z\-]+$'
  # namespaces = {}
  # for ns, url in cast(bs.Tag, dom.find('xbrl')).attrs.items():
  #  if match := re.search(pattern, ns):
  #    namespaces[match.group()] = url
  #
  # return namespaces

  root = dom.xpath("/*")[0]
  namespaces = root.xpath("namespace-uri() | ./namespace::*")
  namespace_dict = {}
  for i in range(0, len(namespaces), 2):
    prefix = namespaces[i]
    uri = namespaces[i + 1]
    namespace_dict[prefix] = uri

  return namespace_dict


def xbrl_items(root: et.Element) -> DataFrame:
  def parse_type(text: str) -> str:
    return text.replace("ItemType", "").split(":")[-1]

  data: list[XbrlElement] = []
  for item in root.findall(".//xs:element", namespaces=NAMESPACE):
    data.append(
      XbrlElement(
        name=item.attrib["name"],
        type=parse_type(item.attrib["type"]),
        period=cast(
          Literal["duration", "instant"] | None,
          item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}periodType'),
        ),
        balance=cast(
          Literal["credit", "debit"] | None,
          item.attrib.get(f'{{{NAMESPACE["xbrli"]}}}balance'),
        ),
      )
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def ifrs_labels(root: et.Element) -> DataFrame:
  pattern = r" \(Deprecated (?P<year>\d{4})(-\d{2}-\d{2})?\)$"
  data: list[XbrlLabel] = []
  for item in root.findall(".//link:label", namespaces=NAMESPACE):
    deprecated: None | int = None
    label = item.text
    if label is not None and (m := re.search(pattern, label)) is not None:
      label = re.sub(pattern, "", label)
      deprecated = int(m.group("year"))

    data.append(
      XbrlLabel(
        # r'(?<=^ifrs-full_)[a-zA-Z]+(?=_(totalLabel|label)$)'
        name=item.attrib["id"].split("_")[1],
        label=label,
        deprecated=deprecated,
      )
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def ifrs_description(root: et.Element) -> DataFrame:
  data: list[dict[str, str]] = []

  pattern = r"^(?:\d{4}-\d{2}-\d{2}|\d{4}(?: New Element)?|" r"\[\d{4}-\d{2}\] \{.+\})$"

  for item in root.findall(".//link:label", namespaces=NAMESPACE):
    description = cast(str, item.text)
    if re.match(pattern, description) is not None:
      continue

    data.append(
      {  # r'(?<=^ifrs-full_)[a-zA-Z]+(?=_documentation$)'
        "name": item.attrib["id"].split("_")[1],
        "description": description,
      }
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def ifrs_calculation(root: et.Element) -> dict[str, dict[str, dict[str, float]]]:
  names: dict[str, str] = {}
  for loc in root.findall(".//link:loc", namespaces=NAMESPACE):
    key = loc.attrib[f'{{{NAMESPACE["xlink"]}}}label']
    names[key] = loc.attrib[f'{{{NAMESPACE["xlink"]}}}href'].split("_")[-1]

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for calc in root.findall(".//link:calculationArc", namespaces=NAMESPACE):
    parent = calc.attrib[f'{{{NAMESPACE["xlink"]}}}from']
    item = calc.attrib[f'{{{NAMESPACE["xlink"]}}}to']
    schema.setdefault(names[parent], {})[names[item]] = {
      "order": float(calc.attrib["order"]),
      "weight": float(calc.attrib["weight"]),
    }

  for key, value in schema.items():
    schema[key] = {
      subkey: subdict
      for subkey, subdict in sorted(value.items(), key=lambda i: i[1]["order"])
    }

  return schema


def calculation_text(calc: dict[str, dict[str, float]]) -> str:
  def parse_weight(weight: float) -> str:
    result = "- " if weight < 0 else "+ "
    result += "" if (norm := np.abs(weight)) == 1.0 else f"{norm}*"

    return result

  text: list[str] = []

  for k, v in calc.items():
    text.append(f'{parse_weight(v["weight"])}{k}')

  text[0] = text[0].replace("+", "")

  return " ".join(text).strip()


def calculation_df(schema: dict[str, dict[str, dict[str, float]]]) -> DataFrame:
  data = [{"name": k, "calculation": calculation_text(v)} for k, v in schema.items()]

  df = pd.DataFrame(data)
  df.drop_duplicates(inplace=True)
  return cast(DataFrame, df)


def taxonomy_df(
  year: int,
  items: DataFrame,
  labels: DataFrame,
  description: DataFrame,
  calculation: DataFrame,
) -> DataFrame[Taxonomy]:
  items = cast(DataFrame, items.merge(labels, how="left", on="name"))
  items = cast(DataFrame, items.merge(description, how="left", on="name"))
  items = cast(DataFrame, items.merge(calculation, how="left", on="name"))
  items.drop_duplicates(inplace=True)

  duplicates = items.loc[
    items.duplicated(subset=["name", "type"], keep=False), :
  ].copy()
  drop_rows: list[int] = []
  for name in duplicates["name"].unique():
    df = duplicates.loc[duplicates["name"] == name, :].copy()
    drop_rows.append(df["label"].str.len().idxmin())

  items.drop(index=drop_rows, inplace=True)
  items.sort_values("name", inplace=True)
  items.insert(0, "year", year, True)

  return cast(DataFrame[Taxonomy], items)


def cleanup_taxonomy(items: DataFrame[Taxonomy]) -> DataFrame[Taxonomy]:
  subset = ["name", "type", "description", "calculation"]
  items.drop_duplicates(subset=subset, inplace=True)

  deprecated = items.loc[items["deprecated"].notna(), :].copy()
  deprecated.set_index(["name", "type", "deprecated"], inplace=True)

  for ix in deprecated.index.unique():
    mask = (items["name"] == ix[0]) & (items["type"] == ix[1])
    cast(Series, items.loc[mask, "deprecated"]).fillna(ix[2], inplace=True)

  duplicates = items.loc[items.duplicated(subset=subset, keep=False), :].copy()
  drop_rows: list[int] = []
  for name in duplicates["name"].unique():
    df = duplicates.loc[duplicates["name"] == name, :].copy()
    drop_rows.append(df["label"].str.len().idxmin())

  items.sort_values(["name", "year"], inplace=True)
  items.drop(index=drop_rows, inplace=True)
  return items


def gaap_items(year: Annotated[int, ">=2011"]) -> DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-{year}{"-01-31" if year < 2022 else ""}.xsd'
  )
  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  return xbrl_items(root)


def gaap_labels(year: Annotated[int, ">=2011"]) -> DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-lab-{year}{"-01-31" if year < 2022 else ""}.xml'
  )
  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  pattern = r" \(Deprecated (?P<year>\d{4})(-\d{2}-\d{2})?\)$"

  data: list[XbrlLabel] = []
  for item in root.findall(".//link:label", namespaces=NAMESPACE):
    deprecated: None | int = None
    label = cast(str, item.text)
    if (m := re.search(pattern, label)) is not None:
      label = re.sub(pattern, "", label)
      deprecated = int(m.group("year"))

    data.append(
      XbrlLabel(
        name=item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split("_")[-1],
        label=label,
        deprecated=deprecated,
      )
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def gaap_description(year: Annotated[int, ">=2011"]) -> DataFrame:
  url = (
    f'https://xbrl.fasb.org/us-gaap/{year}/elts/'
    f'us-gaap-doc-{year}{"-01-31" if year < 2022 else ""}.xml'
  )
  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

    data: list[dict[str, str]] = []

  pattern = r"^(?:\d{4}-\d{2}-\d{2}|\d{4}(?: New Element)?|" r"\[\d{4}-\d{2}\] \{.+\})$"

  for item in root.findall(".//link:label", namespaces=NAMESPACE):
    description = cast(str, item.text)
    if re.match(pattern, description) is not None:
      continue

    data.append(
      {
        "name": item.attrib[f'{{{NAMESPACE["xlink"]}}}label'].split("_")[-1],
        "description": description,
      }
    )

  df = pd.DataFrame(data)
  return cast(DataFrame, df)


def parse_gaap_calculation_urls(
  year: Annotated[int, ">=2011"], suffix: Literal["stm", "dis"]
) -> list[str]:
  url = f"https://xbrl.fasb.org/us-gaap/{year}/{suffix}/"

  response = httpx.get(url)
  dom = parsel.Selector(response.text)

  pattern = rf'-cal-{year}{"-01-31" if year < 2022 else ""}.xml$'

  urls = dom.xpath(f'//a[re:match(@href, "{pattern}")]/@href').getall()
  urls = [f"https://xbrl.fasb.org/us-gaap/{year}/{suffix}/{url}" for url in urls]

  return urls


def gaap_calculation_urls(year: Annotated[int, ">=2011"]) -> list[str]:
  urls: list[str] = []
  for suffix in ("stm", "dis"):
    urls.extend(parse_gaap_calculation_urls(year, cast(Literal["stm", "dis"], suffix)))

  return urls


def parse_gaap_calculation(url: str) -> dict[str, dict[str, dict[str, float]]]:
  with httpx.Client() as client:
    rs = client.get(url)
    root = et.fromstring(rs.content)

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for calc in root.findall(".//link:calculationArc", namespaces=NAMESPACE):
    parent = calc.attrib[f'{{{NAMESPACE["xlink"]}}}from'].split("_")[-1]

    item = calc.attrib[f'{{{NAMESPACE["xlink"]}}}to'].split("_")[-1]
    schema.setdefault(parent, {})[item] = {
      "order": float(calc.attrib["order"]),
      "weight": float(calc.attrib["weight"]),
    }

  for key, value in schema.items():
    schema[key] = {
      subkey: subdict
      for subkey, subdict in sorted(value.items(), key=lambda i: i[1]["order"])
    }

  return schema


def gaap_calculation(year: Annotated[int, ">=2011"]) -> DataFrame:
  urls = gaap_calculation_urls(year)

  schema: dict[str, dict[str, dict[str, float]]] = {}
  for url in urls:
    schema.update(parse_gaap_calculation(url))

  return calculation_df(schema)


def gaap_taxonomy(year: Annotated[int, ">=2011"]) -> DataFrame[Taxonomy]:
  items = gaap_items(year)
  labels = gaap_labels(year)
  description = gaap_description(year)
  calculation = gaap_calculation(year)

  return taxonomy_df(year, items, labels, description, calculation)


def seed_gaap_taxonomy(end_year: Optional[int] = None):
  if end_year is None:
    end_year = date.today().year

  dfs = [gaap_taxonomy(y) for y in range(2011, end_year + 1)]
  items = cast(DataFrame[Taxonomy], pd.concat(dfs, axis=0, ignore_index=True))

  items = cleanup_taxonomy(items)
  insert_sqlite(items, "taxonomy.db", "gaap", "replace", False)


def ifrs_taxonomy(year: Annotated[int, ">=2015"]) -> DataFrame:
  base_url = (
    f"http://xbrl.ifrs.org/taxonomy/{IFRS[year]}/"
    if year < 2018
    else "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomies/"
  )
  zip_stem = "IFRST" if year < 2022 else "IFRSAT"
  zip_stem += "-" if year == 2022 else "_"
  zip_stem += IFRS[year] if year < 2023 else IFRS[year].replace("-", "_")

  url = f"{base_url}{zip_stem}.zip"

  if year == 2023:
    zip_stem = zip_stem.replace("_", "-")

  zip_path = Path(f"temp/{zip_stem}.zip")
  if not zip_path.exists():
    download_file(url, zip_path)

  if not zipfile.is_zipfile(zip_path):
    raise ValueError(f"Corrupt zip file at {zip_path} retrieved from: {url}")

  with zipfile.ZipFile(zip_path, "r") as zip:
    with zip.open(
      f"{zip_path.stem}/full_ifrs/full_ifrs-cor_{IFRS[year]}.xsd", "r"
    ) as item_file:
      root = et.parse(item_file).getroot()
      items = xbrl_items(root)

    with zip.open(
      f"{zip_path.stem}/full_ifrs/labels/lab_full_ifrs-en_{IFRS[year]}.xml", "r"
    ) as lab_file:
      root = et.parse(lab_file).getroot()
      labels = ifrs_labels(root)

    with zip.open(
      f"{zip_path.stem}/full_ifrs/labels/doc_full_ifrs-en_{IFRS[year]}.xml", "r"
    ) as doc_file:
      root = et.parse(doc_file).getroot()
      description = ifrs_description(root)

    schema: dict[str, dict[str, dict[str, float]]] = {}
    for i in zip.infolist():
      if i.is_dir():
        continue

      file_name = i.filename.split("/")[-1]
      if file_name.startswith("cal") and file_name.endswith(".xml"):
        with zip.open(i.filename) as calc_file:
          root = et.parse(calc_file).getroot()
          schema.update(ifrs_calculation(root))

  zip_path.unlink()
  calculation = calculation_df(schema)

  return taxonomy_df(year, items, labels, description, calculation)


def seed_ifrs_taxonomy(end_year: Optional[int] = None):
  if end_year is None:
    end_year = max(IFRS.keys())

  dfs = [ifrs_taxonomy(y) for y in range(2015, end_year + 1)]
  items = cast(DataFrame[Taxonomy], pd.concat(dfs, axis=0, ignore_index=True))

  items = cleanup_taxonomy(items)
  insert_sqlite(items, "taxonomy.db", "ifrs", "replace", False)


def gaap_network() -> tuple[list[dict[str, int | str]], list[dict[str, str]]] | None:
  query = 'SELECT DISTINCT name, calculation FROM gaap WHERE type = "monetary"'
  df = read_sqlite("taxonomy.db", query)
  if df is None:
    return None

  node_id: set[str] = set()
  edges: list[dict[str, str]] = []

  for node, link_text in zip(df["name"], df["calculation"]):
    if link_text is None:
      continue

    visitor = CalcVisitor()
    visitor.visit(ast.parse(link_text))
    links = visitor.links
    node_id.add(node)

    for link, color in links.items():
      node_id.add(link)

      edges.append({"from": node, "to": link, "color": color})

  nodes: list[dict[str, int | str]] = [{"id": i, "label": i} for i in node_id]

  return nodes, edges
