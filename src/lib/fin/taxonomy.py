import ast
from collections import defaultdict
from contextlib import closing
import json
import re
import sqlite3
from typing import cast, Literal, Optional, TypeAlias
from typing_extensions import TypedDict

from glom import glom
import pandas as pd
from pandera.typing import DataFrame
from pydantic import BaseModel, field_validator, model_serializer

from lib.db.lite import get_tables, insert_sqlite, read_sqlite, create_fts_table
from lib.const import DB_DIR

TaxonomyType: TypeAlias = Literal[
  "days",
  "monetary",
  "fundamental",
  "percent",
  "per_day",
  "per_share",
  "personnel",
  "ratio",
  "shares",
]


class AggregateItems(ast.NodeVisitor):
  def __init__(self) -> None:
    self.items: set[str] = set()

  def reset_names(self):
    self.items = set()

  def visit_Name(self, node):
    if node.id.startswith(("average_", "change_")):
      self.items.add(node.id)

    self.generic_visit(node)


class TaxonomyLabel(TypedDict, total=False):
  long: str
  short: Optional[str]


class TaxonomyCalculation(TypedDict, total=False):
  order: int
  all: Optional[list[str]]
  any: Optional[list[str]]
  avg: Optional[str]
  diff: Optional[str]
  fill: Optional[str]
  shift: Optional[str]
  min: Optional[str | float]
  max: Optional[str | float]


class TaxonomyItem(BaseModel):
  type: TaxonomyType
  balance: Optional[Literal["credit", "debit"]] = None
  aggregate: Literal["average", "recalc", "sum", "tail"]
  label: TaxonomyLabel
  gaap: Optional[list[str]] = None
  calculation: Optional[TaxonomyCalculation] = None
  components: Optional[list[str]] = None

  @field_validator("calculation", mode="before")
  @classmethod
  def validate_calculation(cls, value):
    if value is None:
      return value

    if "all" in value and isinstance(value["all"], str):
      value["all"] = [value["all"]]

    if "any" in value and isinstance(value["any"], str):
      value["any"] = [value["any"]]

    return value


class TaxonomyRecord(TypedDict):
  item: str
  type: TaxonomyType
  balance: Optional[Literal["credit", "debit"]]
  aggregate: Literal["average", "recalc", "sum", "tail"]
  long: str
  short: Optional[str]
  gaap: Optional[str]
  calculation: Optional[str]


class Taxonomy(BaseModel):
  data: dict[str, TaxonomyItem]

  @model_serializer
  def ser_model(self) -> dict[str, TaxonomyItem]:
    return dict(sorted(self.data.items()))

  def filter_items(self, filter_: set[str]):
    new_keys = set(self.data.keys()).intersection(filter_)
    self.data = {key: value for key, value in self.data.items() if key in new_keys}

  def add_missing_items(self) -> None:
    visitor = AggregateItems()

    calc_items: set[str] = set()

    for v in self.data.values():
      if (calc := v.calculation) is None:
        continue

      calc_value = calc.get("all", calc.get("any"))

      if isinstance(calc_value, str):
        visitor.visit(ast.parse(calc_value))

      elif isinstance(calc_value, dict):
        calc_items.union(calc_value.keys())

    calc_items.union(visitor.items)
    if not calc_items:
      return

    present_items = set(self.data.keys())
    missing_items = calc_items.difference(present_items)

    for i in missing_items:
      prefix, base_item = i.split("_", 1)

      if base_item not in present_items:
        print(f"{base_item} is missing in the taxonomy!")
        continue

      if (label := self.data[base_item].label) is None:
        continue

      short = label.get("short")
      base_item_calc = self.data[base_item].calculation
      order = 0 if base_item_calc is None else max(0, base_item_calc["order"] - 1)

      if prefix == "average":
        calc = TaxonomyCalculation(order=order, avg=base_item)
        label = TaxonomyLabel(
          long=f'Average {label.get("long")}',
          short=None if short is None else f"Average {short}",
        )
      elif prefix == "change":
        calc = TaxonomyCalculation(order=order, diff=base_item)
        label = TaxonomyLabel(
          long=f'Change in {label.get("long")}',
          short=None if short is None else f"Change in {short}",
        )

      self.data[i] = TaxonomyItem(
        type=self.data[base_item].type,
        aggregate="average" if prefix == "average" else "sum",
        gaap=[],
        label=label,
        calculation=calc,
      )

  def fix_balance(self) -> None:
    query = """
      SELECT i.item, g.balance FROM items i
      JOIN json_each(i.gaap)
      JOIN (
        SELECT name, balance, deprecated, MAX(year) as max_year
        FROM gaap
        GROUP BY name
      ) g ON json_each.value = g.name
      WHERE json_each.value IS NOT NULL 
        AND g.deprecated IS NULL 
        AND i.balance != g.balance
    """

    df = read_sqlite("taxonomy.db", query)
    if df is None or df.empty:
      return

    for item, balance in zip(df["item"], df["balance"]):
      self.data[item].balance = balance

  def resolve_calculation_order(self) -> None:
    order_dict: defaultdict[str, set[str]] = defaultdict(set)

    for k, v in self.data.items():
      if (calc := v.calculation) is None:
        continue

      for i, calc_values in calc.items():
        if i == "order" or isinstance(calc_values, float):
          continue

        calc_values = (
          [calc_values]
          if isinstance(calc_values, str)
          else cast(list[str], calc_values)
        )

        for calc_value in calc_values:
          order_dict[k].update(extract_items(calc_value))

    visited: set[str] = set()
    result: list[str] = []

    def topological_sort(node: str):
      if node in visited:
        return

      visited.add(node)
      for neighbor in order_dict[node]:
        topological_sort(neighbor)
      result.append(node)

    for k in self.data:
      topological_sort(k)

    for k, v in self.data.items():
      if (calc := v.calculation) is not None:
        calc["order"] = result.index(k)

  def remove_duplicate_synonyms(self) -> None:
    for k, v in self.data.items():
      if (gaap := v.gaap) is None:
        continue

      unique_gaap: dict[str, tuple[str, int]] = {}

      for g in gaap:
        lower_g = g.lower()
        capitals = len(re.findall(r"[A-Z]", g))

        if lower_g not in unique_gaap:
          unique_gaap[lower_g] = (g, capitals)
        else:
          if capitals > unique_gaap[lower_g][1]:
            unique_gaap[lower_g] = (g, capitals)

      self.data[k].gaap = [v[0] for v in unique_gaap.values()]

  def select_items(self, target_key, target_value: tuple[str, str]) -> set[str]:
    result = {
      k
      for k, v in self.data.items()
      if glom(v.model_dump(exclude_none=True), target_key) == target_value
    }
    return result

  def rename_schema(self) -> dict[str, str]:
    schema = {
      name: k for k, v in self.data.items() if (names := v.gaap) for name in names
    }
    return schema

  def item_names(self) -> set[str]:
    names = {name for v in self.data.values() if (names := v.gaap) for name in names}
    return names

  def labels(self) -> pd.DataFrame:
    df_data = [
      (k, v.label.get("long", ""), v.label.get("short", ""))
      for k, v in self.data.items()
      if v.label is not None
    ]
    return pd.DataFrame(df_data, columns=["item", "long", "short"])

  def calculation_schema(
    self, select: Optional[set[str]] = None
  ) -> dict[str, TaxonomyCalculation]:
    keys = set(self.data.keys())
    if select:
      keys = keys.intersection(select)
      if not keys:
        return {}

    schema = {
      key: calc for key in keys if (calc := self.data[key].calculation) is not None
    }
    return schema

  def extra_calculation_schema(self) -> dict[str, TaxonomyCalculation]:
    schema = {
      k: calc
      for k, v in self.data.items()
      if v.gaap is not None and (calc := v.calculation)
    }
    return schema

  def to_records(
    self,
  ) -> list[TaxonomyRecord]:
    return [
      TaxonomyRecord(
        item=k,
        type=v.type,
        balance=v.balance,
        aggregate=v.aggregate,
        long=v.label["long"],
        short=v.label.get("short"),
        gaap=None if v.gaap is None else json.dumps(v.gaap),
        calculation=None if v.calculation is None else json.dumps(v.calculation),
      )
      for k, v in self.data.items()
    ]

  def to_sql(self, db_name="taxonomy.db"):
    data = self.to_records()
    df = pd.DataFrame(data)

    insert_sqlite(df, db_name, "items", "replace", False)

  def to_sqlite(self, db_path: str):
    with closing(sqlite3.connect(db_path)) as con:
      cur = con.cursor()

      records = self.to_records()
      cur.execute("DROP TABLE IF EXISTS items")
      cur.execute("""CREATE TABLE IF NOT EXISTS items (
        item TEXT PRIMARY KEY,
        type TEXT,
        balance TEXT,
        aggregate TEXT,
        long TEXT,
        short TEXT,
        gaap TEXT,
        calculation TEXT)
      """)
      query = """INSERT INTO items
        VALUES (:item, :type, :balance, :aggregate, :long, :short, :gaap, :calculation)
      """
      cur.executemany(query, records)
      con.commit()


def extract_items(calc_text: str) -> set[str]:
  class CalcItems(ast.NodeVisitor):
    def __init__(self) -> None:
      self.items: set[str] = set()

    def visit_Name(self, node):
      self.items.add(node.id)
      self.generic_visit(node)

  tree = ast.parse(calc_text)
  visitor = CalcItems()
  visitor.visit(tree)
  return visitor.items


def item_hiearchy(items: str | list[str], levels=3):
  def query(x: str | list[str]):
    item_where = f"= '{x}'" if isinstance(x, str) else f"IN {str(tuple(x))}"
    return f"""
      SELECT item, json_extract(calculation, '$.any[0]') AS child FROM items 
      WHERE child IS NOT NULL AND item {item_where}
    """

  def build_hierarchy(current_item: str, current_level: int) -> dict[str, dict]:
    if current_level == 0 or not current_item:
      return {}

    df = read_sqlite("taxonomy.db", query(current_item))
    if df is None or df.empty:
      return {}

    df.loc[:, "child"] = df["child"].apply(extract_items)
    df = cast(DataFrame, df.explode("child"))
    result = {child: build_hierarchy(child, current_level - 1) for child in df["child"]}

    return result

  df = read_sqlite("taxonomy.db", query(items))
  if df is None or df.empty:
    raise ValueError(f"Item hierarchy not found for {items}")

  df.loc[:, "child"] = df["child"].apply(extract_items)
  df = cast(DataFrame, df.explode("child"))

  result: dict[str, dict] = {item: {} for item in items}
  for i, c in zip(df["item"], df["child"]):
    result[i][c] = build_hierarchy(c, levels - 1)

  return result


def flatten_hierarchy(nested_dict, level=1, flattened=None):
  if flattened is None:
    flattened = {}

  for key, value in nested_dict.items():
    flattened[key] = level
    if isinstance(value, dict):
      flatten_hierarchy(value, level + 1, flattened)

  return flattened


def find_missing_items(taxonomy: Taxonomy) -> set[str]:
  item_set = set(taxonomy.data.keys())

  missing_items: set[str] = set()

  for v in taxonomy.data.values():
    if (calc := v.calculation) is None:
      continue

    calc_any = calc.get("any") or []

    for calc_text in calc_any:
      items = extract_items(calc_text)

      missing_items.update(items.difference(item_set))

  return missing_items


def load_taxonomy(
  path: str = "lex/fin_taxonomy.json", filter_: Optional[set[str]] = None
) -> Taxonomy:
  with open(path, "r") as file:
    data = json.load(file)

  taxonomy = Taxonomy(data=data)
  if filter_ is not None:
    taxonomy.filter_items(filter_)

  return taxonomy


def search_taxonomy(term: str):
  query = """
    SELECT item, type, balance, aggregate, long, short FROM items 
    WHERE EXISTS (
      SELECT 1 FROM json_each(items.gaap) WHERE lower(json_each.value) = :term
    )
  """
  return read_sqlite("taxonomy.db", query, {"term": term.lower()})


def fuzzy_search_taxonomy(term: str):
  query = """
    SELECT item, gaap
    FROM items_fts
    WHERE gaap MATCH :term
  """

  return read_sqlite("taxonomy.db", query, {"term": term.lower()})


def create_taxonomy_fts():
  db_path = DB_DIR / "taxonomy.db"

  create_fts_table("taxonomy.db", "items_fts", ["item", "gaap"], "unicode61")

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()
    cur.execute("""
      INSERT INTO items_fts (item, gaap)
      SELECT item, json_each.value AS gaap FROM items 
      JOIN json_each(gaap) ON 1=1
      WHERE gaap IS NOT NULL;
    """)
    con.commit()


def merge_labels(template: pd.DataFrame, taxonomy: Taxonomy):
  template = template.merge(taxonomy.labels(), on="item", how="left")
  mask = template["short"] == ""
  template.loc[mask, "short"] = template.loc[mask, "long"]
  return template


def gaap_items(sort=False) -> set[str]:
  db_path = DB_DIR / "taxonomy.db"

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    query = """
      SELECT json_each.value AS gaap FROM items 
      JOIN json_each(gaap) ON 1=1
      WHERE gaap IS NOT NULL
    """

    cur.execute(query)
    result = cur.fetchall()

  items = {x[0] for x in result}

  if sort:
    items = set(sorted(items))

  return items


def scraped_items(sort=False) -> set[str]:
  tables = get_tables("statements.db")

  db_path = DB_DIR / "statements.db"
  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    items: set[str] = set()
    for t in tables:
      cur.execute(f'SELECT DISTINCT key FROM "{t[0]}", json_each(data)')
      result = cur.fetchall()
      items = items.union({x[0] for x in result})

  if sort:
    items = set(sorted(items))

  return items


def backup_taxonomy():
  from pathlib import Path
  from shutil import copy

  backup_dir = Path("lex/backup")
  taxonomy_file = Path("lex/fin_taxonomy.json")

  if not backup_dir.exists():
    backup_dir.mkdir(exist_ok=True)

  backup_file = backup_dir / taxonomy_file.name

  if backup_file.exists():
    backup_file.unlink()

  # Move and replace the file
  copy(taxonomy_file, backup_dir)
