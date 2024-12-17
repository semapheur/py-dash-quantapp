import json
from typing import Literal, Optional, TypedDict

import pandas as pd

from lib.db.lite import insert_sqlite, read_sqlite
from lib.fin.taxonomy import extract_items


class Item(TypedDict, total=False):
  balance: Literal["debit", "credit"]
  children: list[str]


class SankeyLink(TypedDict):
  direction: Literal["in", "out"]
  color: str


class SankeyNode(TypedDict, total=False):
  color: str
  links: Optional[dict[str, SankeyLink]]


def load_template(cat: str) -> pd.DataFrame:
  with open("lex/fin_template.json", "r") as file:
    template = json.load(file)

  template = template[cat]

  if cat == "statement":
    data: list = [
      (sheet, item, level)
      for sheet, values in template.items()
      for item, level in values.items()
    ]
    cols: tuple = ("sheet", "item", "level")

  elif cat == "fundamentals":
    data = [(sheet, item) for sheet, values in template.items() for item in values]
    cols = ("sheet", "item")

  elif cat == "sankey":
    data = [
      (sheet, item, entry.get("color", ""), entry.get("links"))
      for sheet, values in template.items()
      for item, entry in values.items()
    ]
    cols = ("sheet", "item", "color", "links")

  elif cat == "dupont":
    data = template
    cols = ("item",)

  return pd.DataFrame(data, columns=cols)


def template_to_sql(db_name: str):
  for template in ("statement", "fundamentals", "sankey", "dupont"):
    df = load_template(template)

    if template == "sankey":
      mask = df["links"].notnull()
      df.loc[mask, "links"] = df.loc[mask, "links"].apply(lambda x: json.dumps(x))

    insert_sqlite(df, db_name, template, "replace", False)


def sankey_hierarchy(root: str, depth: int) -> dict[str, Item]:
  result: dict[str, Item] = {}

  def query(x: str | list[str]):
    item_where = f"= '{x}'" if isinstance(x, str) else f"IN {str(tuple(x))}"
    return f"""
      SELECT item, balance, 
        COALESCE(json_extract(calculation, '$.any[0]'), json_extract(calculation, '$.all[0]')) AS child 
      FROM items 
      WHERE item {item_where}
    """

  def build_hierarchy(current_item: str, current_level: int):
    df = read_sqlite("taxonomy.db", query(current_item))
    if df is None or df.empty:
      return

    balance = df["balance"].iloc[0]
    if current_level == 0 or df["child"].iloc[0] is None:
      result[current_item] = {
        "balance": balance,
      }
      return

    df["child"] = df["child"].apply(extract_items)
    df = df.explode("child")

    result[current_item] = {
      "balance": balance,
      "children": df["child"].tolist(),
    }

    for child in df["child"]:
      build_hierarchy(child, current_level - 1)

  build_hierarchy(root, depth)

  return result


def sankey_graph_dict(root: str, depth: int) -> dict:
  items = sankey_hierarchy(root, depth)

  def build_hierarchy(current_item: str):
    balance = items[current_item]["balance"]
    node: SankeyNode = {
      "color": "rgba(0,255,0,1)" if balance == "debit" else "rgba(255,0,0,1)",
    }
    children = items.get(current_item, {}).get("children", [])
    if children:
      node["links"] = {}

    for child in children:
      child_balance = items[child]["balance"]

      node["links"][child] = {
        "direction": "in" if child_balance == "debit" else "out",
        "color": "rgba(0,255,0,0.3)"
        if child_balance == "debit"
        else "rgba(255,0,0,0.3)",
      }

    return node, children

  def build_graph(item):
    graph = {}
    queue = [item]
    visited = set()

    while queue:
      current_item = queue.pop(0)
      if current_item in visited:
        continue

      visited.add(current_item)
      node, children = build_hierarchy(current_item)
      graph[current_item] = node
      queue.extend(children)

    return graph

  return build_graph(root)


def sankey_balance(depth: int) -> dict:
  sankey = {
    "assets": {
      "color": "rgba(0,0,255,1)",
      "links": {
        "assets_current": {"direction": "in", "color": "rgba(0,255,0,0.3)"},
        "assets_noncurrent": {"direction": "in", "color": "rgba(0,255,0,0.3)"},
        "liabilities": {"direction": "out", "color": "rgba(255,0,0,0.3)"},
        "equity": {"direction": "out", "color": "rgba(0,255,0,0.3)"},
      },
    }
  }

  for item in sankey["assets"]["links"]:
    sankey |= sankey_graph_dict(item, depth)

  return sankey


def sankey_cashflow(depth: int) -> dict:
  sankey = {
    "cashflow": {
      "color": "",
      "links": {
        "cashflow_operating": {"direction": "in", "color": ""},
        "cashflow_investing": {"direction": "in", "color": ""},
        "cashflow_financing": {"direction": "in", "color": ""},
      },
    }
  }

  for item in sankey["cashflow"]["links"]:
    sankey |= sankey_graph_dict(item, depth)

  return sankey
