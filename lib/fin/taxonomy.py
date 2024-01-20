import ast
from collections import defaultdict
import json
import sqlite3
from typing import cast, Literal, Optional, TypedDict

from glom import glom
import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, TEXT

from lib.const import DB_DIR


class AggregateItems(ast.NodeVisitor):
  def __init__(self) -> None:
    self.items: set[str] = set()

  def reset_names(self):
    self.items = set()

  def visit_Name(self, node):
    if node.id.startswith(('average_', 'change_')):
      self.items.add(node.id)

    self.generic_visit(node)


class Template(TypedDict):
  income: dict[str, int]
  balance: dict[str, int]
  cashflow: dict[str, int]


class TaxonomyLabel(BaseModel):
  long: str
  short: Optional[str] = None


class TaxononmyCalculationItem(BaseModel):
  weight: int | float
  sign: Literal[-1, 1]


class TaxononmyCalculation(BaseModel):
  order: int = Field(ge=0)
  all: Optional[str | dict[str, TaxononmyCalculationItem]] = None
  any: Optional[str | dict[str, TaxononmyCalculationItem]] = None
  avg: Optional[str] = None
  diff: Optional[str] = None


class TaxonomyItem(BaseModel):
  unit: Literal[
    'days',
    'monetary',
    'monetary_ratio',
    'percent',
    'per_day',
    'per_share',
    'price_ratio',
    'shares',
  ]
  aggregate: Literal['average', 'recalc', 'sum', 'tail']
  gaap: list[str] = []
  label: TaxonomyLabel
  calculation: Optional[TaxononmyCalculation] = None


class Taxonomy(BaseModel):
  data: dict[str, TaxonomyItem]

  def filter_items(self, filter_: set[str]):
    new_keys = set(self.data.keys()).intersection(filter_)
    self.data = {key: value for key, value in self.data.items() if key in new_keys}

  def add_missing_items(self) -> None:
    visitor = AggregateItems()

    calc_items: set[str] = set()

    for v in self.data.values():
      if (calc := v.calculation) is None:
        continue

      calc_value = calc.all or calc.any

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
      prefix, base_item = i.split('_', 1)

      if base_item not in present_items:
        print(f'{base_item} is missing in the taxonomy!')
        continue

      short = self.data[base_item].label.short
      base_item_calc = self.data[base_item].calculation
      order = 0 if base_item_calc is None else max(0, base_item_calc.order - 1)

      if prefix == 'average':
        calc = TaxononmyCalculation(order=order, avg=base_item)
        label = TaxonomyLabel(
          long=f'Average {self.data[base_item].label.long}',
          short=None if short is None else f'Average {short}',
        )
      elif prefix == 'change':
        calc = TaxononmyCalculation(order=order, diff=base_item)
        label = TaxonomyLabel(
          long=f'Change in {self.data[base_item].label.long}',
          short=None if short is None else f'Change in {short}',
        )

      self.data[i] = TaxonomyItem(
        unit=self.data[base_item].unit,
        aggregate='average' if prefix == 'average' else 'sum',
        gaap=[],
        label=label,
        calculation=calc,
      )

  def resolve_calculation_order(self) -> None:
    order_dict = defaultdict(set)

    for k, v in self.data.items():
      if (calc := v.calculation) is None:
        continue

      calc_value = cast(str | dict, calc.all or calc.any or calc.avg or calc.diff)

      if isinstance(calc_value, str):
        order_dict[k] = extract_items(calc_value)

      elif isinstance(calc_value, dict):
        order_dict[k] = set(calc_value.keys())

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
        calc.order = result.index(k)

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
    df_data = [(k, v.label.long) for k, v in self.data.items()]
    return pd.DataFrame(df_data, columns=['item', 'long', 'short'])

  def calculation_schema(
    self, select: Optional[set[str]] = None
  ) -> dict[str, TaxononmyCalculation]:
    keys = set(self.data.keys())
    if select:
      keys = keys.intersection(select)
      if not keys:
        return {}

    schema = {
      key: calc for key in keys if (calc := self.data[key].calculation) is not None
    }
    return schema

  def extra_calculation_schema(self) -> dict[str, TaxononmyCalculation]:
    schema = {
      k: calc
      for k, v in self.data.items()
      if v.gaap is not None and (calc := v.calculation)
    }
    return schema

  def to_records(
    self,
  ):
    result = []
    for k, v in self.data.items():
      gaap = json.dumps(v.gaap) if v.gaap is not None else None
      calc = json.dumps(v.calculation) if v.calculation is not None else None

      result.append(
        (
          k,
          v.unit,
          v.period,
          v.label.long,
          v.label.short,
          gaap,
          calc,
        )
      )

    return result

  def to_sql(self, db_path: str):
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    columns = ('item', 'value', 'period', 'long', 'short', 'gaap', 'calculation')
    data = self.to_records()
    df = pd.DataFrame(data, columns=columns)

    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql(
        'items',
        con=con,
        if_exists='replace',
        index=False,
        dtype={'gaap': TEXT, 'calculation': TEXT},
      )

  def to_sqlite(self, db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    fields = {
      'item': 'TEXT PRIMARY KEY',
      'value': 'TEXT',
      'period': 'TEXT',
      'long': 'TEXT',
      'short': 'TEXT',
      'gaap': 'JSON',
      'calculation': 'JSON',
    }
    columns = ','.join(tuple(fields.keys()))
    fields_text = ','.join([' '.join((k, v)) for k, v in fields.items()])

    values = self.to_records()
    cur.execute('DROP TABLE IF EXISTS items')
    cur.execute(f'CREATE TABLE IF NOT EXISTS items ({fields_text})')
    con.commit()

    query = f"""INSERT INTO items 
      ({columns}) VALUES (?,?,?,?,?,?)
    """
    cur.executemany(query, values)
    con.commit()
    con.close()


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


def load_taxonomy(
  path: str = 'lex/fin_taxonomy.json', filter_: Optional[set[str]] = None
) -> Taxonomy:
  with open(path, 'r') as file:
    data = json.load(file)

  taxonomy = Taxonomy(data=data)
  if filter_ is not None:
    taxonomy.filter_items(filter_)

  return taxonomy


def load_template(cat: str) -> pd.DataFrame:
  with open('lex/fin_template.json', 'r') as file:
    template = json.load(file)

  template = template[cat]

  if cat == 'statement':
    data: list = [
      (sheet, item, level)
      for sheet, values in template.items()
      for item, level in values.items()
    ]
    cols: tuple = ('sheet', 'item', 'level')

  elif cat == 'fundamentals':
    data = [(sheet, item) for sheet, values in template.items() for item in values]
    cols = ('sheet', 'item')

  elif cat == 'sankey':
    data = [
      (sheet, item, entry.get('color', ''), entry.get('links'))
      for sheet, values in template.items()
      for item, entry in values.items()
    ]
    cols = ('sheet', 'item', 'color', 'links')

  elif cat == 'dupont':
    data = template
    cols = ('item',)

  return pd.DataFrame(data, columns=cols)


def template_to_sql(db_path: str):
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  for template in ('statement', 'fundamentals', 'sankey', 'dupont'):
    df = load_template(template)

    if template == 'sankey':
      mask = df['links'].notnull()
      df.loc[mask, 'links'] = df.loc[mask, 'links'].apply(lambda x: json.dumps(x))

    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql(template, con=con, if_exists='replace', index=False)


def merge_labels(template: pd.DataFrame, taxonomy: Taxonomy):
  template = template.merge(taxonomy.labels(), on='item', how='left')
  mask = template['short'] == ''
  template.loc[mask, 'short'] = template.loc[mask, 'long']
  return template


def gaap_items(sort=False) -> set[str]:
  db_path = DB_DIR / 'taxonomy.db'

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  query = """
    SELECT json_each.value AS gaap FROM items 
    JOIN json_each(gaap) ON 1=1
    WHERE gaap IS NOT NULL
  """

  cur.execute(query)
  result = cur.fetchall()
  con.close()

  items = {x[0] for x in result}

  if sort:
    items = set(sorted(items))

  return items
