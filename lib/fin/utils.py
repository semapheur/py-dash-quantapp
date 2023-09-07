from typing import Optional, TypedDict
import json

import pandas as pd

from lib.db.tiny import read_tinydb

class Template(TypedDict):
  income: dict[str, int]
  balance: dict[str, int]
  cashflow: dict[str, int]

class TaxonomyLabel(TypedDict):
  long: str
  short: str

class TaxononmyCalculation(TypedDict):
  order: int
  all: Optional[dict[str, int]]
  any: Optional[dict[str, int]]

class TaxonomyItem(TypedDict):
  gaap: list[str]
  label: TaxonomyLabel
  calculation: Optional[TaxononmyCalculation]

class Taxonomy:
  _data: dict[str, TaxonomyItem]

  def __init__(self, _filter: Optional[set[str]] = None):
    with open('lex/fin_taxonomy.json') as file:
      self._data = json.load(file)

    if _filter:
      new_keys = set(self._data.keys()).intersection(_filter)

      self._data = {
        key: value for key, value in self._data.items() 
        if key in new_keys
      }

  @property
  def data(self):
    return self._data

  def rename_schema(self, source: str) -> dict[str, str]:
    schema = {
      name: key for key, values in self._data.items()
      if (names := values.get(source)) for name in names
    }
    return schema
  
  def item_names(self, source: str) -> set[str]:
    names = {
      name for values in self._data.values()
      if (names := values.get(source)) for name in names
    }
    return names
  
  def labels(self) -> pd.DataFrame:
    df_data = [
      (key, value['label'].get('long', ''), value['label'].get('short', '')) 
      for key, value in self._data.items()
    ]

    return pd.DataFrame(df_data, columns=['item', 'long', 'short'])
    
  def calculation_schema(
    self, 
    select: Optional[set[str]] = None
  ) -> dict[str, TaxononmyCalculation]:

    keys = set(self._data.keys())
    if select:
      keys = keys.intersection(select)

    schema = {
      key: calc for key in keys if (calc := self._data[key].get('calculation'))
    }
   
    return schema

  def extra_calculation_schema(self, source: str) -> dict[str, TaxononmyCalculation]:
    schema =  {
      key: calc for key, value in self._data.items() 
      if not value.get(source) and (calc := value.get('calculation'))
    }
    return schema

def load_template(cat: str) -> pd.DataFrame:
  template = read_tinydb('lex/fin_template.json', tbl=cat)

  if cat == 'sheet':
    data = [
      (sheet, item, level) for sheet, values in template.items() 
      for item, level in values.items()
    ]
    cols = ['sheet', 'item', 'level']

  elif cat == 'sankey':
    data = [
      (sheet, item, entry['color'], entry.get('links',{})) 
      for sheet, values in template.items() 
      for item, entry in values.items()
    ]
    cols = ['sheet', 'item', 'color', 'links']

  return pd.DataFrame(data, columns=cols)

def merge_labels(template: pd.DataFrame, taxonomy: Taxonomy):
  template = template.merge(taxonomy.labels(), on='item', how='left')
  mask = template['short'] == ''
  template.loc[mask, 'short'] = template.loc[mask, 'long']
  return template

def calculate_items(
  financials: pd.DataFrame, 
  schemas: dict[str, TaxononmyCalculation],
  recalc: bool = False
) -> pd.DataFrame:
  
  def apply_calculation(
    df: pd.DataFrame,
    item: str,
    schema: dict[str, int]
  ) -> pd.DataFrame:

    key, value = schema.popitem()
    temp = value * df[key]

    for key, value in schema.items():
      temp += value * df[key]

    new_columns = pd.DataFrame({item: temp})
    df = pd.concat([df, new_columns], axis=1)

    return df

  col_set = set(financials.columns)

  if not recalc:
    keys = set(schemas.keys()).difference(col_set)
    schemas = {
      key: schemas[key] for key in keys
    }

  schemas = dict(sorted(schemas.items(), key=lambda x: x[1]['order']))
  

  for key, value in schemas.items():
    if isinstance(value.get('all'), dict):
      items = set(value.get('all').keys())
      if items.issubset(col_set):
        financials = apply_calculation(financials, key, value.get('all'))

    elif isinstance(value.get('any'), dict):
      schema = {
        k: v for k, v in value.get('any').items() if k in col_set
      }
      if schema:
        financials = apply_calculation(financials, key, schema)

  return financials