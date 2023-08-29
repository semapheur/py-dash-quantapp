from typing import Literal, Optional, TypedDict
import json

import pandas as pd

from lib.const import DB_DIR
from lib.utils import load_json

class Template(TypedDict):
  income: dict[str, int]
  balance: dict[str, int]
  cashflow: dict[str, int]

class TaxonomyLabel(TypedDict):
  long: str
  short: str

class TaxononmyCalculation(TypedDict):
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
      for key, value in self._data
    ]

    return pd.DataFrame(df_data, columns=['item', 'long', 'short'])
  
  def all_calculation_schema(self) -> dict[str, TaxononmyCalculation]:
    schema =  {
      key: calc for key, value in self._data.items() 
      if (calc := value.get('calculation'))
    }
    return schema
  
  def extra_calculation_schema(self, source: str) -> dict[str, TaxononmyCalculation]:
    schema =  {
      key: calc for key, value in self._data.items() 
      if (calc := value.get('calculation')) and not value.get(source)
    }
    return schema

def load_template() -> pd.DataFrame:
  with open('lex/fin_template.json') as file:
    template: Template = json.load(file)

  items = [
    (sheet, item, level) for sheet, values in template.items() 
    for item, level in values.items()
  ]

  return pd.DataFrame(items, columns=['sheet', 'item', 'level'])

def calculate_items(
  financials: pd.DataFrame, 
  schemas: dict[str, TaxononmyCalculation]
) -> pd.DataFrame:
  
  def apply_calculation(
    item: str,
    schema: dict[str, int]
  ) -> pd.DataFrame:

    key, value = schema.popitem()
    financials[item] = value * financials[key]

    for key, value in schema.items():
      financials[item] += value * financials[key]

    return financials

  col_set = set(financials.columns)

  for key, value in schemas.items():
    if isinstance(value.get('all'), dict):
      items = set(value.get('all').keys())
      if items.issubset(col_set):
        financials = apply_calculation(key, items, value.get('all'))

    elif isinstance(value.get('any'), dict):
      schema = {
        k: v for k, v in value.get('any').items() if k in col_set
      }
      if schema:
        financials = apply_calculation(key, items, schema)

  return financials