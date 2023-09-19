from typing import Literal, Optional, TypedDict
import json
import sqlite3

from glom import glom
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, TEXT

from lib.utils import df_month_difference

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

  def __init__(self, 
    path: str = 'lex/fin_taxonomy.json', 
    _filter: Optional[set[str]] = None
  ):
    with open(path, 'r') as file:
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
  
  def select_items(self, query: dict[str, str]) -> set[str]:
    target_key, target_value = query.popitem()

    result = {
      k for k, v in self._data.items()
      if glom(v, target_key) == target_value
    }
    return result

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
  
  def to_sql(self, db_path: str):
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    columns = ('item', 'period', 'long', 'short', 'gaap', 'calculation')
    data = []
    for k, v in self._data.items():
      if (gaap := v.get('gaap')) is not None:
        gaap = json.dumps(gaap)

      if (calc := v.get('calculation')) is not None:
        calc = json.dumps(calc)

      data.append((
        k,
        v.get('period'), 
        v['label'].get('long'),
        v['label'].get('short'),
        gaap,
        calc
      ))

    df = pd.DataFrame(data, columns=columns)

    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql('items', 
        con=con, 
        if_exists='replace', 
        index=False,
        dtype={
          'gaap': TEXT,
          'calculation': TEXT
        }
      )

  def to_sqlite(self, db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    fields = {
      'name': 'TEXT PRIMARY KEY',
      'period': 'TEXT',
      'long': 'TEXT',
      'short': 'TEXT',
      'gaap': 'JSON',
      'calculation': 'JSON',
    }
    columns = ','.join(tuple(fields.keys()))
    fields_text = ','.join([' '.join((k, v)) for k, v in fields.items()])

    values: list[tuple[str|None]] = []
    for k, v in self._data.items():
      if (gaap := v.get('gaap')) is not None:
        gaap = json.dumps(gaap)

      if (calc := v.get('calculation')) is not None:
        calc = json.dumps(calc)

      values.append((
        k,
        v.get('period'), 
        v['label'].get('long'),
        v['label'].get('short'),
        gaap,
        calc
      ))

    #with engine.connect() as con:
    cur.execute('DROP TABLE IF EXISTS items')
    cur.execute(f'CREATE TABLE IF NOT EXISTS items ({fields_text})')
    con.commit()

    query = f'''INSERT INTO items 
      ({columns}) VALUES (?,?,?,?,?,?)
    '''
    cur.executemany(query,values)
    con.commit()
    con.close()

def load_template(cat: Literal['table', 'sankey']) -> pd.DataFrame:
  with open('lex/fin_template.json', 'r') as file:
    template = json.load(file)

  template = template[cat]

  if cat == 'table':
    data = [
      (sheet, item, level) for sheet, values in template.items() 
      for item, level in values.items()
    ]
    cols = ['sheet', 'item', 'level']

  elif cat == 'sankey':
    data = [
      (sheet, item, entry.get('color', ''), entry.get('links',{})) 
      for sheet, values in template.items() 
      for item, entry in values.items()
    ]
    cols = ['sheet', 'item', 'color', 'links']

  return pd.DataFrame(data, columns=cols)

def template_to_sql(db_path: str):
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  dtypes = {
    'table': {},
    'sankey': {
      'links': TEXT
    }
  }

  for template in ('table', 'sankey'):
    df = load_template(template)

    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql(template, 
        con=con, 
        if_exists='replace', 
        index=False, 
        dtype=dtypes[template]
      )
  
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
  
  def applyer(s: pd.Series, fn: str) -> pd.Series:
    
    slices = (
      (slice(None), slice('FY'), slice(12)),
      (slice(None), slice(None), slice(3))
    )
    update = [pd.Series()] * len(slices)
    
    for i, ix in enumerate(slices):
      _s = s.loc[ix]
      _s.sort_index(level='date', inplace=True)

      dates = pd.to_datetime(_s.index.get_level_values('date'))
      month_diff = pd.Series(df_month_difference(dates).array, index=_s.index)

      if fn == 'diff':
        _s = _s.diff()
      elif fn == 'avg':
        _s = _s.rolling(window=2, min_periods=2).mean()
      
      _s = _s.loc[month_diff == ix[2]]
      update[i] = _s

    update = pd.concat(update, axis=0)
    nan_index = pd.Index(list(set(s.index).difference(update.index)))

    s.loc[update.index] = update
    s.loc[nan_index] = np.nan

    return s
    
  def apply_calculation(
    df: pd.DataFrame,
    df_cols: set[str],
    calculee: str,
    schema: dict[str, int]
  ) -> pd.DataFrame:
    
    def parameter(calculer: str, instruction: int|dict) -> pd.Series:
      result = df[calculer]

      if isinstance(instruction, int):
        result *= instruction
      elif isinstance(instruction, dict):
        if 'sign' in instruction:
          result = result.where(result.apply(np.sign) == instruction['sign'], 0)
        
        if (op := instruction.get('apply')) is not None:
          result = applyer(result, op)
          
        result *= instruction.get('weight', 1) 

      return result.fillna(0)

    key, value = schema.popitem()
    temp = parameter(key, value)

    for key, value in schema.items():
      op = value.get('operation')
      match op:
        case 'div':
          temp /= parameter(key, value)
        case _:
          temp += parameter(key, value)

    if calculee in df_cols:
      df.loc[:,calculee] = temp if recalc else df[calculee].combine_first(temp)

    else:
      new_column = pd.DataFrame({calculee: temp})
      df = pd.concat([df, new_column], axis=1)

    return df

  schemas = dict(sorted(schemas.items(), key=lambda x: x[1]['order']))

  for key, value in schemas.items():
    col_set = set(financials.columns)

    if isinstance(value.get('all'), dict):
      items = set(value.get('all').keys())
      if items.issubset(col_set):
        financials = apply_calculation(financials, col_set, key, value.get('all'))

    elif isinstance(value.get('any'), dict):
      schema = {
        k: v for k, v in value.get('any').items() if k in col_set
      }
      if schema:
        financials = apply_calculation(financials, col_set, key, schema)

  return financials