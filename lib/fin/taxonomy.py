from typing import Optional, TypedDict
import json
import sqlite3

from glom import glom
import pandas as pd
from sqlalchemy import create_engine, TEXT

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

def load_template(cat: str) -> pd.DataFrame:
  with open('lex/fin_template.json', 'r') as file:
    template = json.load(file)

  template = template[cat]

  if cat == 'statement':
    data = [
      (sheet, item, level) for sheet, values in template.items() 
      for item, level in values.items()
    ]
    cols = ['sheet', 'item', 'level']

  elif cat == 'sankey':
    data = [
      (sheet, item, entry.get('color', ''), entry.get('links')) 
      for sheet, values in template.items() 
      for item, entry in values.items()
    ]
    cols = ['sheet', 'item', 'color', 'links']

  elif cat == 'dupont':
    data = template
    cols = ['item']

  return pd.DataFrame(data, columns=cols)

def template_to_sql(db_path: str):
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  dtypes = {
    'statement': {},
    'sankey': {
      'links': TEXT
    },
    'dupont': {}
  }

  for template in ('statement', 'sankey', 'dupont'):
    df = load_template(template)

    if template == 'sankey':
      mask = df['links'].notnull()
      df.loc[mask, 'links'] = df.loc[mask, 'links'].apply(lambda x: json.dumps(x))

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