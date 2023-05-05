from typing import Literal, Optional

import pandas as pd

from lib.const import DB_DIR
from lib.utils import load_json

def load_taxonomy(source: str) -> pd.DataFrame:
  df = pd.read_csv('fin_taxonomy.csv', usecols=[source, 'member', 'item', 'label'])
  df.dropna(subset=source, inplace=True)
  df.set_index(source, inplace=True)
  return df

def load_items(
  cols: Optional[str|list[str]] = None, 
  _filter: Optional[list[str]] = None,
  fill_empty: Optional[bool] = False
) -> pd.DataFrame|pd.Series:

  series = False
  if isinstance(cols, str):
    cols = [cols]
    series = True

  df = pd.read_csv('fin_labels.csv', usecols=cols)

  if {'short', 'long'}.issubset(df.columns) and fill_empty:
    df['short'].fillna(df['long'], inplace=True)

  if _filter is not None:
    df = df.loc[df['item'].isin(_filter)]

  if series:
    return df[cols].squeeze()

  return df

def load_labels(_type: Literal['long', 'short']) -> dict[str, str]:
  df = load_items(['item', _type])
  
  return {item: label
    for item, label in zip(df['item'], df[_type])
  }

def items_to_csv():
  path = DB_DIR / 'fin_template.json'
  template = load_json(path)

  items: list[dict[str, str]] = []
  for sheet in template:
    for item, props in template[sheet].items():
      items.append({
        'sheet': sheet,
        'item': item,
        'label': props['label']
      })
      if 'members' in props:
        for member, _props in props['members'].items():
          items.append({
            'sheet': sheet,
            'item': member,
            'label': _props['label']
          })

  csv_path = DB_DIR / 'fin_items.csv'
  df = pd.DataFrame.from_records(items)
  df.to_csv(csv_path, index=False)