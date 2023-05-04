import pandas as pd

from lib.const import DB_DIR
from lib.utils import load_json

def load_taxonomy(source: str) -> pd.DataFrame:
  df = pd.read_csv('fin_taxonomy.csv', usecols=[source, 'member', 'item', 'label'])
  df.dropna(subset=source, inplace=True)
  df.set_index(source, inplace=True)
  return df

def load_labels() -> dict:
  df = pd.read_csv('fin_labels.csv')
  return {item: label
    for item, label in zip(df['item'], df['label'])
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