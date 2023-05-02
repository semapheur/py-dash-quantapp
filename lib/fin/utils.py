import json
from pathlib import Path

import pandas as pd

from lib.db import DB_DIR

def template_items():
  json_path = DB_DIR / 'fin_template.json'
  if not json_path.exists():
    raise Exception(f'Template file ("fin_template.json") does not exist in folder {DB_DIR}')

  with open(json_path, 'r') as f:
    template = json.load(f)

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