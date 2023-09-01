import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

def load_json(path: str|Path) -> dict:
  with open(path, 'r') as f:
    return json.load(f)

def update_json(path: str|Path, data: dict):
  if isinstance(path, str):
    path = Path(str)

  if path.suffix != '.json':
    path = path.with_suffix('.json')
  
  try:
    with open(path, 'r') as f:
      file_data = json.load(f)

  except (FileNotFoundError, json.JSONDecodeError):
    file_data = {}

  file_data.update(data)
  
  with open(path, 'w') as f:
    json.dump(file_data, f)

def minify_json(path: str|Path, new_name: Optional[str] = None):
  if isinstance(path, str):
    path = Path(path)

  with open(path, 'r') as f:
    data = json.load(f)

  if not new_name:
    new_path = path.with_name(f'{path.stem}_mini.json')
  else:
    new_path = path.with_name(new_name).with_suffix('.json')
  
  with open(new_path, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

def replace_all(text, dic):
  for i, j in dic.items():
    text = text.replace(i, j)
  return text

def camel_split(txt: str) -> list[str]:
  pattern = r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))'
  return re.findall(pattern, txt)

def camel_abbreviate(txt: str, chars: int=2):
  words = camel_split(txt)
  words[0] = words[0].lower()
  words = [word[:chars] for word in words]
  return ''.join(words)

def snake_abbreviate(txt: str, chars: int=2):
  words = camel_split(txt)
  words = [word[:chars].lower() for word in words]
  return '_'.join(words)

# Rename DataFrame columns
class renamer():
  def __init__(self):
    self.d = dict()

  def __call__(self, x: str):
    if x not in self.d:
      self.d[x] = 0
      return x
    else:
      self.d[x] += 1
      return '%s_%d' % (x, self.d[x])
    
def insert_characters(string: str, inserts: dict[str, list[int]]):
  result = string
  offset = 0

  for char in inserts.keys():
    for pos in inserts[char]:
      result = result[:pos + offset] + char + result[pos + offset:]
      offset += len(char)

  return result

def combine_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    
  duplicated = df.columns.duplicated()

  if not duplicated.any():
    return df

  df_duplicated = combine_duplicate_columns(df.loc[:, duplicated])
  df = df.loc[:, ~duplicated]

  for col in df_duplicated.columns:
    df.loc[:, col] = df[col].combine_first(df_duplicated[col])

  return df