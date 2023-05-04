import json
import re
from pathlib import Path

def load_json(file_path: str|Path) -> dict:
  with open(file_path, 'r') as f:
    return json.load(f)

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