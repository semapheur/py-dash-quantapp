import re

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