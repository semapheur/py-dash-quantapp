import re


def replace_all(text: str, replacements: dict[str, str]) -> str:
  for k, v in replacements.items():
    text = text.replace(k, v)
  return text


def camel_split(text: str) -> list[str]:
  pattern = r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))"
  return re.findall(pattern, text)


def camel_case(text: str) -> str:
  text = text.replace("/", " ")
  text = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u017F]", "", text)
  words = text.split()
  camel_case_text = words[0].lower() + "".join(word.capitalize() for word in words[1:])
  return camel_case_text


def pascal_case(text: str) -> str:
  text = text.replace("/", " ")
  text = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u017F]", "", text)
  words = text.split()
  pascal_case_text = "".join(word.capitalize() for word in words)
  return pascal_case_text


def split_pascal_case(pascal_str: str) -> str:
  pascal_str = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", pascal_str)
  pascal_str = re.sub(r"([A-Z]+)([A-Z][a-z0-9])", r"\1 \2", pascal_str)
  return pascal_str


def insert_characters(string: str, inserts: dict[str, list[int]]):
  result = string
  offset = 0

  for char in inserts.keys():
    for pos in inserts[char]:
      result = result[: pos + offset] + char + result[pos + offset :]
      offset += len(char)

  return result


def remove_words(
  strings: list[str], blacklist: list[str], escape: bool = False
) -> list[str]:
  pattern = r"(?!^)\b(?:{})\b".format(
    "|".join(map(re.escape, blacklist) if escape else blacklist)
  )
  return [
    re.sub(r"\s+", " ", re.sub(pattern, "", s, flags=re.I).strip()) for s in strings
  ]
