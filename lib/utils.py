def replace_all(text, dic):
  for i, j in dic.items():
    text = text.replace(i, j)
  return text