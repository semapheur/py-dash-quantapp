from itertools import combinations
from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from tqdm import tqdm

from lib.utils import remove_words

stock_trim_words = [  # (?!^)
  r'\s[.-:]+\s',
  r'\d(\.\d+)?\s?%',
  r'\d/\d+(th)?',
  r'-([a-z]|\d+?)-',
  r'd/d+(th)?',
  'ab',
  r'a\.?dr?',
  'ag',
  'alien market',
  r'a/?sa?',
  'bearer',
  'bhd',
  'brdr',
  r'\(?buyback\)?',
  'cad',
  r'c?dr',
  'cedear',
  r'(one-(half)? )?cl(as)?s -?[a-z]-?',
  r'dep(osits?)?',
  r'(((brazili|canadi|kore)an|taiwan) )?deposit(a|o)ry (interests?|receipts?)',
  r'exch(angeable)?',
  'fixed',
  'fltg',
  'foreign',
  'fxdfr ',
  'gbp',
  'gmbh',
  r'\(?[a-z]{3} hedged\)?',
  'inc',
  r'int(terests?)?',
  'into',
  'jsc',
  r'kc?sc',
  'kgaa',
  'lp',
  'maturity',
  'na',
  r'\(new\)',
  r'(non)?-?conv(ert((a|i)ble)?)?',
  r'(non)?-?cum',
  r'(limited|non|sub(ord)?)?-?vo?t(in)?g',
  r'nv(dr)?',
  r'ord(inary)?',
  'partly paid',
  'pcl',
  r'perp(etual)?( [a-z]{3})?',
  'pfd',
  'php',
  'plc',
  'pref',
  'prf',
  'psc',
  'red',
  r'registere?d',
  r'repr(\.|esents)?',
  'restricted',
  r'r(ig)?ht?s?',
  r'\(?rs\.\d{1,2}(\.\d{2})?\)?',
  'rt',
  r's\.?a\.?',
  'sae',
  'sak',
  'saog',
  r'ser(ies?)? -?[a-z0-9]-?',
  r'sh(are)?s?',
  'spa',
  'sr',
  'sub',
  'tao',
  'tbk',
  r'(unitary )?(144a/)?reg s',
  r'units?',
  r'undated( [a-z]{3})',
  r'(un)?sponsored',
  r'(\d )?vote',
  r'(one(-half)? )?war(rant)?s?',
  'without',
]


def levenshtein_distance(token1: str, token2: str) -> int:
  distances = np.zeros((len(token1) + 1, len(token2) + 1))

  for t1 in range(len(token1) + 1):
    distances[t1][0] = t1

  for t2 in range(len(token2) + 1):
    distances[0][t2] = t2

  a = 0
  b = 0
  c = 0

  for t1 in range(1, len(token1) + 1):
    for t2 in range(1, len(token2) + 1):
      if token1[t1 - 1] == token2[t2 - 1]:
        distances[t1][t2] = distances[t1 - 1][t2 - 1]
      else:
        a = distances[t1][t2 - 1]
        b = distances[t1 - 1][t2]
        c = distances[t1 - 1][t2 - 1]

        if a <= b and a <= c:
          distances[t1][t2] = a + 1
        elif b <= a and b <= c:
          distances[t1][t2] = b + 1
        else:
          distances[t1][t2] = c + 1

  return distances[len(token1)][len(token2)]


def token_sort(s):
  return ' '.join(sorted(s.lower().split()))


def token_sort_ratio(s1, s2):
  sorted_s1 = token_sort(s1)
  sorted_s2 = token_sort(s2)

  max_len = max(len(sorted_s1), len(sorted_s2))
  if max_len == 0:
    return 100.0

  distance = levenshtein_distance(sorted_s1, sorted_s2)
  ratio = ((max_len - distance) / max_len) * 100
  return ratio


def fuzzy_threshold(str1: str, str2: str, base_threshold: float) -> float:
  # Get the lengths of both strings
  len1, len2 = len(str1), len(str2)

  # Calculate the average length
  avg_length = (len1 + len2) / 2

  # Define adjustment factors
  min_length = 5.0  # Minimum length for adjustment
  max_length = 20.0  # Length at which adjustment reaches its maximum
  max_adjustment = 10.0  # Maximum adjustment to the threshold

  if avg_length <= min_length:
    adjustment = 0.0
  elif avg_length >= max_length:
    adjustment = max_adjustment
  else:
    # Linear interpolation between min_length and max_length
    adjustment = (avg_length - min_length) / (max_length - min_length) * max_adjustment

  # Apply the adjustment to the base threshold
  adjusted_threshold = base_threshold + adjustment

  # Ensure the adjusted threshold is between 0 and 100
  return max(0.0, min(100.0, adjusted_threshold))


def group_fuzzy_matches(
  strings: list[str],
  threshold=90,
  scorer=fuzz.token_set_ratio,
  trim_words: Optional[list[str]] = None,
) -> list[list[str]]:
  n = len(strings)

  trimmed = strings
  if trim_words:
    trimmed = remove_words(strings, trim_words)

  # Initialize each string as its own group
  groups = list(range(n))

  def find(x: int) -> int:
    if groups[x] != x:
      groups[x] = find(groups[x])
    return groups[x]

  def union(x: int, y: int):
    groups[find(x)] = find(y)

  # Create blocks based on the first letter
  blocks: dict[str, list[int]] = {}
  for i, s in enumerate(trimmed):
    first_letter = s[0].lower()
    if first_letter not in blocks:
      blocks[first_letter] = []
    blocks[first_letter].append(i)

  # Compare strings within each block
  for block in tqdm(blocks.values()):
    for i in range(len(block)):
      for j in range(i + 1, len(block)):
        if find(block[i]) != find(block[j]):
          similarity = scorer(trimmed[block[i]], trimmed[block[j]])
          adjusted_threshold = fuzzy_threshold(
            trimmed[block[i]], trimmed[block[j]], threshold
          )
          if similarity >= adjusted_threshold:
            union(block[i], block[j])

  # Create final groups
  final_groups: dict[int, list[str]] = {}
  for i, string in enumerate(strings):
    group = find(i)
    if group not in final_groups:
      final_groups[group] = []
    final_groups[group].append(string)

  final_groups = {key: sorted(value) for key, value in final_groups.items()}

  return list(final_groups.values())


def pairwise_fuzzy_match(nested_list: list[str]) -> dict[float, list[tuple[str, str]]]:
  result: dict[float, list[tuple[str, str]]] = {}

  for sublist in nested_list:
    # Generate all unique pairs in the sublist
    pairs = combinations(sublist, 2)

    for str1, str2 in pairs:
      # Calculate similarity score
      score = fuzz.ratio(str1, str2)

      # If this score is not in the result dict, add it
      if score not in result:
        result[score] = []

      # Add the pair to the list for this score
      result[score].append((str1, str2))

  return dict(sorted(result.items()))
