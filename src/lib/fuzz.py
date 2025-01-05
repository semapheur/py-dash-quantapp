import pandas as pd
from rapidfuzz import fuzz, process


def fuzzy_merge(
  df1: pd.DataFrame, df2: pd.DataFrame, on: str, threshold: float = 80, limit: int = 1
):
  s = df2[on].tolist()
  matched_rows = []

  for _, row in df1.iterrows():
    name = row[on]
    best_match = process.extractOne(name, s, scorer=fuzz.partial_ratio)
    if best_match and best_match[1] >= threshold:
      match_index = s.index(best_match[0])
      matched_row_df2 = df2.iloc[match_index].rename({on: f"{on}_match"})

      matched_row = pd.concat([row, matched_row_df2])
      matched_rows.append(matched_row)

  return pd.DataFrame(matched_rows).reset_index(drop=True)
