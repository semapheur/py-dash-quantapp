import pandas as pd

def make_sparkline(
  df_wide: pd.DataFrame, 
  num_format: str = '2G'
) -> pd.Series:
  
  # Normalize
  max = df_wide.max(axis=1)
  min = df_wide.min(axis=1)

  df_spark = (df_wide.sub(min, axis='index')
    .div((max - min), axis='index')
    .mul(100)
    .fillna(0)
  )
  # Format numbers
  df_spark['spark'] = df_spark.astype(int).astype(str).agg(','.join, axis=1)

  # Endpoint numbers
  df_spark['start'] = (df_wide.bfill(axis=1)
    .fillna(0)
    .loc[:,df_wide.columns[0]]
    .round(0)
  )
  df_spark['end'] = (df_wide.ffill(axis=1)
    .fillna(0)
    .loc[:,df_wide.columns[-1]]
    .round(0)
  )

  return (
    df_spark['start'].apply(lambda x: f'{x:.{num_format}}') + 
    '{' + df_spark['spark'] + '}' + 
    df_spark['end'].apply(lambda x: f'{x:.{num_format}}')
  )