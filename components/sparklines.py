from typing import cast

from pandera.typing import DataFrame, Series


def make_sparkline(df_wide: DataFrame, num_format: str = '2G') -> Series[str]:
  # Normalize
  max = df_wide.max(axis=1)
  min = df_wide.min(axis=1)

  df_spark = (
    df_wide.sub(min, axis='index').div((max - min), axis='index').mul(100).fillna(0)
  )
  # Format numbers
  df_spark['spark'] = df_spark.astype(int).astype(str).agg(','.join, axis=1)

  # Endpoint numbers
  df_spark['start'] = (
    df_wide.bfill(axis=1).fillna(0).loc[:, df_wide.columns[0]].round(0)
  )
  df_spark['end'] = df_wide.ffill(axis=1).fillna(0).loc[:, df_wide.columns[-1]].round(0)

  return (
    cast(Series[str], df_spark['start'].apply(lambda x: f'{x:.{num_format}}'))
    + '{'
    + cast(Series[str], df_spark['spark'])
    + '}'
    + cast(Series[str], df_spark['end'].apply(lambda x: f'{x:.{num_format}}'))
  )
