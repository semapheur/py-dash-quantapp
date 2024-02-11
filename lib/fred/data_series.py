from dataclasses import dataclass
from datetime import datetime as dt
from typing import Optional

import httpx
import io
import pandas as pd

from lib.const import HEADERS


@dataclass(slots=True)
class DataSeries:
  ticker: str

  def time_series(
    self, start_date: dt, series_name: Optional[str] = None
  ) -> pd.DataFrame | None:
    end_date = dt.now().strftime('%Y-%m-%d')
    params = {
      'id': self.ticker,
      'cosd': start_date.strftime('%Y-%m-%d'),
      'coed': end_date,
    }
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

    # Retrieve csv
    with httpx.Client() as client:
      rs = client.post(url, headers=HEADERS, params=params)
      csv = rs.text

    buffer = io.StringIO(csv)  # Parse csv

    try:
      df = pd.read_csv(
        buffer,
        delimiter=',',
        index_col=0,
        parse_dates=True,
        date_format='%Y-%m-%d',
      )
      buffer.close()
      df.index.rename('date', inplace=True)
      df[self.ticker] = pd.to_numeric(df[self.ticker], errors='coerce')

      # Rename
      if series_name is not None:
        df.rename(columns={self.ticker: series_name}, inplace=True)

    except ValueError:
      print(f'Could not fetch time series for {self.ticker}')

    finally:
      return df
