from datetime import datetime as dt
from typing import Optional

import httpx
import io
import pandas as pd

from lib.const import HEADERS

class Ticker():

  def __init__(self, ticker: str):
    self._ticker = ticker

  def time_series(self, 
    start_date: str, 
    series_name: Optional[str] = None
  ) -> pd.DataFrame:
    # URL parameters
    end_date = dt.now().strftime('%Y-%m-%d')
    params = {
      'id': self._ticker,
      'cosd': start_date,
      'coed': end_date,
    }
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
    
    # Retrieve csv
    with httpx.Client() as client:
      rs = client.post(url, headers=HEADERS, params=params)
      csv = rs.text
    
    buff = io.StringIO(csv) # Parse csv
    
    try:
      df = pd.read_csv(buff, delimiter=',', index_col=0, parse_dates=True, 
        date_parser=lambda x: pd.to_datetime(x, infer_datetime_format=True)
      )
      buff.close()
      df.index.rename('date', inplace=True)
        
      # Convert to numeric valie
      df[self._ticker] = pd.to_numeric(df[self._ticker], errors='coerce')
        
      # Rename
      if series_name is not None:
          df.rename(columns={self._ticker: series_name}, inplace=True)
        
      return df
    
    except ValueError:
      return None