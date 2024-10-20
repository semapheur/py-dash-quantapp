from dataclasses import dataclass
from datetime import datetime as dt
from typing import cast

import httpx
import io
import pandas as pd

from lib.const import HEADERS


@dataclass(slots=True)
class DataSeries:
  sid: str

  def time_series_info(self) -> dt:
    params = {
      "id": self.sid,
    }

    url = "https://fred.stlouisfed.org/graph/api/series/"
    with httpx.Client() as client:
      response = client.get(url, headers=HEADERS, params=params)
      data = response.json()

    min_date = cast(str, data["chart"]["min_date"])

    return dt.strptime(min_date, "%Y-%m-%d")

  async def time_series(
    self, start_date: dt | None = None, series_name: str | None = None
  ) -> pd.DataFrame | None:
    end_date = dt.today().strftime("%Y-%m-%d")

    if start_date is None:
      start_date = self.time_series_info()

    params = {
      "id": self.sid,
      "cosd": start_date.strftime("%Y-%m-%d"),
      "coed": end_date,
    }
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # Retrieve csv
    async with httpx.AsyncClient() as client:
      response = await client.post(url, headers=HEADERS, params=params)
      csv = response.text

    buffer = io.StringIO(csv)  # Parse csv

    try:
      df = pd.read_csv(
        buffer,
        delimiter=",",
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
      )
      buffer.close()
      df.index.rename("date", inplace=True)
      df[self.sid] = pd.to_numeric(df[self.sid], errors="coerce")

      # Rename
      if series_name is not None:
        df.rename(columns={self.sid: series_name}, inplace=True)

    except ValueError:
      print(f"Could not fetch time series for {self.sid}")

    finally:
      return df
