import pandas as pd

from datetime import datetime as dt

import requests
import io

class Ticker():

    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-GPC': '1',
        'TE': 'Trailers',
    }

    def __init__(self, ticker):
        self._ticker = ticker

    def timeSeries(self, startDate, seriesName=None):
    
        # URL parameters
        endDate = dt.now().strftime('%Y-%m-%d')
        params = (
            ('id', self._ticker),
            ('cosd', startDate),
            ('coed', endDate),
        )
        
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
        
        # Retrieve csv
        with requests.Session() as s:
            rc = s.post(url, headers=self._headers, params=params)
            csv = rc.text
        
        buff = io.StringIO(csv) # Parse csv
        
        dateparse = lambda x: pd.to_datetime(x, infer_datetime_format=True)  # dt.strptime(x, '%Y-%m-%d') 
        
        try:
            df = pd.read_csv(buff, delimiter=',', index_col=0, parse_dates=True, 
                date_parser=dateparse)
            buff.close()
            
            df.index.rename('date', inplace=True)
            
            # Convert to numeric valie
            df[self._ticker] = pd.to_numeric(df[self._ticker], errors='coerce') #df[ticker].astype('float')
            
            # Rename
            if seriesName is not None:
                df.rename(columns={self._ticker: seriesName}, inplace=True)
            
            return df
        
        except ValueError:
            return None

def batchSeries(tickers, seriesNames, startDate):

    if isinstance(tickers, str) and isinstance(seriesNames, str):
        ticker = Ticker(tickers)
        return ticker.timeSeries(startDate)

    elif isinstance(tickers, list):
        dfs = []

        for t, s in zip(tickers, seriesNames):
            ticker = Ticker(t)
            dfs.append(ticker.timeSeries(startDate, s))
        
        return pd.concat(dfs, axis=1)

    else:
        return None