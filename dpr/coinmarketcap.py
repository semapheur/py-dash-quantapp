import pandas as pd

import requests
import json

# Date
from datetime import datetime as dt
from datetime import timezone as tz

class Ticker():

    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'TE': 'Trailers',
    }

    def __init__(self, link):
        self._link = link
        
    def ohlcv(self, startDate):

        if isinstance(startDate, dt):
                startStamp = int(startDate.replace(tzinfo=tz.utc).timestamp())

        elif isinstance(startDate, str):
            startDate = dt.strptime(startDate, '%Y-%m-%d')
            startStamp = int(startDate.replace(tzinfo=tz.utc).timestamp())

        endStamp = int(dt.now().replace(tzinfo=tz.utc).timestamp()) * 1000
        endStamp += 3600 * 24
    
        params = (
            ('convert', 'USD'),
            ('slug', self._link),
            ('time_end', str(endStamp)),
            ('time_start', str(startStamp)),
        )
        url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical'
        
        with requests.Session() as s:
            rc = s.get(url, headers=self._headers, params=params)
            content = json.loads(rc.text)
        
        scrap = []
        for q in content['data']['quotes']:
            
            d = q['quote']['USD']
            
            scrap.append({
                'date': dt.strptime(d.get('timestamp'), '%Y-%m-%dT%H:%M:%S.%fZ'),
                'open': d.get('open'),
                'high': d.get('high'),
                'low': d.get('low'),
                'close': d.get('close'),
                'volume': d.get('volume')
            })
        
        df = pd.DataFrame.from_records(scrap)
        
        return df

# Coinmarketcap tickers
def getTickers():
    
    params = (
        ('convert', 'USD'),
        ('cryptocurrency_type', 'all'),
        ('limit', '5000'),
        ('sort', 'market_cap'),
        ('sort_dir', 'desc'),
        ('start', '1'),
    )
    
    url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    
    with requests.Session() as s:
        rc = s.get(url, headers=Ticker._headers, params=params)
        content = json.loads(rc.text)
            
    scrap = []
    for d in content['data']:
        scrap.append({
            'type': 'crypto',
            'cmcId': d.get('id'),
            'name': d.get('name'),
            'ticker': d.get('symbol'),
            'link': d.get('slug')
        })
    
    df = pd.DataFrame.from_records(scrap)
    return df

