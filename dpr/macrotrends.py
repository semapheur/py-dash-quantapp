# Data analysis
import numpy as np
import pandas as pd

# Date
from datetime import datetime as dt

# API
#import yfinance as yf

# Web scrapping
import requests
import bs4 as bs
import json

# Utils
import re
#from tqdm import tqdm

# Local
#from foos import renamer
from lib.finlib import finItemRenameDict

class Ticker:

    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Sec-GPC': '1',
        'TE': 'Trailers',
    }

    def __init__(self, ticker, link):
        self._ticker = ticker
        self._link = link

    def financials(self):
        finItems = finItemRenameDict('Macrotrends')

        sheets = ['income-statement', 'balance-sheet', 'cash-flow-statement']
        
        pattern = r'(?<=var originalData = )\[.+\}\]'
        #pattern = r'originalData = \[.+\]'
        
        dfSheets = [None] * len(sheets)
        for i, sh in enumerate(sheets):

            url = (f'https://www.macrotrends.net/stocks/charts/{self._link}/'
                f'{sh}?freq=Q')

            with requests.Session() as s:
                rc = s.post(url)
                html = rc.text
                content = re.search(pattern, html, re.DOTALL)
                parse = json.loads(content.group())
                
            cols = [None] * len(parse)
            figures = [None] * len(parse)
            for j, pst in enumerate(parse):
                soup = bs.BeautifulSoup(pst['field_name'], 'lxml')
                
                cols[j] = soup.text
                
                dKeys = list(pst.keys())[2:]
                dates = [None] * len(dKeys)
                entry = [None] * len(dKeys)
                for k, d in enumerate(dKeys):
                    dates[k] = dt.strptime(d, '%Y-%m-%d')
                    
                    if pst[d]:
                        entry[k] = (float(pst[d])*1e6)
                        
                    else:
                        entry[k] = np.nan
                        
                figures[j] = entry
                
            temp = pd.DataFrame(np.array(figures).T, index=dates, columns=cols)
            temp.sort_index(inplace=True)
            
            if sh == 'income-statement':
                
                cols = temp.columns.difference(['Basic Shares Outstanding', 
                                                'Shares Outstanding']) 
                
                temp[cols] = temp[cols].rolling(4, min_periods=4).sum()
                
                #for c in cols:
                    #temp[c + '_'] = temp[c].rolling(4, min_periods=4).sum()
                    
            elif sh == 'cash-flow-statement':
                
                temp.drop('Net Income/Loss', axis=1, inplace=True)
                temp = temp.rolling(4, min_periods=4).sum()
                
                #for c in temp.columns:
                    #temp[c + '_'] = temp[c].rolling(4, min_periods=4).sum()

            dfSheets[i] = temp
            
        data = pd.concat(dfSheets, axis=1)
        data.index.rename('date', inplace=True)
        data = data.loc[:,~data.columns.duplicated()]
        
        #ttmPosts = {f'{k}_': v[:-1] for k, v in rnm.items()}
        
        #for d in [rnm, ttmPosts]:
        #    data.rename(columns=d, inplace=True)
        
        data.rename(columns=finItems, inplace=True)
            
        data[['eps', 'epsDil']] /= 1e6
        
        data['opEx'] = data['opEx'] - data['rvnEx'] 
        
        # Short-term investments/debt, account payables
        data['stInv'] = data['chgStInv'].cumsum().clip(0)
        data['stDbt'] = data['chgStDbt'].cumsum().clip(0)
        data['accPyb'] = data['chgPyb'].cumsum().clip(0)
        
        data['totDbt'] = data['stDbt'].fillna(0) + data['ltDbt']
        
        data['intEx'] = data['ebit'] - data['ebt']
        
        data['intCvg'] = (data['ebit'] / data['intEx'])
        
        data['taxRate'] = data['taxEx'] / data['ebt'] 
        
        # Capital expenditures
        data['capEx'] = data['ppe'].diff() + data['da']
        
        # Working capital
        data['wrkCap'] = (
            data['totCrtAst'].rolling(2, min_periods=0).mean() -
            data['totCrtLbt'].rolling(2, min_periods=0).mean()
        )
        
        # Preferred equity
        data['prfEqt'] = (
            data['totEqt'] - data['cmnEqt'] - data['othEq'] - data['rtnErn'])
        
        # Tangible equity
        data['tgbEqt'] = (data['totEqt'] - data['prfEqt'] - 
                            data['gwItgbAst'])
        if data['tgbEqt'].isnull().all():
            data['tgbEqt'] = data['totEqt']
        
        # Free cash flow
        if data['chgWrkCap'].isnull().all():
            data['chgWrkCap'] = data['wrkCap'].diff()
        
        data['freeCfFirm'] = (
            data['netInc'] + 
            data['da'] +
            data['intEx'] * (1 - data['taxRate']) - 
            data['chgWrkCap'] -
            data['capEx']
        )
        
        data['freeCf'] = (
            data['freeCfFirm'] + 
            data['totDbt'].diff() - 
            data['intEx'] * (1 - data['taxRate']))
        
        if data['freeCf'].isna().all():
            data['freeCf'] = data['freeCf'].fillna(0)
        
        # Drop nan columns
        #cols = ['rev', 'revEx', 'da', 'div', 'rcv', 'invnt', 'stInv', 'ltInv', 
        #        'accPayab', 'chgStInv', 'chgLtInv', 'chgStDbt', 'capEx', 'freeCf']
        
        #for c in cols: 
        #    data[c] = data[c].fillna(0)
        
        #data.dropna(axis=1, how='all', inplace=True)
             
        return data

    def getTickers():
        
        params = (
            ('_', '1604752192860'),
        )
        
        url = 'https://www.macrotrends.net/assets/php/ticker_search_list.php'
        
        with requests.Session() as s:
            rc = s.post(url, headers=Ticker._headers, params=params)
            parse = json.loads(rc.text)
        
        df = pd.DataFrame.from_records(parse)
        
        df.sort_values(by='n', inplace=True)
        df.rename(columns={'n': 'query', 's': 'macrotrendsLink'}, inplace=True)
        df['ticker'] = df['macrotrendsLink'].str.split('/', n=1)  

        # Add security type
        df['type'] = 'stock'
        mask = df['name'].str.contains(r'\b(et(c|f|n|p)(s)?)\b', regex=True, flags=re.I)
        df.loc[mask, 'type'] = 'etf'
        
        return df