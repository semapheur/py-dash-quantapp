import numpy as np
import pandas as pd

# Databasing
import sqlalchemy as sqla

# Web scrapping
import requests
import json
import bs4 as bs

# Date
from datetime import datetime as dt

# Utils
import re
from pathlib import Path

class Ticker():

    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest',
        'DNT': '1',
        'Connection': 'keep-alive',
    }

    def __init__(self, name, link, pairId=''):
        self._pairId = pairId
        self._name = name
        self._link = link

    def ohlcv(self, startDate='01/01/2000'):
    
        if isinstance(startDate, dt):
            startDate = startDate.strftime('%m/%d/%Y') # Convert datetime to string

        if '/' not in startDate:
            startDate = dt.strptime(startDate, '%Y-%m-%d').strftime('%m/%d/%Y')

        endDate = dt.now().strftime('%m/%d/%Y')
        
        # Get post args
        if not self._pairId:
            url = f'https://www.investing.com/{self._link}-historical-data'
            
            with requests.Session() as s:
                rq = s.get(url, stream=True, headers=self._headers)
                soup = bs.BeautifulSoup(rq.text, 'lxml')
                
            pattern = re.compile('pairId: \d+')
            script = soup.find('script', text=pattern)
            
            if script:
                match = pattern.search(script.text)
                if match:
                    self._pairId = match.group().split(' ', 2)[1]
        
        # Scrap OHLCV
        data = [
            ('curr_id', self._pairId),
            #('smlID', smlId),
            ('header', self._name + ' Historical Data'),
            ('st_date', startDate),
            ('end_date', endDate),
            ('interval_sec', 'Daily'),
            ('sort_col', 'date'),
            ('sort_ord', 'DESC'),
            ('action', 'historical_data'),
        ]
        
        url = 'https://www.investing.com/instruments/HistoricalDataAjax'
        
        # Get historical data
        with requests.Session() as s:
            rq = s.post(url, headers=self._headers, data=data)
            soup = bs.BeautifulSoup(rq.text, 'lxml')
        print(soup)
        table = soup.find('table', {'id': 'curr_table'})
        
        scrap = []
        
        for row in table.findAll('tr')[1:-1]:
            col = row.findAll('td')[:-1]
            scrap.append({
                'date': col[0].text,
                'close': col[1].text,
                'open': col[2].text,
                'high': col[3].text,
                'low': col[4].text,
                'volume': col[5].text
            })

        df = pd.DataFrame.from_records(scrap)
        df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y') # Parse dates
        df.set_index('date', inplace=True)
        
        # Cast to int
        df['volume'] = df['volume'].str[:-1]
        
        # Cast to int
        for c in ['close', 'open', 'high', 'low']:
            df[c] = df[c].str.replace(',', '')
            
        if df['volume'].isnull().values.all():
            df.drop(columns='volume', inplace=True)
            
        else:
            factor = df['volume'].str[:-1]
            factor = factor.map({'K': 1e3, 'M': 1e6, 'B': 1e9})
            
            df['volume'] = df['volume'].str[:-1]
            df['volume'] = df['volume'].apply(pd.to_numeric, errors='coerce')
            df['volume'] = df['volume'] * factor
            
        df = df.apply(pd.to_numeric, errors='coerce')

        return df

def getCountryIds():
    
    url = ('https://www.investing.com/stock-screener/'
        '?sp=country::5|sector::a|industry::a|equityType::a|exchange::a%3Ceq_market_cap;1')
        
    with requests.Session() as s:
        rq = s.get(url, headers=Ticker._headers)
        soup = bs.BeautifulSoup(rq.text, 'lxml')
    
    ul = soup.find('ul', {'id': 'countriesUL'})
    
    scrap = []
    for li in ul.findAll('li'):
        scrap.append({
            'id': li.get('data-value'), # Country id
            'country': re.sub('[^A-Za-z ]+', '', li.text) # Country
        })
    
    df = pd.DataFrame.from_dict(scrap)
    
    url = 'https://www.investing.com/stock-screener/Service/SearchStocks'
    
    nEquities = []
    for c in df['id']:
        
        data = {
            'country[]': c,
            'sector': 'a',
            'industry': 'a',
            'equityType': (
                'ORD,DRC,Preferred,Unit,ClosedEnd,REIT,ELKS,'
                'OpenEnd,Right,ParticipationShare,'
                'CapitalSecurity,PerpetualCapitalSecurity,'
                'GuaranteeCertificate,IGC,Warrant,SeniorNote,'
                'Debenture,ETF,ADR,ETC,ETN'
            ),
            'pn': '1',
            'order[col]': 'name_trans', # tech_sum_month
            'order[dir]': 'a'
        }
        with requests.Session() as s:
            rq = s.post(url, headers=Ticker._headers, data=data)
            parse = json.loads(rq.text)
        
        nEquities.append(parse['totalCount'])
        
    df['equities'] = nEquities
        
    path = Path().cwd() / 'data' / 'investingCountryId.csv'
    df.to_csv(path, index=False, encoding='utf-8-sig')
    
def getTickers(instrument):

    def extractJson(parse):
        scrap = []
        for t in parse['hits']:
            scrap.append({
                'ticker': t['viewData'].get('symbol'),
                'name': t['viewData'].get('name'),
                'investingId': t.get('pair_ID'),
                'investingLink': t['viewData'].get('link'),
                'sector': t.get('sector_trans'),
                'industry': t.get('industry_trans'),
                'exchangeId': t.get('exchange_ID'), 
                'exchange': t.get('exchange_trans'), 
                #'country': t['viewData'].get('flag'),
                'countryId': c,
            })
        return scrap
    
    if instrument == 'stock':
        
        data = {
            'country[]': '',
            'sector': 'a',
            'industry': 'a',
            'equityType': (
                'ORD,DRC,Preferred,Unit,ClosedEnd,REIT,ELKS,'
                'OpenEnd,Right,ParticipationShare,'
                'CapitalSecurity,PerpetualCapitalSecurity,'
                'GuaranteeCertificate,IGC,Warrant,SeniorNote,'
                'Debenture,ETF,ADR,ETC,ETN'
            ),
            'pn': '',
            'order[col]': 'name_trans', # tech_sum_month
            'order[dir]': 'a'
        }
        url = 'https://www.investing.com/stock-screener/Service/SearchStocks'
        
        path = Path().cwd() / 'data' / 'investingCountryId.csv'
        
        if not path.exists():
            getCountryIds()
            
        countries = pd.read_csv(path)
        countries['pages'] = np.ceil(countries['equities']/50)
        countries['pages'] = countries['pages'].astype(int)
        
        scrap = []
        for c, pages in zip(countries['id'], countries['pages']):
    
            for p in range(1, min(pages, 200)+1):
        
                data['country[]'] = c
                data['pn'] = str(p)
                
                with requests.Session() as s:
                    rq = s.post(url, headers=Ticker._headers, data=data)
                    parse = json.loads(rq.text)
                
                scrap.extend(extractJson(parse))
            
            if pages > 200:
                rest = pages - 200
                for p in range(1, rest + 1):
                
                    data['pn'] = str(p)
                    data['order[dir]'] = 'd'
            
                    try:
                        with requests.Session() as s:
                            rq = s.post(url, headers=Ticker._headers, data=data)
                            parse = json.loads(rq.text)
                        
                        scrap.extend(extractJson(parse))
                        
                    except:
                        print('Error at pn: ', p)
                        continue
    
    elif instrument == 'index':
        params = (
            ('majorIndices', 'on'),
            ('primarySectors', 'on'),
            ('additionalIndices', 'on'),
            ('otherIndices', 'on'),
        )
        url = 'https://www.investing.com/indices/world-indices'
        
        with requests.Session() as s:
            rc = s.get(url, headers=Ticker._headers, params=params)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
        
        tbls = soup.findAll('table', {'class': 'genTbl closedTbl crossRatesTbl elpTbl elp30'})

        scrap = []
        for t in tbls:
            for r in t.findAll('tr')[1:]:
                tds = r.findAll('td')
                
                scrap.append({
                    'investingId': tds[1].find('span').get('data-id'),
                    'name': tds[1].find('a').text,
                    'country': tds[0].find('span').get('title'),
                    'investingLink': tds[1].find('a').get('href')
                })
    
    df = pd.DataFrame.from_records(scrap)
    df.drop_duplicates(inplace=True)

    if instrument == 'stock':
        # Add security type
        df['type'] = 'stock'
        patterns = {
            'fund': r'((?<!reit)|(?<!estate)|(?<!property)|(?<!exchange traded)).+\bfundo?\b',
            'etf': r'(\b(et(c|f|n|p)(s)?)\b)|(\bxtb\b)|(\bexch(ange)? traded\b)',
            'bond': (
                r'(\s\d(\.\d+)?%\s)|'
                r'(t-bill snr)|((\bsnr )?\b(bnd|bds|bonds)\b)|'
                r'(\bsub\.? )?\bno?te?s( due/exp(iry)?)?(?! asa)'),
            'warrant': r'(\bwt|warr(na|an)ts?)\b(?!.+\b(co(rp)?|l(imi)?te?d|asa|ag|plc|spa)\.?\b)',
            'reit': (
                r'(\breit\b)|(\b(estate|property)\b.+\b(fund|trust)\b)|'
                r'(\bfii\b|\bfdo( de)? inv imob\b)|'
                r'(\bfundos?( de)? in(f|v)est(imento)? (?=imob?i?li(a|á)?ri(a|o)\b))')
        }
        for k, v in patterns.items():
            mask = df['name'].str.contains(v, regex=True, flags=re.I)
            df.loc[mask, 'type'] = k
        # Remove bonds, warrants and REITs
        df = df[~df['type'].isin(['bond', 'warrant', 'reit'])]

        # Remove warrants
        mask = (
            df['ticker'].str.contains(r'_t[au0-9]*$', regex=True) |
            df['investingLink'].str.contains(r'-w(nt|ts)$', regex=True)
        )
        df = df[~mask]

        # Label preferred stocks
        mask = (
            (df['ticker'].str.contains(r'_p$', regex=True) |
            df['investingLink'].str.contains(r'-pref(-[a-z]+)*$', regex=True)
            ) & ~df['name'].str.contains(r'\bpref\b', regex=True, flags=re.I)
        )
        df.loc[mask, 'name'] += ' Pref'

        # Trim tickers
        df['tickerTrim'] = df['ticker'].str.lower()

        patterns = {
            'Oslo': r'-me$',
            'Moscow': r'-rm(dr)?$',
            'Johannesburg': r'(?<=^[0-9a-z]{3})jn?$',
            'Hamburg': r'(?<=^[0-9a-z]{3})p$',
            'Hong Kong': r'(^0+(?=[1-9][0-9]+))',
            'Tel Aviv': r'-(l|m)$'
        }
        for k, v in patterns.items():
            mask = df['exchange'] == k
            df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'].str.replace(v, '', regex=True)

        pattern = (
            r'(\s|\.|-|_|/|\')'
        )
        df['tickerTrim'] = df['ticker'].str.replace(pattern, '', regex=True)

        # Change exchange of XLON tickers starting with 0
        mask = (df['exchange'] == 'XLON') & df['ticker'].str.contains(r'^0')
        df.loc[mask, 'exchange'] = 'LTS'

    return df

def stockData(exchange):
        
    query = f'SELECT countryId FROM investing WHERE exchange = "{exchange}"'
    engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\ticker.db')

    assets = pd.read_sql(query, con=engine)
    pages = int(np.ceil(len(assets)/50))
    cId = assets.iloc[0,0].squeeze()
        
    url = 'https://www.investing.com/stock-screener/Service/SearchStocks'
    
    scrap = []
    for p in range(1, pages + 1):
        data = {
            'country[]': cId,
            'sector': 'a',
            'industry': 'a',
            'exchange': exchange,
            'equityType': (
                'ORD,DRC,Preferred,Unit,ClosedEnd,REIT,ELKS,'
                'OpenEnd,Right,ParticipationShare,'
                'CapitalSecurity,PerpetualCapitalSecurity,'
                'GuaranteeCertificate,IGC,Warrant,SeniorNote,'
                'Debenture,ETF,ADR,ETC,ETN'
            ),
            'pn': str(p),
            'order[col]': 'name_trans', # tech_sum_month
            'order[dir]': 'a'
        }
        
        with requests.Session() as s:
            rc = s.post(url, headers=Ticker._headers, data=data)
            parse = json.loads(rc.text)
        
        for t in parse['hits']:
            scrap.append({
                'name': t['viewData'].get('name'),
                'ticker': t['viewData'].get('symbol'), # Ticker
                'invistingId': t.get('pair_ID'), # Investing ID
                'exchangeId': t.get('exchange_ID'), # Exchange ID
                'exchange': t.get('exchange_trans'), # Exchange
                'country': t['viewData'].get('flag'), # Country
                'industry': t.get('industry_trans'), # Industry
                'sector': t.get('sector_trans'), # Sector
                'ret1d': float(t.get('daily', np.nan)), # Daily return
                'ret1w': float(t.get('week', np.nan)), # Weekly return
                'ret1m': float(t.get('month', np.nan)), # Monthly return
                'retTy': float(t.get('ytd', np.nan)), # Return this year
                'ret1y': float(t.get('year', np.nan)), # Annual return
                'ret3y': float(t.get('3year', np.nan)), # Return (3y)
                'beta': float(t.get('eq_beta', np.nan)),
                'eps': float(t.get('eq_eps', np.nan)), # EPS
                'epsGwth5y': float(t.get('epstrendgr', np.nan)), # EPS growth (5y)
                'revGwth5y': float(t.get('revtrendgr', np.nan)), # Sales growth (5y)
                'capExGwth5y': float(t.get('csptrendgr', np.nan)), # Capital spending growth (5y)
                'peRatio': float(t.get('eq_pe_ratio', np.nan)), # P/E
                'pbRatio': float(t.get('price2bk', np.nan)), # P/B
                'psRatio': float(t.get('ttmpr2rev', np.nan)), # P/S
                'pfcfRatio': float(t.get('ttmprfcfps', np.nan)), # P/FCF
                'ato': float(t.get('ttmastturn', np.nan)), # Asset turnover
                'revEmpl': float(t.get('ttmrevpere', np.nan)), # Revenue/Employee (ttm)
                'netIncEmpl': float(t.get('ttmniperem', np.nan)), # Net Income/Employee (ttm)
                'cuRatioMrq': float(t.get('qcurratio', np.nan)), # Current ratio (mrq)
                'quickRatioMrq': float(t.get('qquickrati', np.nan)), # Quick ratio (mrq)
                'dbtEqRatioMrq': float(t.get('qtotd2eq', np.nan))/100, # Debt/Equity (mrq)
                'ltDbtEqRatioMrq': float(t.get('qltd2eq', np.nan))/100, # Long-term debt/equity (mrq)
                'npm5yAvg': float(t.get('margin5yr', np.nan))/100, # Net Profit Margin (5ya)
                'opm5yAvg': float(t.get('opmgn5yr', np.nan))/100, # Operating Margin (5ya)
                'divYld': float(t.get('yield', np.nan)), # Dividend yield
                'divYldGwth': float(t.get('divgrpct', np.nan)), # Dividend yield growth
                'signalWly': t.get('tech_sum_week'),
                'signalMly': t.get('tech_sum_month'),                    
                'adx': float(t.get('ADX', np.nan)),
                'atr': float(t.get('ATR', np.nan)),
                'cci': float(t.get('CCI', np.nan)),
                'rsi': float(t.get('RSI', np.nan)),
                'macd': float(t.get('MACD', np.nan)),
                'so': float(t.get('STOCH', np.nan)),
                'uo': float(t.get('UO', np.nan)),
                'williamsR': float(t.get('WilliamsR', np.nan))
            })
    
    df = pd.DataFrame.from_records(scrap)
    
    return df
            
def getInvestingData(urlExt, instrument='indices', startDate='01/01/2000', ticker='', df=None):
    # Date format: %m/%d/%Y
    
    if isinstance(startDate,  dt):
        startDate = startDate.strftime('%m/%d/%Y') # Convert datetime to string
    
    endDate = dt.datetime.now().strftime('%m/%d/%Y') 
    url = f'https://www.investing.com/{instrument}/{urlExt}-historical-data'

    # Get post args
    with requests.Session() as s:
        rc = s.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs.BeautifulSoup(rc.text, 'lxml')

    patternPair = re.compile('pairId: \d+')
    patternSml = re.compile('smlId: \d+')

    script = soup.find('script', text=patternSml)

    if script:
        match = patternPair.search(script.text)
        if match:
            pairId = match.group().split(' ', 2)[1]
        match = patternSml.search(script.text)
        if match:
            smlId = match.group().split(' ', 2)[1]

    headerVal = soup.find('div', {'class': 'instrumentHeader'}).findChild().text

    data = [
        ('curr_id', pairId),
        ('smlID', smlId),
        ('header', headerVal),
        ('st_date', startDate),
        ('end_date', endDate),
        ('interval_sec', 'Daily'),
        ('sort_col', 'date'),
        ('sort_ord', 'DESC'),
        ('action', 'historical_data'),
    ]
    
    url = 'https://www.investing.com/instruments/HistoricalDataAjax'
    
    # Get historical data
    with requests.Session() as s:
        rc = s.post(url, headers=Ticker._headers, data=data)
        soup = bs.BeautifulSoup(rc.text, 'lxml')

    table = soup.find('table', {'id': 'curr_table'})
    
    scrap = []
    
    for row in table.findAll('tr')[1:-1]:
        scrapRow = []
        for col in row.findAll('td'):
            scrapRow.append(col.text)
        scrap.append(scrapRow)
    
    cols = ['date', 'close', 'open', 'high', 'low', 'volume', 'roc']
    data = pd.DataFrame(scrap, columns=cols)
    data['date'] = pd.to_datetime(data['date'], format='%b %d, %Y') # Parse dates
    data.set_index('date', inplace=True)
    
    # Cast to int
    data['roc'] = data['roc'].str[:-1]
    data['volume'] = data['volume'].str[:-1]
    data = data.apply(pd.to_numeric, errors='coerce')
    
    if ticker and (df is not None):
        data.columns = [f'{ticker}_' + str(col) for col in data.columns]
        if df.empty:
            df = data
        else:
            df.combine_first(data)
        return df
    else:
        return data

#def getInvestingTechSum(): # Update
    
def indexData(link):
    url = f'https://www.investing.com/indices/{link}'
    
    with requests.Session() as s:
        rq = s.get(url, stream=True, headers=Ticker._headers)
        soup = bs.BeautifulSoup(rq.text, 'lxml')
    table = soup.find('table', {'class': 'genTbl closedTbl technicalSummaryTbl'})
    
    td = table.findAll('tr')[3].findAll('td')

    scrap = [link]
    for col in td[len(td)-3:]:
        scrap.append(col.text)

    cols = ['investingLink', 'daily', 'weekly', 'monthly']
    df = pd.DataFrame(scrap, columns=cols)
    
    return df