# Data analysis
import numpy as np
import pandas as pd

# Date
from datetime import datetime as dt
from datetime import timezone as tz
import time

# Web scrapping
import requests
import bs4 as bs
import json

# Utils
import re
from tqdm import tqdm
import textwrap

class Ticker:
    
    # Class variable
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Sec-GPC': '1',
        'TE': 'Trailers',
    }
    
    def __init__(self, ticker):
        self._ticker = ticker

    @classmethod
    def _crumbCookies(cls):
        
        url = 'https://finance.yahoo.com/lookup'
        with requests.session() as s:
            rq = requests.get(url, headers=cls._headers)
            soup = bs.BeautifulSoup(rq.text, 'lxml')
            crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

        return crumb[0], rq.cookies
    
    def ohlcv(self, startDate='', period=None):
        
        def parseJson(startStamp, endStamp):
            params = (
                ('formatted', 'true'),
                #('crumb', '0/sHE2JwnVF'),
                ('lang', 'en-US'),
                ('region', 'US'),
                ('includeAdjustedClose', 'true'),
                ('interval', '1d'),
                ('period1', str(startStamp)),
                ('period2', str(endStamp)),
                ('events', 'div|split'),
                ('corsDomain', 'finance.yahoo.com'),
            )
            url = f'https://query2.finance.yahoo.com/v8/finance/chart/{self._ticker}'

            with requests.Session() as s:
                rq = s.get(url, headers=self._headers, params=params)
                parse = json.loads(rq.text)
            
            return parse['chart']['result'][0]
                    
        if startDate and (period is None):

            if isinstance(startDate, str):
                startDate = dt.strptime(startDate, '%Y-%m-%d')

            startStamp = int(startDate.replace(tzinfo=tz.utc).timestamp())
            
        elif period == 'max':
            startStamp = int(dt.today().timestamp())
            startStamp -= 3600 * 24

        else:
            startStamp = int(dt(2000, 1, 1).replace(tzinfo=tz.utc).timestamp())
        
        endStamp = int(dt.now().replace(tzinfo=tz.utc).timestamp())
        endStamp += 3600 * 24

        parse = parseJson(startStamp, endStamp)
        
        if period == 'max':
            startStamp = parse['meta']['firstTradeDate']
            parse = parseJson(startStamp, endStamp)
        
        # Index
        ix = parse['timestamp']

        # OHLCV
        ohlcv = parse['indicators']['quote'][0]
        adjClose = parse['indicators']['adjclose'][0]['adjclose']

        data = {
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'adjClose': adjClose,
            'volume': ohlcv['volume'],
        }

        # Parse to DataFrame
        df = pd.DataFrame.from_dict(data, orient='columns')
        df.index = pd.to_datetime(ix, unit='s').floor('D') # Convert index from unix to date
        df.index.rename('date', inplace=True)

        return df

    def priceTargets(self):

        url = f'https://finance.yahoo.com/quote/{self._ticker}/analysis'

        with requests.Session() as s:
            rq = s.get(url, headers=self._headers)
            soup = bs.BeautifulSoup(rq.text, 'lxml')

        div = soup.find('div', {'id': 'Col2-9-QuoteModule-Proxy'})

        scrap = div.find('div', {'aria-label': True}).get('aria-label')

        pt = scrap.split(' ')[1::2]
        pt.insert(0, self._ticker)
        cols = ['ticker', 'low', 'current', 'average', 'high']
        df = pd.DataFrame(pt, columns=cols)
        return df

    def financials(self):
        
        #filePath = Path.cwd() / 'finItems.csv'
        filePath = r'C:\Users\danfy\OneDrive\FinAnly\data\finItems.csv'
        dfItems = pd.read_csv(filePath)
        
        def parse(period):
            
            mask = dfItems['source'] == 'Yahoo'
            items = dfItems.loc[mask, 'sourceLabel'].tolist()

            items = [period + i for i in items]
            items = ','.join(items)

            endStamp = int(dt.now().timestamp()) + 3600*24

            params = (
                ('lang', 'en-US'),
                ('region', 'US'),
                ('symbol', self._ticker),
                ('padTimeSeries', 'true'),
                ('type', items),
                ('merge', 'false'),
                ('period1', '493590046'),
                ('period2', str(endStamp)),
                ('corsDomain', 'finance.yahoo.com'),
            )
            url = (
                'https://query2.finance.yahoo.com/ws/'
                f'fundamentals-timeseries/v1/finance/timeseries/{self._ticker}'
            )
            with requests.Session() as s:
                rq = s.get(url, headers=self._headers, params=params)
                parse = json.loads(rq.text)

            dfs = []
            pattern = r'^(annual|quarterly)'
            for r in parse['timeseries']['result']:
                item = r['meta']['type'][0]

                if item in r:
                    scrap = {}
                    for e in list(filter(None, r[item])):
                        date = dt.strptime(e['asOfDate'], '%Y-%m-%d')
                        scrap[date] = e['reportedValue']['raw']

                    df = pd.DataFrame.from_dict(
                        scrap, orient='index', 
                        columns=[re.sub(pattern, '', item)])

                    dfs.append(df)

            df = pd.concat(dfs, axis=1)
            df['period'] = period[0]

            return df
        
        dfs = []
        for p in ['annual', 'quarterly']:
            dfs.append(parse(p))
            
        df = pd.concat(dfs)
        
        mask = dfItems['source'] == 'Yahoo'
        dfItems = dfItems.loc[mask, ['sourceLabel', 'itemVal']]
        colMap = {k: v for k, v in zip(dfItems['sourceLabel'], 
                                       dfItems['itemVal'])}
        
        df.rename(columns=colMap, inplace=True)
        df.set_index('period', append=True, inplace=True)
        df.index.names = ['date', 'period']

        # Additional items
        df['intCvg'] = (df['ebit'] / df['intEx'])
        df['taxRate'] = df['taxEx'] / df['ebt']

        return df

    def optionChains(self):

        def parseOptionChain(parse, stamp):
            
            def getEntry(opt, key):
                if key in opt:
                    if isinstance(opt[key], dict):
                        entry = opt[key].get('raw')
                    elif isinstance(opt[key], bool):
                        entry = opt[key]
                else:
                    entry = np.nan
                    
                return entry
            
            opts = parse['optionChain']['result'][0]['options'][0]

            cols = ['strike', 'impliedVolatility', 'openInterest', 'lastPrice', 
                    'ask', 'bid', 'inTheMoney']
            cells = np.zeros((len(cols), (len(opts['calls']) + len(opts['puts']))))
            
            # Calls
            for i, opt in enumerate(opts['calls']):
                for j, c in enumerate(cols):                               
                    cells[j, i] = getEntry(opt, c)
                    
            # Puts
            for i, opt in enumerate(opts['puts']):
                for j, c in enumerate(cols):
                    cells[j, i+len(opts['calls'])] = getEntry(opt, c)

            data = {k:v for k, v in zip(cols, cells)}
            data['optionType'] = np.array(
                ['call'] * len(opts['calls']) + ['put'] * len(opts['puts']))
            
            # Parse to data frame
            df = pd.DataFrame.from_records(data)

            # Add expiry date
            date = dt.utcfromtimestamp(stamp)
            df['expiry'] = date
                    
            return df
        
        params = [
            ('formatted', 'true'),
            #('crumb', '2ztQhfMEzsm'),
            ('lang', 'en-US'),
            ('region', 'US'),
            ('corsDomain', 'finance.yahoo.com'),
        ]
        
        url = (
            'https://query1.finance.yahoo.com/'
            f'v7/finance/options/{self._ticker}'
        )

        with requests.Session() as s:
            rq = s.get(url, headers=self._headers, params=params)
            parse = json.loads(rq.text)
        
        # Maturity dates
        stamps = parse['optionChain']['result'][0]['expirationDates']

        # Parse first option chain to dataframe
        dfs = []
        dfs.append(parseOptionChain(parse, stamps[0]))

        # Parse remaining option chains
        for i in range(1, len(stamps)):
            time.sleep(1)
            
            params[-1] = ('date', stamps[i])
            
            #q = (i % 2) + 1
            #url = f'https://query{q}.finance.yahoo.com/v7/finance/options/{ticker}'
            
            with requests.Session() as s:
                rq = s.get(url, headers=self._headers, params=params)
                parse = json.loads(rq.text)

            dfs.append(parseOptionChain(parse, stamps[i]))

        # Concatenate
        df = pd.concat(dfs)
        
        return df
    
def batchOhlcv(tickers, startDate='', period=None):

    if isinstance(tickers, str):
        ticker = Ticker(tickers)
        return Ticker.ohlcv(startDate, period)

    elif isinstance(tickers, list):
        dfs = []
        for t in tickers:
            ticker = Ticker(t)
            
            df = ticker.ohlcv(startDate, period)

            cols = pd.MultiIndex.from_product([[t], [c for c in df.columns]])
            df.columns = cols

            dfs.append(df)

        return pd.concat(dfs, axis=1)
    else:
        return None

def getTickers():

    def jsonParams(size, offset, region, price=0):

        regionQuery = f'{{operator:EQ,operands:[region,{region}]}}'

        if np.isinf(price):
            price = ""

        data = textwrap.dedent(f'''
            {{size:{size}, offset:{offset}, sortField:intradayprice, 
            sortType:asc, quoteType:EQUITY, topOperator:AND,
            query: {{operator:AND, operands:[
                {{operator:or,operands:[{regionQuery}]}},
                {{operator:gt,operands:[intradayprice,{price}]}}
            ]}},userId:"",userIdType:guid}}
        ''')

        return data

    def parseJson(crumb, cookies, size, offset, region, price=0):

        params = (
            ('crumb', crumb),
            ('lang', 'en-US'),
            ('region', 'US'),
            ('formatted', 'true'),
            ('corsDomain', 'finance.yahoo.com'),
        )
        url = 'https://query1.finance.yahoo.com/v1/finance/screener'

        with requests.Session() as s:
            rs = s.post(
                url, headers=Ticker._headers, params=params, cookies=cookies, 
                data=jsonParams(size, offset, region, price))                
            parse = json.loads(rs.text)

        return parse['finance']['result'][0]

    def scrapData(quotes):

        scrap = []
        for i in quotes:
            scrap.append({
                'ticker': i['symbol'],
                'name': i.get('longName', i.get('shortName', '').capitalize()),
                'exchange': i['exchange'],
                'exchangeName': i['fullExchangeName'],
            })

        return scrap

    crumb, cookies = Ticker.crumbCookies()

    regions = [
        'ar', 'at', 'au', 'be', 'bh', 'br', 'ca', 'ch', 'cl', 'cn', 'cz', 
        'de', 'dk', 'eg', 'es', 'fi', 'fr', 'gb', 'gr', 'hk', 'hu', 'id', 
        'ie', 'il', 'in', 'it', 'jo', 'jp', 'kr', 'kw', 'lk', 'lu', 'mx', 
        'my', 'nl', 'no', 'nz', 'pe', 'ph', 'pk', 'pl', 'pt', 'qa', 'ru', 
        'se', 'sg', 'sr', 'tf', 'th', 'tl', 'tn', 'tr', 'tw', 'us', 've',  
        'vn', 'za'  
    ]
    numRes = 1000
    size = 250
    limit = 10000

    scrap = []

    for r in tqdm(regions):
        offset = 0

        parse = parseJson(crumb, cookies, size, offset, r)
        offset += size

        if parse['quotes']:
            scrap.extend(scrapData(parse['quotes']))

            numRes = parse['total']

            if numRes > limit:

                price = 0
                scrapCount = len(scrap)
                flag = False
                while scrapCount <= numRes:

                    while offset < limit:

                        parse = parseJson(crumb, cookies, size, offset, r, price)

                        if parse['quotes']:
                            scrapBatch = scrapData(parse['quotes'])
                            scrap.extend(scrapBatch)
                            scrapCount += len(scrapBatch)

                            offset += size

                            if offset >= limit:
                                price = parse['quotes'][-1]['regularMarketPrice']['raw']

                        else:
                            flag = True
                            break

                    if flag:
                        break

                    offset = 0

            else:
                while offset < numRes:
                    parse = parseJson(crumb, cookies, size, offset, r)
                    scrap.extend(scrapData(parse['quotes']))
                    offset += size

    df = pd.DataFrame.from_dict(scrap)
    df.drop_duplicates(inplace=True)
    df = df[df['name'].astype(bool)] # Remove empty names

    # Remove options
    patterns = {
        'ASX': r'^Eqt xtb',
        'CCS': r'^(P|O)\.\s?c.',
        'VIE': r'^(Rc|Eg?)b\b',
        'OSL': r'^B(ull|ear)\b|\bpro$'
    }
    for k, v in patterns.items():
        mask = (df['exchange'] == k) & df['name'].str.contains(v, regex=True)
        df = df[~mask]

    patterns = {
        'SAO': r'F.SA$',
        'OSL': r'-PRO.OL$'
    }
    for k, v in patterns.items():
        mask = (df['exchange'] == k) & df['ticker'].str.contains(v, regex=True)
        df = df[~mask]

    # Remove warrants
    pattern = r'-W(R|T)?[A-DU]?(\.[A-Z]{1,3})?$'
    mask = df['ticker'].str.contains(pattern, regex=True)
    df = df[~mask]

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
    # Remove bonds
    df = df[df['type'] != 'bond']

    # Fix name
    replacements = (
        (r'(d(k|l)|eo|hd|i(l|s)|ls|nk|rc|s(d|f|k)|yc)?\s?\d?(-,|,-)(\d+)?', ''),
        (r'(^\s|["])', ''),
        (r'\s+', ' '),
        (r'&amp;', '&')
    )
    for old, new in replacements:
        df['name'] = df['name'].str.replace(old, new, regex=True)

    # Trim tickers
    df['tickerTrim'] = df['ticker'].str.lower().split('.')[0] #.apply(lambda x: x.lower().split('.')[0]) 

    patterns = {
        'TLV': r'-(l|m)$'
    }
    for k, v in patterns.items():
        mask = df['exchange'] == k
        df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'].str.replace(v, '', regex=True)

    # Remove suffixes and special characters
    pattern = r'(\s|-|_|/|\'|^0+(?=[1-9]\d*)|[inxp]\d{4,5}$)'
    pattern = (
        r'((?<=\.p)r(?=\.?[a-z]$))|' # .PRx
        r'((?<=\.w)i(?=\.?[a-z]$))|' #.WIx
        r'((?<=\.(r|u|w))t(?=[a-z]?$))|' # .RT/UT/WT
        r'((\.|/)?[inpx][047]{4,5}$)|'
        r'\s|\.|_|-|/|\')'
    )
    df['tickerTrim'] = df['tickerTrim'].str.replace(pattern, '', regex=True) 

    # Change exchange of XLON tickers starting with 0
    mask = (df['exchange'] == 'LSE') & df['ticker'].str.contains(r'^0')
    df.loc[mask, 'exchange'] = 'IOB'

    return df
