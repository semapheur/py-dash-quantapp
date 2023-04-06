import numpy as np
import pandas as pd

import requests
import bs4 as bs

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Date
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

# Utils
import re
from tqdm import tqdm

# Local
from lib.foos import renamer, replaceAll
from lib.finlib import finItemRenameDict

class Ticker():

    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-GPC': '1',
        'TE': 'Trailers',
    }

    def __init__(self, ticker, exchange, country):
        self._url = (
            'https://www.barrons.com/quote/stock/'
            f'{country}/{exchange}/{ticker}')

    def financials(self):

        rnm = finItemRenameDict('Barrons')
            
        rplFig = {'(': '-', ',': '', ')': ''} # Character replacements
        rplPost = {'+': '', '}': ')'}
    
        url = self._url + '/financials'
    
        with requests.Session() as s:
            rc = s.get(url, headers=self._headers, stream=True)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
        
        if soup.find('span', {'class': 'data_none'}) is not None:
            return None
        
        cl = "table table-condensed table-hover table--grid border-bottom"
        tbls = soup.findAll('table', {'class': cl})
        
        if not tbls:
            
            path = 'C:/Users/danfy/OneDrive/FinAnly/driver/geckodriver.exe'
                
            ffOptions = Options()
            ffOptions.add_argument('--headless')
            #ffOptions.add_argument('--window-size=1920x1080')
            
            browser = Firefox(
                options=ffOptions,
                executable_path=path)
            browser.get(url)
            
            # Wait to load tables
            wait = WebDriverWait(browser, 10)
            try:
                loadTbls = wait.until(EC.presence_of_all_elements_located(
                    (By.XPATH, f'//table[@class = {cl}]')))
            except:
                print('')
                
            # html = ''
            # for lt in loadTbls:
            #     html += lt.get_attribute('innerHTML')
                
            html = browser.page_source
            soup = bs.BeautifulSoup(html, 'lxml')
            
            tbls = soup.findAll('table', {'class': cl})
            browser.quit()
            
            if not tbls:
                return None
        
        dfs = [] # [None] * len(tbls)
        ok = []
        for i, t in enumerate(tbls):
    
            # Headers
            header = t.findAll('th')
    
            dscl = header[0].text
            
            pattern = '(?<=-)[ADFJMNOS][a-z]{2,8}(?=\.)'
            
            match = re.search(pattern, dscl)
            
            if not match:
                continue
            
            ok.append(i)
            
            month = match.group()
            
            if 'Thousands' in dscl:
                mltp = 1e3
    
            elif 'Millions' in dscl:
                mltp = 1e6
    
            # Dates
            dates = [None] * len(header[1:-1])
                
            for j, h in enumerate(header[1:-1]):
    
                strDate = h.text
    
                if re.search('^\d{4}$', strDate):
                    #strDate = f'12/31/{strDate}'
                    strDate = f'1 {month} {strDate}'
                    date = dt.strptime(strDate, '%d %B %Y')
                    date += relativedelta(months=1) - relativedelta(days=1)
                    dates[j] = date
                else:
                    dates[j] = dt.strptime(strDate, '%m/%d/%Y')
    
            # Figures
            scrap = {}
            for r in t.findAll('tr')[1:]:
    
                post = r.find('td', {'class': 'rowTitle'}).text
                post = replaceAll(post, rplPost)
    
                expt = ['Growth', 'Margin', 'Turnover', '/ Total Assets', 
                        '/ Sales', 'Yield']
                if not any(x in post for x in expt):
                    tds = r.findAll('td')[1:-1]
                    scrapRow = [None] * len(tds)
    
                    for k, td in enumerate(tds):
    
                        fig = td.text
    
                        if fig == '-' or fig == '' or fig == '()':
                            fig = np.nan
    
                        elif '%' in fig:
                            fig = float(fig.replace('%', ''))/100
    
                        else:
                            fig = float(replaceAll(fig, rplFig))
    
                            if 'EPS' not in post:
                                fig *= mltp
    
                        scrapRow[k] = fig
    
                    scrap.update({post: scrapRow})
    
            temp = pd.DataFrame.from_dict(scrap, orient='index', columns=dates)
            temp = temp.T
            #temp.index = dates
            temp.index.rename('date', inplace=True)
            temp.sort_index(ascending=True, inplace=True)
    
            if i == 2 or i > 10:
    
                pattern = 'Shares'
    
                mask = temp.columns.str.contains(pattern, regex=True)
    
                for c in temp.loc[:, ~mask]:
                    temp[c] = temp[c].rolling(4, min_periods=4).sum()
    
            dfs.append(temp)
        
        # Concat
        
        b = 2 # Balance sheet start index
        c = 8 # Cash flow sheets start index
        
        # Income
        if all([i in ok for i in [0, 1]]): 
            dfInc = pd.concat(dfs[0:2])
            dfInc = dfInc.loc[~dfInc.index.duplicated(keep='first')]
            
        else:
            dfInc =dfs[0]
            b -= 1
            c -= 1
            
        dfInc.drop('Preferred Dividends', axis=1, inplace=True)
        
        # Balance
        if all([i in ok for i in range(2, 8)]):
            dfBlcA = pd.concat(dfs[b:b+3], axis=1) 
            dfBlcQ = pd.concat(dfs[b+3:b+6], axis=1)
            dfBlc = pd.concat([dfBlcA, dfBlcQ])
            dfBlc = dfBlc.loc[~dfBlc.index.duplicated(keep='first')]
            
        else:
            dfBlc = pd.concat(dfs[b:b+3], axis=1)
            c -= 3
        
        dfBlc.rename(columns={'Deferred Tax': 'Deferred Tax Liabilities'}, 
                     inplace=True)
        
        # Cash
        if all([i in ok for i in range(8,14)]):
            dfCashA = pd.concat(dfs[c:c+3], axis=1)
            dfCashQ = pd.concat(dfs[c+3:], axis=1)
            dfCash = pd.concat([dfCashA, dfCashQ])
            dfCash = dfCash.loc[~dfCash.index.duplicated(keep='first')]
        
        else:
            dfCash = pd.concat(dfs[c:c+3], axis=1)
        
        df = pd.concat([dfInc, dfBlc, dfCash], axis=1)
    
        # Remove duplicate index
        #df = df[~df.index.duplicated(keep='first')]
        
        # Rename columns
        df.rename(columns=renamer(), inplace=True) # Rename duplicate columns
        df.rename(columns=rnm, inplace=True)
            
        # Drop duplicate columns
        df = df.loc[:,~df.columns.duplicated()]
        
        df['capEx'] *= -1
        
        # Additional posts
        df['opInc'] = df['grsPrft'] - df['opEx']
        df['intCov'] = (df['ebitda'] + df['da']) / df['intEx']
        df['taxRate'] = df['taxEx'] / df['ebt'] 
    
        df['wrkCap'] = (
            (df['cce'] + df['stInv'] + 0.75 * df['rcv'] +
             0.5 * df['ivty']).rolling(2, min_periods=0).mean() - 
            df['totCrtLbt'].rolling(2, min_periods=0).mean())
    
        # Tangible equity
        df['tgbEqt'] = (df['totEqt'] - df['prfEqt'].fillna(0) - 
                       df['gw'].fillna(0) - df['itgbAst'].fillna(0))
    
        # Total non-current assets
        df['totNoCrtAst'] = (df['ppe'].fillna(0) + df['ltInv'].fillna(0) +
                           df['itgbAst'].fillna(0) +
                           df['othNoCrtAst'].fillna(0))
    
        # Total non-current liabilities
        df['totNoCrtLbt'] = (df['ltDbt'].fillna(0) + 
                             df['noCrtDfrTaxLbt'].fillna(0) + 
                             df['prvRiskChrg'].fillna(0) +
                             df['othLbt'].fillna(0))
    
        # Total debt
        df['totDbt'] = (df['stDbt'].fillna(0) + df['ltDbt'].fillna(0))
        
        return df
    
    def quote(self):
    
        with requests.Session() as s:
            rc = s.get(self._url, headers=self._headers)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
            
        quote = soup.find('span', {'class': 'market__price bgLast'})
        
        return quote

def getTickers():
    
    def parse(tbl, country):
        
        noXchg = {
            # Stocks with missing exchange
            'ACON S2 Acquisition Corp. Wt STWOW': 'XNAS',
            'AGNC Investment Corp. 6.125% Cum. Redeem. Pfd. Series F AGNCP': 'XNAS',
            'Allegro MicroSystems Inc. ALGM': 'XNAS',
            'Arch Capital Group Ltd. Dep. Pfd. (Rep. 1/1000th 4.550% Pfd. Series G) ACGLN': 'XNAS',
            'ARYA Sciences Acquisition Corp. V Cl A ARYE': 'XNAS',
            'Atlantic Street Acquisition Corp. Wt ASAQ.WT': 'XNYS',
            'Atlas Financial Holdings Inc. AFHIF': 'OOTC',
            'Blue Safari Group Acquisition Corp. Cl A BSGA': 'XNAS',
            'Burford Capital Ltd. BUR': 'XNYS',
            'Burgundy Technology Acquisition Corp. Wt BTAQW': 'XNAS',
            'CBL & Associates Properties Inc. CBLAQ': 'OOTC',
            'CM Life Sciences III Inc. CMLTU': 'XNAS',
            'Coliseum Acquisition Corp. MITAU': 'XNAS',
            'Corner Growth Acquisition Corp. Cl A COOL': 'XNAS',
            'Destination XL Group Inc. DXLG': 'OOTC',
            'Dune Acquisition Corp. Cl A DUNE': 'XNAS',
            'Emmis Communications Corp. EMMS': 'OOTC',
            'Exasol AG EXLGF': 'PSGM',
            'Extraction Oil & Gas Inc. XOGAQ': 'XNAS',
            'GreenPower Motor Co. Inc. GP': 'XNAS',
            'Greenrose Acquisition Corp. Wt GNRSW': 'OOTC',
            'Hancock Whitney Corp. 6.25% Sub. Notes due 2060 HWCPZ': 'XNAS',
            'Healthcare Services Acquisition Corp. Cl A HCAR': 'XNAS',
            'Horizonte Minerals PLC HZMMF': 'OOTC',
            'Interpace Biosciences Inc. IDXG': 'OOTC',
            'LATAM Airlines Group S.A. ADR LTMAQ': 'OOTC',
            'Leisure Acquisition Corp. Wt LACQW': 'XNAS',
            'Linx S.A. ADR LINX': 'XNYS',
            'Lionheart Acquisition Corp. II Wt LCAPW': 'XNAS',
            'Mallinckrodt PLC MNKKQ': 'OOTC',
            'Medalist Diversified REIT Inc. Cum. Redeem. Pfd. Series A MDRRP': 'XNAS',
            'Noble Corp. PLC NEBLQ': 'XNYS',
            'Oasis Petroleum Inc. OASPQ': 'XNAS',
            'Obsidian Energy Ltd. OBELF': 'OOTC',
            'OceanTech Acquisitions I Corp. Cl A OTEC': 'XNAS',
            'Oxford Lane Capital Corp. 6.75% Notes due 2031 OXLCL': 'XNAS',
            'Paringa Resources Ltd. ADR PNRLY': 'OOTC',
            'PropTech Investment Corp. II Cl A PTIC': 'XNAS',
            'Roth CH Acquisition I Co. ROCH': 'XNAS',
            'Savannah Resources PLC SAVNF': 'OOTC',
            'Superconductor Technologies Inc. SCON': 'OOTC',
            'Tandy Leather Factory Inc. TLFA': 'OOTC',
            'USHG Acquisition Corp. HUGS.UT': 'XNYS',
            'Westell Technologies Inc. WSTL': 'OOTC',
            'Wins Finance Holdings Inc. WINSF': 'OOTC',
        }
        scrap = []
        for tr in tbl.findAll('tr')[1:]:
            td = tr.findAll('td')

            ticker = td[0].find('a').get('href').split('/')[-1]
            countryCode = td[0].find('a').get('href').split('/')[-3]
            
            if country == 'United States':
                xchg = noXchg.get(td[0].find('a').text, td[1].text)
            else:
                xchg = td[1].text

            scrap.append({
                'ticker': ticker,
                'name': td[0].find('a').text,
                'sector': td[2].text,
                'exchange': xchg,
                'country': countryCode,
            }) 
        return scrap
        
    url = 'https://www.barrons.com/quote/company-list'
    
    with requests.Session() as s:
        rc = s.get(url, headers=Ticker._headers)
        soup = bs.BeautifulSoup(rc.text, 'lxml')
        
    ulCountry = soup.findAll('ul', {'class': 'cl-list'})[1]
    
    dctCountry = {}
    for li in ulCountry.find_all('li'):
        
        #if 'break' not in li['class']:
        if li.find('a') is not None:
            key = li.find('a').text # Country
            val = li.find('a').get('href')
            dctCountry[key] = val
    
    # Get tickers
    scrap = []
    for k in tqdm(dctCountry):
        url = dctCountry[k] 
        
        with requests.Session() as s:
            rc = s.get(url, headers=Ticker._headers)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
            
        # Scrap first page
        tbl = soup.find('table', {'class': 'cl-table'})
        scrap += parse(tbl, k)
        
        # Find pagination
        pg = soup.find('ul', {'class': 'cl-pagination'})
        if pg is not None:
            numPages = pg.findAll('li')
        
            lastPage = numPages[-1].find('a').text
        
            if '-' in lastPage:
                lastPage = lastPage.split('-')[-1]
            
            if k == 'United States':
            
                urlPage = url + f'/{lastPage}'
                
                with requests.Session() as s:
                    rc = s.get(urlPage, headers=Ticker._headers)
                    soup = bs.BeautifulSoup(rc.text, 'lxml')
                    
                # Find pagination
                pg = soup.find('ul', {'class': 'cl-pagination'})
                numPages = pg.findAll('li')
                lastPage = numPages[-1].find('a').text
                
                if '-' in lastPage:
                    lastPage = lastPage.split('-')[-1]
                
            lastPage = int(lastPage)
            pages = [str(i) for i in range(2, lastPage+1)]
        
            for p in pages:
                url = dctCountry[k] + f'/{p}'
                
                with requests.Session() as s:
                    rc = s.get(url, headers=Ticker._headers)
                    soup = bs.BeautifulSoup(rc.text, 'lxml')
            
                tbl = soup.find('table', {'class': 'cl-table'})
                
                #scrap.extend(parse(tbl, k))
                scrap += parse(tbl, k)

    df = pd.DataFrame.from_records(scrap)
    df.drop_duplicates(inplace=True)
    
    # Remove warrants and pre-IPO stocks
    mask = (
        df['ticker'].str.contains( r'\.WT[ABCDRS]?$', regex=True) | 
        (df['exchange'] == 'IPO') 
    )
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
    df = df[~df['type'].isin(['bond', 'warrant', 'reit'])]

    # Split tickers from name
    #pattern = r'\s[0-9A-Z]+(\.?[0-9A-Z]+)*$'
    #df['name'] = df['name'].apply(lambda x: re.sub(pattern, '', x))
    df['name'] = [a.replace(b, '').strip() for a, b in zip(df['name'], df['ticker'])]

    # Trim tickers
    df['tickerTrim'] = df['ticker'].str.lower()

    patterns = {
        'XIST': r'\.e$',
        'XTAE': r'\.(m|l)$',
    }
    for k, v in patterns.items():
        mask = df['exchange'] == k
        df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'].str.replace(v, '', regex=True)

    pattern = (
        r'((?<=\.p)r(?=\.?[a-z]$))|' # .PRx
        r'((?<=\.w)i(?=\.?[a-z]$))|' #.WIx
        r'((?<=\.(r|u|w))t(?=[a-z]?$))|' # .RT/UT/WT
        r'((\.|/)?[inpx]\d{4,5}$|\.ca\.?$|\.zw$|\s|\.|_|-|/|\')|'
        r'(^0+(?=[1-9][0-9a-z]*))' # Leading zeroes
    )
    df['tickerTrim'] = df['tickerTrim'].str.replace(pattern, '', regex=True)

    # Change exchange of XLON tickers starting with 0
    mask = (df['exchange'] == 'XLON') & df['ticker'].str.contains(r'^0')
    df.loc[mask, 'exchange'] = 'LTS'
                
    return df

import sqlalchemy as sqla
if __name__ == '__main__':
    df = getTickers()
    engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\ticker.db')

    src = 'barrons'
    for t in df['type'].unique(): # Store stock and etf tickers separately
        mask = df['type'] == t
        df.loc[mask, df.columns != 'type'].to_sql(
            f'{src}{t.capitalize()}', con=engine, index=False, if_exists='replace')
       
      
