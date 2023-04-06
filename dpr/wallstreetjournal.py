import numpy as np
import pandas as pd

import requests
import bs4 as bs

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from datetime import datetime as dt

from pathlib import Path

from foos import replaceAll, finItemRenameDict

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
        self._url = ('https://www.wsj.com/market-data/quotes/'
            f'{country}/{exchange}/{ticker}/')
    
    def quote(self):
        
        with requests.Session() as s:
            rc = s.get(self._url, headers=self._headers)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
            
        quote = soup.find('span', {'id': 'quote_val'})
        
        if quote is None:
            driverPath = Path().cwd() / 'driver' / 'geckodriver.exe'
                        
            ffOptions = Options()
            ffOptions.add_argument('--headless')
            #ffOptions.add_argument("--window-size=1920x1080")
        
            browser = Firefox(options=ffOptions,
                executable_path=driverPath)
            browser.get(self._url)
            
            wait = WebDriverWait(browser, 10)
            quote = wait.until(EC.presence_of_element_located((By.XPATH, './/span[@id = "quote_val"]')))
            
        return float(quote.text.replace(',', ''))

    def financials(self):
    
        rnm = finItemRenameDict('Barrons')
            
        sheets = ['income-statement', 'balance-sheet', 'cash-flow']
        periods = ['annual/', 'quarter/']
        
        rpl = {'(': '-', ',': '', ')': ''} # Character replacements
        
        dfSheets = []
        for sh in sheets:
            
            dfPeriod = []
            for p in periods:
                url = self._url + f'/financials/{p}{sh}' 
                
                # with requests.Session() as s:
                #     rc = s.get(url, headers=headers, stream=True)
                #     soup = bs.BeautifulSoup(rc.text, 'lxml')
                        
                # tbls = soup.findAll('table', {'class': 'cr_dataTable'})
                
                driverPath = Path().cwd() / 'driver' / 'geckodriver.exe'
                
                ffOptions = Options()
                ffOptions.add_argument('--headless')
                #ffOptions.add_argument('--window-size=1920x1080')
                
                browser = Firefox(options=ffOptions,
                                            executable_path=driverPath)
                browser.get(url)
                
                # Wait to load tables
                wait = WebDriverWait(browser, 10)
                try:
                    loadTbls = wait.until(EC.presence_of_all_elements_located(
                        (By.XPATH, '//table[@class = "cr_dataTable"]')))
                except:
                    print('')
                    
                # html = ''
                # for lt in loadTbls:
                #     html += lt.get_attribute('innerHTML')
                    
                html = browser.page_source
                soup = bs.BeautifulSoup(html, 'lxml')
                
                tbls = soup.findAll('table', {'class': 'cr_dataTable'})
                browser.quit()
                
                # Find multiplier
                header = tbls[0].find('tr')
                dscl = header.find('th').text
                
                if 'Thousands' in dscl:
                    mltp = 1e3
                    
                elif 'Millions' in dscl:
                    mltp = 1e6
                
                # Dates
                dates = [] #[None] * len(header.findAll('th')[1:5])
                for d in header.findAll('th')[1:5]:
                    
                    if d.text == ' ':
                        continue
                    
                    else:
                    
                        if p == 'annual/':
                            strDate = f'31-Dec-{d.text}'
                            
                        else:
                            strDate = d.text
                    
                    dates.append(dt.strptime(strDate, '%d-%b-%Y'))         
                
                #dates = list(filter(None, dates))
                
                # Scrap data
                scrap = {}
                for tbl in tbls:
                    
                    trs = tbl.findAll(lambda x:
                        x.name == 'tr' and
                        not any(cl in x.get('class', []) for cl in ['barPos', 'barNeg'])
                    )[1:]
                    
                    for tr in trs:
                        # Post name
                        key = tr.find('td').text
                        
                        # Figures
                        if 'Growth' not in key:
                            scrapRow = [None] * len(dates)
                            for j, td in enumerate(tr.findAll('td')[1:len(dates)]):
                                
                                entry = td.text
                                
                                if entry == '-':
                                    entry = np.nan
                                
                                elif '%' in entry:
                                    entry = float(entry.replace('%', ''))/100
                                
                                else:
                                    entry = float(replaceAll(entry, rpl))
                                    
                                    if 'EPS' not in key:
                                        entry *= mltp
                                        
                                scrapRow[j] = entry
                                
                            scrap.update({key: scrapRow})
                        
                temp = pd.DataFrame.from_dict(scrap, orient='index', columns=dates)
                temp = temp.T
                temp.index.rename('date', inplace=True)
                temp.sort_index(ascending=True, inplace=True)
                #temp.dropna(axis=1, how='all', inplace=True)
                
                # TTM
                cond = (
                    (sh == 'income-statement' or sh == 'cash-flow') and
                    p == 'quarter/'
                )
                pattern = r'(Shares|/ Sales)'
                
                if cond:
                    mask = temp.columns.str.contains(pattern, regex=True)
                    
                    for c in temp.loc[:, ~mask]:
                        temp[c] = temp[c].rolling(4, min_periods=4).sum()
                
                dfPeriod.append(temp)
            
            temp = pd.concat(dfPeriod)
            dfSheets.append(temp)
        
        df = pd.concat(dfSheets, axis=1)
        
        # Remove duplicate index
        df = df[~df.index.duplicated(keep='first')]
        
        # Drop nan rows
        df.fillna(value=np.nan, inplace=True)
        df.dropna(how='all', inplace=True)
        
        # Rename
        df.rename(columns=rnm, inplace=True)
        
        # Drop duplicate columns
        df = df.loc[:,~df.columns.duplicated()]
        
        # Additional posts
        df['opInc'] = df['grsPrft'] - df['opEx']
        
        df['intCvg'] = (df['opInc'] / df['intEx'])
        df['taxRate'] = df['taxEx'] / df['ebt'] 
        
        df['wrkCap'] = (
            (df['cce'] + df['stInv'] + 0.75 * df['rcv'] +
             0.5 * df['ivty']).rolling(2, min_periods=0).mean() - 
            df['totCrtLbt'].rolling(2, min_periods=0).mean())
        
        # Tangible equity
        df['tgbEqt'] = (df['totEqt'] - df['prfEqt'].fillna(0) - 
            df['gw'].fillna(0) - df['itgbAst'].fillna(0))
        
        # Total non-current assets
        df['totNoCrtAst'] = (df['ppe'].fillna(0) + 
            df['ltInv'].fillna(0) + df['itgbAst'].fillna(0) +
            df['othNoCrtAst'].fillna(0))
        
        # Total non-current liabilities
        df['totNoCrtLbt'] = (df['ltDbt'].fillna(0) + 
            df['noCrtDefTaxLbt'].fillna(0) + 
            df['provRiskChrg'].fillna(0) +
            df['othNoCrtLbt'].fillna(0))
        
        # Total debt
        df['totDbt'] = (df['stDbt'].fillna(0) + df['ltDbt'].fillna(0))
        
        # Drop nan columns
        posts = [
            'sgaEx', 'rdEx', 'stInv', 'ivty', 'ltInv', 'gw', 
            'itgbAst', 'stDbt', 'ltCapLeas', 'prfEqt', 'dvd']
        for p in posts:
            df[p] = df[p].fillna(0)
        
        df.dropna(axis=1, how='all', inplace=True)
        
        return df

def getTickers():
    
    def parse(tbl, country):
        
        scrap = []
        for tr in tbl.findAll('tr')[1:]:
            td = tr.findAll('td')
            
            scrap.append({
                'name': td[0].find('span').text,
                'ticker': td[0].find('a').get('href').split('/')[-1],
                'sector': td[2].text,
                'exchange': td[1].text,
                'country': country,
            }) 
            
        return scrap
    
    # Get country list
    url = 'https://www.wsj.com/market-data/quotes/company-list'
    
    with requests.Session() as s:
        rc = s.get(url, headers=Ticker._headers)
        soup = bs.BeautifulSoup(rc.text, 'lxml')
        
    #ulCountry = soup.find('ul', {'class': 'cl-list'})
    
    ulCountry = soup.find(lambda x:
        x.name == 'ul' and
        'cl-list' in x.get('class', []) and
        not 'cl-tree' in x['class']
    )
    
    dctCountry = {}
    for li in ulCountry.findAll('li'):
        
        #if 'break' not in li['class']:
        if li.find('a') is not None:
            key = li.find('a').text # Country
            val = li.find('a').get('href')
            dctCountry.update({key: val})
    
    # Get tickers
    scrap = []
    for k in dctCountry:
        url = dctCountry[k] 
        
        with requests.Session() as s:
            rc = s.get(url, headers=Ticker._headers)
            soup = bs.BeautifulSoup(rc.text, 'lxml')
            
        # Scrap first page
        tbl = soup.find('table', {'class': 'cl-table'})
        scrap += parse(tbl, k)
        
        # Find pagination
        pg = soup.find('ul', {'class': 'cl-pagination'})
        numPages = pg.findAll('li')
        
        if numPages:
            lastPage = numPages[-2].find('a').text
        
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
    return df
        
