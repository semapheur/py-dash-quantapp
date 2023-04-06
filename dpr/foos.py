# -*- coding: utf-8 -*-

#%% Libraries

# Data analysis
from bs4.element import nonwhitespace_re
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.stats import linregress, boxcox, norm
from scipy.special import inv_boxcox
from scipy.fftpack import fft, ifft
from scipy.optimize import root_scalar, brentq #fixed_point

# Data optimization
import cvxpy as cp

# Machine learning
from sklearn.preprocessing import StandardScaler # StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler
from sklearn import metrics, preprocessing, neighbors, svm # StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler
from sklearn.linear_model import SGDRegressor # LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
#from fastai.tabular import add_datepart # Date features

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM # CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam # RMSprop

# Statistical analysis
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, Holt
# from statsmodels.tsa.vector_ar import VECM
#from statsmodels.tsa.stattools import grangercausalitytest # adfuller, kpss, STL
from arch import arch_model
#from pmdarima.arima import auto_arima
from fbprophet import Prophet
from hmmlearn.hmm import GaussianHMM
from pykalman import KalmanFilter

# Financial analysis
#import talib

# Scrapping
import bs4 as bs
import requests
import requests_cache

# API
import quandl
#import pandas_datareader.data as web
#from yahoofinancials import YahooFinancials
import yfinance as yf
import json # print(json.dumps(json_object, indent=4)) 
from jsonpath_ng import parse
#from newsapi import NewsApiClient
from pytrends.request import TrendReq # Google Trends
import world_bank_data as wb

# Utils
import re # Lookahead (pos/neg): ?= / ?! | Lookbehind (pos/neg): ?<= / ?<!
#import types
from tqdm import tqdm
#import string
import textwrap
from functools import reduce
#import itertools
from collections import deque, Counter
from pathlib import Path
import os
#import pycountry
#from copy import deepcopy

#from glom import glom

# Database
import sqlalchemy as sqla #import create_engine #, MetaData, Table, Column, Integer, Text, Float
#import sqlite3
from neo4j import GraphDatabase
from pymongo import MongoClient, InsertOne, UpdateOne

# I/O
import io
from unidecode import unidecode

# Object dumping

# File explorer
#import os
import glob

# Date manipulation
from datetime import datetime as dt
from datetime import timezone as tz
from dateutil.relativedelta import relativedelta
import time

# Random values
#import random

# Math
from findiff import FinDiff

# Fuzzy matching
from fuzzywuzzy import fuzz, process

# Local
#import morningstar as mstar
from lib.yahoo_finance import Ticker as yhTicker
from lib.fred import Ticker as fredTicker
#import investing
#import marketwatch as mw

# FMP API key: 35220880cad76b7452c77c67fc91f9a5

#%% Helper functions

def tblType(x):
        if x == np.object:
            result = 'text'
        elif x == np.float64:
            result = 'numeric'
        else:
            result = 'any'
        return result

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def splitFilterPart(filterPart):
    for o in operators:
        for i in o:
            if i in filterPart:
                namePart, valuePart = filterPart.split(i, 1)
                name = namePart[namePart.find('{') + 1: namePart.rfind('}')]

                valuePart = valuePart.strip()
                v0 = valuePart[0]
                if (v0 == valuePart[-1] and v0 in ("'", '"', '`')):
                    value = valuePart[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(valuePart)
                    except ValueError:
                        value = valuePart

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, o[0].strip(), value

    return [None] * 3

# Rename duplicate DataFrame columns
class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return '%s_%d' % (x, self.d[x])

def fuzzMatch(left, right):
    
    matches = [] #np.full(len(left), '')
    ratio = [] #np.zeros(len(left)) #[]
    for i, l in enumerate(left):

        if l in right:
            matches.append(i) #matches[i] = r
            ratio.append(100) #ratio[i] = 100
        else:
            x = process.extractOne(
                l, right, 
                scorer=fuzz.token_set_ratio,
                score_cutoff=70
            )
            if x is not None:
                matches.append(x[0])
                ratio.append(x[1])
                
            else:
                matches.append('')
                ratio.append(np.nan)

        df = pd.DataFrame(
            list(zip(left, matches, ratio)),
            columns=['original', 'match', 'ratio']
        )
        df.dropna(inplace=True)
            
    return df

def replaceAll(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def unixToDate(ts, ms=True):
        if ms:
            ts /= 1000
        return dt.utcfromtimestamp(ts).date()

def dateToUnix(strDate):
    try:
        if isinstance(strDate, str):
            date = dt.strptime(strDate, '%Y-%m-%d')
        elif isinstance(strDate,  dt):
            date = strDate
        
        unix = int(dt(date.year, date.month, date.day, 0, 0).replace(tzinfo=dt.timezone.utc).timestamp())
        return unix
    except Exception as e:
        print('Invalid date input given: ', str(e))

def lastBusinessday(inDate=None):
    
    if inDate is None:
        inTime = dt.min.time()
        inDate = dt.combine(dt.now(), inTime)
        
    if inDate.weekday() == 6: # Sunday
        diff = -2
        
    elif inDate.weekday() == 5: # Saturday
        diff = -1
    
    else:
        diff = 0
    
    return inDate + relativedelta(days=diff)

def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    
    return df

def rgbToHex(rgb):
    return '#' + ''.join([f'{c:02x}' for c in rgb])

def rgb(mag, cmin, cmax):
    ''' Return a tuple of integers for R, G and B. '''
    r, g, b = rgbFloat(mag, cmin, cmax)
    return int(r * 255), int(g * 255), int(b * 255)

def rgbFloat(mag, cmin, cmax):
    ''' Return a tuple of floats between 0 and 1 for R, G and B. '''
    
    # Normalize
    try: x = float(mag - cmin) / (cmax - cmin)
    except ZeroDivisionError: x = 0.5 # cmax == cmin
    
    r = min((max((4 * (x - 0.25), 0.)), 1.))
    g = min((max((4 * np.fabs(x - 0.5) - 1., 0.)), 1.))   
    b = min((max((4 * (0.75 - x), 0.)), 1.))
    
    return r, g, b

def updateDict(dct, key, value):
    
    for k, v in dct.items():
        
        if re.search(key, k, re.I):
            
            if callable(value):
                dct[k] = value(dct[k])

            else:
                dct[k] = value
                
        elif isinstance(v, dict):
            updateDict(v, key, value)
            
        elif isinstance(v, list):
            
            for i in v:
                if isinstance(i, dict):
                    updateDict(i, key, value)


#%% Database

#conn = Neo4jConnection(uri='bolt://localhost:7687', user='neo4j', pwd='gH31m3p')
class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd))
            
        except Exception as e:
            print('Failed to create the driver: ', e)
            
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, 'Driver not initialized!'
        
        session = None
        response = None
        
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
            
        except Exception as e:
            print('Query failed: ', e)
            
        finally:
            if session is not None:
                session.close()
                
        return response 

def insertNeo4jData(conn, query, rows, batchSize = 10000):
    # Function to handle the updating the Neo4j database in batch mode.
    
    # Constraint to speed up node creation
    conn.query('CREATE CONSTRAINT stock IF NOT EXISTS ON (s:security:stock) ASSERT s.id IS UNIQUE')
    
    total = 0
    batch = 0
    #start = time.time()
    result = None
    
    while batch * batchSize < len(rows):

        res = conn.query(query, 
                         parameters = {'rows': rows[batch*batchSize:(batch+1)*batchSize].to_dict('records')})
        total += res[0]['total']
        batch += 1
        result = {'total':total, 
                  'batches':batch, 
                  #"time":time.time()-start
                 }
        print(result)
        
    return result
#%% Option trading
class BlackScholes:
    
    def __init__(self, S, K, T, sigma, r=.01, q=0):
        self.S = S # Spot price
        self.K = K # Strike price
        self.T = T # Time to maturity
        self.r = r # Market free rate
        self.sigma = sigma # Implied volatility
        self.q = q # Dividend compound rate
        
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {
            'S': self.S,
            'K': self.K,
            'r': self.r,
            'sigma': self.sigma,
            'q': self.q
        }
    
    def d1(self):
        return (
            (np.log(self.S / self.K) + 
            (self.r - self.q + self.sigma**2 / 2) * self.T) /
            (self.sigma * np.sqrt(self.T))
        )
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def _callPrice(self):
        return (
            self.S * np.exp(-self.q * self.T) * self.N(self.d1()) -
            self.K * np.exp(-self.r * self.T) * self.N(self.d2())
        )
    
    def _putPrice(self):
        return (
            self.K * np.exp(-self.r * self.T) * self.N(-self.d2()) -
            self.S * np.exp(-self.q * self.T) * self.N(-self.d1())
        )
    
    def price(self, type_='call'):
        
        if type_ == 'call': # Call option
            return self._callPrice()
        
        elif type_ == 'put': # Put option
            return self._putPrice()
        
        elif type_ == 'both':
            return {'call': self._callPrice(), 'put': self._putPrice()}
        
        else:
            raise ValueError('Unrecognized type')
        
    def delta(self, type_='call'):
        
        if type_ == 'call':
            return self.N(self.d1())
        
        elif type_ == 'put':
            return self.N(self.d1()) - 1

def bsPrice(flag, S, K, t, r, sigma, q=0):
    
    def d1(S, K, t, r, q, sigma):
        return (
            (np.log(S / K) + (r - q + sigma**2 / 2) * t) /
            (sigma * np.sqrt(t))
        )
    
    def d2(S, K, t, r, q, sigma):
        return d1(S, K, t, r, q, sigma) - sigma * np.sqrt(t)
    
    D1 = d1(S, K, t, r, q, sigma)
    D2 = d2(S, K, t, r, q, sigma)
    
    if flag == 'call':
        return S * np.exp(-q * t) * norm.cdf(D1) - K * np.exp(-r * t) * norm.cdf(D2)
    
    elif flag == 'put':
        return K * np.exp(-r * t) * norm.cdf(-D2) - S * np.exp(-q * t) * norm.cdf(-D1)

def bsImpliedVolatility(flag, V, S, K, t, r, q=0):
    
    f = lambda sigma: V - bsPrice(flag, S, K, t, sigma, r, q)
    
    return brentq(
        f, 
        a=1e-12, b=100, # bounds
        xtol=1e-15, rtol=1e-15,
        maxiter=1000,
        full_output=False
    )

def yfOptionData(ticker):

    # Yahoo connection
    session = requests_cache.CachedSession('yfinance.cache')
    session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0'
    tick = yf.Ticker(ticker, session=session)

    # Option expiration dates
    optDate = tick.options

    dfs = [None] * len(optDate)

    for i in range(len(optDate)):

        # Option data
        optData = tick.option_chain(optDate[i])

        # Add option type columns
        optData.calls['type'] = 'c'
        optData.puts['type'] = 'p'
        
        # Concatenate call/put data and include expiration date in multicolumn 
        tmp = pd.concat([optData.calls, optData.puts])
        date = dt.strptime(optDate[i], '%Y-%m-%d')
        multiCols = pd.MultiIndex.from_product([[date], tmp.columns])
        tmp.columns = multiCols

        dfs[i] = tmp

    df = pd.concat(dfs, axis=1)

    return df

def ndaqOptionData(ticker):
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Origin': 'https://www.nasdaq.com',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Referer': 'https://www.nasdaq.com/',
        'Sec-GPC': '1',
        'TE': 'Trailers',
    }

    params = (
        ('assetclass', 'stocks'),
        ('limit', '10000'),
        ('fromdate', 'all'),
        ('todate', 'undefined'),
        ('excode', 'oprac'),
        ('callput', 'callput'),
        ('money', 'all'),
        ('type', 'all'),
    )
    
    url = f'https://api.nasdaq.com/api/quote/{ticker}/option-chain'
    
    with requests.Session() as s:
        rs = s.get(url, headers=headers, params=params)
        parse = json.loads(rs.text)
    
    # Get spot price
    pattern = '(?<=\$)\d+(\.\d+)?'

    match = re.search(pattern, parse['data']['lastTrade'])
    spot = float(match.group())
    
    expiry = ''    
    scrap = []
    dfs = []
    
    y = lambda x: np.nan if x == '--' else float(x)
    for r in parse['data']['table']['rows']:
        
        if r.get('expirygroup'):
            expiry = dt.strptime(r.get('expirygroup'), '%B %d, %Y')
            
            if scrap:
                dfs.append(pd.DataFrame.from_records(scrap))
                scrap = []
            
            continue
        
        strike = y(r.get('strike'))
        
        # Calls
        scrap.append({
            'expiry': expiry,
            'strike': strike,
            'openInterest': y(r.get('c_Openinterest')),
            'lastPrice': y(r.get('c_Last')),
            'ask': y(r.get('c_Ask')),
            'bid': y(r.get('c_Bid')),
            'inTheMoney': float(strike <= spot),
            'optionType': 'call'
        })
        
        # Puts
        scrap.append({
            'expiry': expiry,
            'strike':y(r.get('strike')),
            'openInterest': y(r.get('p_Openinterest')),
            'lastPrice': y(r.get('p_Last')),
            'ask': y(r.get('p_Ask')),
            'bid': y(r.get('p_Bid')),
            'inTheMoney': float(strike >= spot),
            'optionType': 'put'
        })
        
    df = pd.concat(dfs)
    
    # Implied volatility
    try:
        bondYield = yf.Ticker('^TNX').history(period='1d')
        riskfreeRate = bondYield['Close'].iloc[-1] / 100
        
    except:
        startDate = dt.strftime(dt.now() - relativedelta(days=3), '%Y-%m-%d')
        bondYield = yhTicker('^TNX').ohlcv(startDate)
        riskfreeRate = bondYield['close'].iloc[-1] / 100
    
    def f(row, spot, rfr):
        
        if row[['lastPrice', 'strike']].isna().any():
            return np.nan
        
        try:
            t = (row['expiry'] - dt.now()).days / 365

            return bsImpliedVolatility(
                row['optionType'], row['lastPrice'], spot, row['strike'], t, rfr)
        
        except:
            return np.nan
        
    df['impliedVolatility'] = df.apply(f, axis=1, args=(spot, riskfreeRate))
    
    # Delta
    def g(row, spot, rfr):
        t = (row['expiry'] - dt.now()).days / 365
        return BlackScholes(spot, row['strike'], t, row['impliedVolatility'],
                            rfr).delta(row['optionType'])
    
    df['delta'] = df.apply(g, axis=1, args=(spot, riskfreeRate))
    
    return df, spot, riskfreeRate

#%% Technical analysis

# Buying pressure
def buyingPressure(df):
    df.sort_values('date', inplace=True)
    df['shift'] = df['close'].shift(periods=1)
    bp = df['close'] - df[['low', 'shift']].min(axis=1)
    return bp

# True range
def trueRange(ohlc):
    ohlc.sort_values('date', inplace=True)
    ohlc['shift'] = ohlc['close'].shift(periods=1)
    tr = ohlc[['high', 'shift']].max(axis=1) - ohlc[['low', 'shift']].min(axis=1)
    return tr

def priceChannel(ohlc, p):
    ohlc = ohlc.tail(p)
    
    dates = pd.DatetimeIndex(ohlc.index)
    ohlc['x'] = (dates.date - dates.date.min()) #.astype('timedelta64[D]')
    ohlc['x'] = ohlc['x'].dates.days + 1
            
    # High trend line
    temp = ohlc.copy()
    
    while len(temp) > 3:
    
        reg = linregress(x=temp['x'], y=temp['high'])
        temp = temp.loc[temp['high'] > reg[0] * temp['x'] + reg[1]]
    
    reg = linregress(x=temp['x'], y=temp['high'])
    
    high = reg[0] * ohlc['x'] + reg[1]
    
    # Low trend line
    temp = ohlc.copy()
    while len(temp) > 3:
        reg = linregress(x=temp['x'], y=temp['low'])
        temp = temp.loc[temp['low'] < reg[0] * temp['x'] + reg[1]]

    reg = linregress(x=temp['x'], y=temp['low'])
    low = reg[0] * ohlc['x'] + reg[1]
        
    return low, high

def fibonacciLevels(close, p, trend='u'):
    # Trend: u/d
    close = close.tail(p)
    priceDiff = close.max() - close.min()
    
    if trend == 'u':
        anchor = close.min()
    elif trend == 'd': 
        anchor = close.max()
        priceDiff = -priceDiff
    
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    fl = [anchor + r * priceDiff for r in ratios]
        
    return fl

# Moving average

# Simple Moving Average
def sma(x, p):
    # vals: Pandas Series
    sma = x.rolling(window=p, min_periods=0).mean()
    return sma

# Exponential Moving Average
def ema(x, p):
    # x: Pandas Series
    ema = x.ewm(span=p, min_periods=0, adjust=False).mean()
    return ema
    
# Smoothed Moving Average
def smma(x, p):
    alpha = 1/p
    smma = x.ewm(alpha=alpha, min_periods=0, adjust=False).mean()
    return smma

# Zero Lag Exponential Moving Average
def zlema(x, p):
    # x: Pandas Series
    lag = int((p - 1) / 2)
    data = x + x.diff(lag) # (x - x.shift(lag))
    zlema = data.ewm(span=p, min_periods=0, adjust=False).mean()
    return zlema

# Hull Moving Average
def hma(x, p):
    temp = 2*zlema(x, int(p/2)) - zlema(x, p)    
    hull = zlema(temp, int(np.sqrt(p)))
    return hull

# Kaufman's Adaptive Moving Average
def kama(x, n=10, pow1=2, pow2=30):
    '''
    x: Pandas Series
    
    er: Efficiency Ratio
    sc: Smoothing Constant
    '''
    
    erNom = x.diff(n).abs()
    erDen = x.diff().abs().rolling(n).sum()
    er = erNom / erDen

    sc = ( er*( 2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1) ) ** 2

    answer = np.zeros(sc.size)
    N = len(answer)
    first_value = True

    for i in range(N):
        if sc[i] != sc[i]:
            answer[i] = np.nan
        else:
            if first_value:
                answer[i] = x[i]
                first_value = False
            else:
                answer[i] = answer[i-1] + sc[i] * (x[i] - answer[i-1])
    return pd.Series(answer, index=x.index)
    
# Fractal Adaptive Moving Average
def frama(x, p=20, fc=1, sc=200):
    # x: Pandas Series
    
    frama = x
    w = np.log(2 / (sc + 1))
    
    for i in range(p, len(x)):
        # Split in 2 batches
        b1 = x[i-p:i-int(p/2)]
        b2 = x[i-int(p/2):i]
        
        # 1st batch
        h1 = b1.max()
        l1 = b1.min()
        n1 = (h1 - l1) / (p/2)
        
        # 2nd batch
        h2 = b2.max()
        l2 = b2.min()
        n2 = (h2 - l2) / (p/2)
        
        # n3
        h = np.max([h1, h2])
        l = np.min([l1, l2])
        n3 = (h - l) / p
        
        # Fractal dimension
        dim = 0
        if n1 > 0 and n2 > 0 and n3 > 0:
            dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        
        alpha = np.exp(w*(dim - 1))
        alpha = np.max([alpha, 0.01])
        alpha = np.min([alpha, 1])
        
        oN = (2 - alpha) / alpha
        n = (((sc - fc)*(oN - 1))/(sc - 1)) + fc
        alpha = 2 / (n + 1)
        
        alpha = np.max([alpha, 2 / (sc + 1)])
        alpha = np.min([alpha, 1])
        
        frama[i] = alpha * x[i] + (1 - alpha) * frama[i-1]
        
    return frama

# Moving average convergence/divergence 
def MACD(close, pS, pL):

    macd = zlema(close, pS) - zlema(close, pL)
    signal = ema(macd, 20)
    
    '''
    if norm:
        macd = normalize(macd)
        signal = normalize(signal)
    '''
    
    return macd, signal

# Accumulation/distribution index
def ADI(ohlcv):
    n = ((ohlcv['close'] - ohlcv['low']) - (ohlcv['high'] - ohlcv['close']))/(ohlcv['high'] - ohlcv['low'])
    m = n * ohlcv['volume']    
    adi = m.cumsum()
        
    return adi    

# Chaikin oscillator
def CO(ohlcv, pS, pL):
    co = zlema(ADI(ohlcv), pS) - zlema(ADI(ohlcv), pL)
    return co
    
# Ease of movement
def EMV(ohlcv, p):
    pm = (ohlcv['high'] + ohlcv['low'])/2 - (ohlcv['high'].shift(1) + ohlcv['low'].shift(1))/2 
    br = ohlcv['volume']/(ohlcv['high'] - ohlcv['low'])
    emv = pm/br
    
    return zlema(emv, p)
    
# Force index
def FI(ohlcv, p):
    f = ohlcv['close'].diff() * ohlcv['volume']
    fi = zlema(f, p)
        
    return fi

# Parabolic stop and reverse
def psar(ohlc, iaf = 0.02, maxaf = 0.2):
    psar = ohlc['close']
    
    psarBull = pd.Series([np.nan]*len(ohlc.index), index=ohlc.index) 
    psarBear = psarBull.copy()
    
    bull = True
    af = iaf
    ep = ohlc['low'][0]
    hp = ohlc['high'][0]
    lp = ohlc['low'][0]
    
    for i in range(2,len(psar)):
        if bull:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
        
        reverse = False
        
        if bull:
            if ohlc['low'][i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = ohlc['low'][i]
                af = iaf
        else:
            if ohlc['high'][i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = ohlc['high'][i]
                af = iaf
    
        if not reverse:
            if bull:
                if ohlc['high'][i] > hp:
                    hp = ohlc['high'][i]
                    af = min(af + iaf, maxaf)
                if ohlc['low'][i-1] < psar[i]:
                    psar[i] = ohlc['low'][i-1]
                if ohlc['low'][i-2] < psar[i]:
                    psar[i] = ohlc['low'][i-2]
            else:
                if ohlc['low'][i] < lp:
                    lp = ohlc['low'][i]
                    af = min(af + iaf, maxaf)
                if ohlc['high'][i-1] > psar[i]:
                    psar[i] = ohlc['high'][i-1]
                if ohlc['high'][i-2] > psar[i]:
                    psar[i] = ohlc['high'][i-2]
                    
        if bull:
            psarBull[i] = psar[i]
        else:
            psarBear[i] = psar[i]
    
    return psarBear, psarBull     

# Bollinger Bands
def bollinger(close, p, k=2):
    mBand = zlema(close, p)
    lBand = mBand - k*close.ewm(span=p, min_periods=0).std()
    uBand = mBand + k*close.ewm(span=p, min_periods=0).std()
    
    return lBand, uBand

# Average directional movement index
def ADX(ohlc, p=20):
    tr = trueRange(ohlc)

    pdm = ohlc['high'].diff().copy() # Positive directional movement
    ndm = ohlc['low'].diff(-1).copy() # Negative directional movement

    pdm.where(pdm > 0, 0, inplace=True)
    ndm.where(ndm < 0, 0, inplace=True)

    pdm.where(pdm >= ndm, 0, inplace=True)
    ndm.where(ndm <= pdm, 0, inplace=True)

    pdi = zlema(x=(pdm / tr), p=p) # Positive directional index
    ndi = zlema(x=(ndm / tr), p=p) # Negative directional index
    

    dx = (pdi - ndi) / (pdi + ndi)
    dx.abs()
    adx = zlema(dx, p)
    return adx

# Commodity channel index
def CCI(ohlc, p=20):
    tp = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3 # True price
    cci = (1 / 0.015) * (tp - tp.rolling(p).mean()) / tp.rolling(p).std()
    return cci / 100

# Mass index
def MI(ohlc):
    pr = ohlc['high'] - ohlc['low'] # Price range
    ma1 = zlema(pr, 10)
    ma2 = zlema(ma1, 10)
    #ema1 = pr.ewm(span=10).mean()
    #ema2 = ema1.ewm(span=10).mean()
    mass = ma1 / ma2
    mi = mass.rolling(30).sum()
    return mi

# Relative strength index
def RSI(close, p=20):
    # Takes Pandas Series with closing prices
    um = close.diff().copy()
    dm = close.diff().copy()
    um.where(um > 0, 0, inplace=True) # Upward movement
    dm.where(dm < 0, 0, inplace=True) # Downward movement
    dm = dm.abs()
    rs = zlema(um, p) / zlema(dm, p)
    #rs = smma(um, p) / smma(dm, p)
    rsi = 1 - (1 / (1 + rs))
            
    return rsi
# Stochastic RSI

# Money flow index
def MFI(ohlcv, p):
    tp = (ohlcv['high'] + ohlcv['low'] + ohlcv['close'])/3 # typical price
    mf = tp * ohlcv['volume'] # money flow
    
    pmf = mf.diff().copy()
    nmf = mf.diff().copy()
    
    pmf.where(pmf > 0, 0, inplace=True)
    nmf.where(nmf < 0, 0, inplace=True)
    
    mr = pmf.rolling(window=p, min_periods=0).sum() / nmf.rolling(window=p, min_periods=0).sum()
    
    mfi = 1 - (1 / (1 + mr))
    
    return mfi

# Stochastic oscillator
def SO(ohlc, p=20):
    k = (ohlc['close'] - ohlc['low'].rolling(p).min()) / (ohlc['high'].rolling(p).max() - ohlc['low'].rolling(p).min())
    d = zlema(k, 5)
    dSlow = zlema(d, 5)
    
    return k, d, dSlow

# Triple exponential
def TRIX(close, p=20):
    ma1 = zlema(close, p)
    ma2 = zlema(ma1, p)
    ma3 = zlema(ma2, p)

    trix = (ma3 / ma3.shift(1)) - 1
    return trix

# True strength index
def TSI(close, r=30, s=15):
    m = close.diff() # movement
    absM = m.abs()
    ma1 = zlema(m, r)
    absMa1 = zlema(absM, r)
    ma2 = zlema(ma1, s)
    absMa2 = zlema(absMa1, s)
    result = ma2 / absMa2
    return result

# Ultimate oscillator
def UO(ohlc, p1=5, p2=20, p3=60):
    bp = buyingPressure(ohlc)
    tr = trueRange(ohlc)
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    uo = ((p3 / p1) * avg1 + (p3/p2) * avg2 + avg3) / ((p3 / p1) + (p3 / p2) + 1)
    return uo

# Vortex indicator
def VI(ohlc, p=20):
    tr = trueRange(ohlc)

    VMp = np.abs(ohlc['high'] - ohlc['low'].shift(-1))
    VMn = np.abs(ohlc['low'] - ohlc['high'].shift(-1))

    VIp = VMp.rolling(p).sum() / tr.rolling(p).sum()
    VIn = VMn.rolling(p).sum() / tr.rolling(p).sum()
    return VIp, VIn

# Donchian channel
def DC(ohlc, p=20):
    
    # Lower channel
    lc = ohlc['low'].rolling(p).min()
    
    # Upper channel
    uc = ohlc['high'].rolling(p).max()
    
    # Middle channel
    mc = (uc - lc) / 2
    
    return lc, uc, mc

# Keltner channel
def KC(ohlc, p, k=2):
    
    # Typical price
    tp = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
    
    # Average true range
    atr = zlema(trueRange(ohlc), p)
    
    # Middle line 
    ml = zlema(tp, p)
    
    # Lower channel
    lc = ml - k * atr
     
    # Upper channel
    uc = ml + k * atr
    
    return lc, uc, ml

# Pivot points
def pivotPoint(x, isMin):
    
    dx = 1 #1 day interval
    dDx = FinDiff(0, dx, 1)
    d2Dx2 = FinDiff(0, dx, 2)
    clarr = np.asarray(x)
    mom = dDx(clarr)
    momacc = d2Dx2(clarr)
    
    # Local maxima
    pivMax = [i for i in range(len(mom))
        if momacc[i] < 0 and
        (
            mom[i] == 0 or # Slope is 0
            (
                i != len(mom) - 1 and # Check next day
                (
                    mom[i] > 0 and mom[i+1] < 0 and
                    clarr[i] >= clarr[i+1]
                ) or
                i != 0 and # Check prior day
                (
                    mom[i-1] > 0 and mom[i] < 0 and
                    clarr[i-1] < clarr[i] 
                )
            )             
        )
    ]
    
    # Local minima
    pivMin = [i for i in range(len(mom))
        if momacc[i] > 0 and
        (
            mom[i] == 0 or # Slope is 0
            (
                i != len(mom) - 1 and # Check next day
                (
                    mom[i] < 0 and mom[i+1] > 0 and
                    clarr[i] <= clarr[i+1] # Minima
                ) or
                i != 0 and # Check prior day
                (

                    mom[i-1] < 0 and mom[i] > 0 and
                    clarr[i-1] > clarr[i]
                )
            )             
        )         
    ]
    
    return pivMax, pivMin

#%% Statistical analysis
# Gemetric Brownian Motion
def gbm(close, steps, arch=True):
    
    ret = close.pct_change().dropna()
    sim = [close[-1]]
    # if arch:
    #     arimaMdl = auto_arima(ret.values*100, 
    #         start_p=1, start_q=1,
    #         max_p=3, max_q=3, # maximum p and q
    #         test='kpss', # adf
    #         d=None, # let model determine 'd'
    #         seasonal=False, # For SARIMA set True 
    #         trace=False,
    #         scoring='mse',
    #         error_action='ignore',  
    #         suppress_warnings=True, 
    #         stepwise=True,
    #     )
        
    #     order = arimaMdl.get_params()['order']
        
    #     am = arch_model(ret*100, vol='Garch', p=order[0], o=0, q=order[-1], dist='t', mean='AR')
    #     result = am.fit(disp='off') #update_freq=5
        
    #     fc = result.forecast(horizon=steps, start=None, align='origin')
    #     mu = fc.mean.iloc[-1, 0] / 100
        
    #     var = fc.variance.iloc[-1] / 100
        
    #     for i in range(steps):
    #         nxt = sim[-1] * np.exp((mu - (var[i]**2) / 2) + var[i] * np.random.normal())
    #         sim.append(nxt)
    # else:
    mu = ret.mean()
    sigma = ret.std()
    
    for i in range(steps):
        nxt = sim[-1] * np.exp((mu - (sigma**2) / 2) + sigma * np.random.normal())
        sim.append(nxt)
    
    simDates = pd.bdate_range(start=close.index[-1] + relativedelta(days=1), periods=steps)
    y = pd.Series(sim[1:], index=simDates)
    
    return y
# Machine Learning
def mlReg(data, fcPeriod, algo, rec=True, valSplit=0.1):
    if rec:
        shft = 1
    else:
        shft = fcPeriod
    
    # Features
    data = techFeatures(data, rec)
    
    # External features
    path = 'data/Features.csv'
    startDate = data.index[0] - relativedelta(days=5)
    if os.path.exists(path):
        featDate = dt.utcfromtimestamp(os.path.getmtime(path)).date()
        if featDate < dt.now():
            features = macroFeatures(startDate, True)
        else:
            features = pd.read_csv('data/Features.csv', index_col=0, parse_dates=True)
    else:
        features = macroFeatures(startDate, True)

    data = pd.merge(data, features, on='Date', how='left')
    
    # Preprocess
    data.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
    
    if rec:
        data.drop('Volume', axis=1, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    
    # Labels
    data['Label'] = data['Close'].pct_change().shift(-shft)
    
    X = data.drop('Label', axis=1).to_numpy()
    y = data['Label'][:-shft].to_numpy()
    
    # Normalization
    scaleF = preprocessing.StandardScaler() # MinMaxScaler(feature_range=(0, 1)) scaler.inverse_transform()
    X = scaleF.fit_transform(X.reshape(len(X), 1))
    
    # Train/test split
    xTrain, xTest, yTrain, yTest = train_test_split(X[:-shft], y, test_size=valSplit)
    
    if algo == 'LinReg': # Linear models
        model = SGDRegressor() # LinearRegression/SGDRegressor/Ridge/ElasticNet/Lasso
    elif algo == 'SVM': # Support Vector Machine
        svr = svm.SVR() # kernel: linear (degree=3)/poly/rbf/sigmoid | gamma = auto/scale
        params = {
            'kernel': ['linear', 'rbf'],
            'gamma': ['auto', 'scale'],
            'C': [0.1,1,10]
        }
        model = GridSearchCV(svr, params, cv=5)
    elif algo == 'kNN': # k Nearest Neighbours
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)
    elif algo == 'XGB': # Extreme Gradient Boosting
        model = XGBRegressor(objective='reg:squarederror',
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            min_child_weight=15,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            gamma=0,
        )
    
    # Fit model
    model.fit(xTrain, yTrain.ravel())
    
    # Accuracy
    yTestHat = model.predict(xTest).flatten()
    mse = metrics.mean_squared_error(yTest, yTestHat)
    rmse = np.sqrt(mse)
    
    # Forecast
    
    if rec:
        fcFeat = var(features, fcPeriod, 10)
        yL = yU = [data['close'][-1]]
        
        for i in range(1, fcPeriod):
            # Forecast sequences        
            fcX = X[-1].reshape(1, -1)

            # Predict
            fcY = model.predict(fcX).flatten()[-1]
            
            # Confidence interval
            yL.append(((fcY - i * rmse) + 1) * yL[-1])
            yU.append(((fcY + i * rmse) + 1) * yU[-1])
            
            fcDate = pd.bdate_range(start=data.index[-1] + relativedelta(days=1), periods=1)
            
            data.loc[fcDate[0], 'close'] = data['close'][-1] * (1 + fcY)
            data.loc[fcDate[0], [c for c in fcFeat.columns]] = fcFeat.loc[fcDate[0], [c for c in fcFeat.columns]]

            data = techFeatures(data, rec)

            # Rescale
            X = data.drop('label', axis=1).to_numpy()
            X = scaleF.fit_transform(X)
        yHat = data['close'][-fcPeriod:]
    else:
        fcX = X[-shft:]
        fcY = model.predict(fcX).flatten()
        fcDate = pd.bdate_range(start=data.index[-1] + relativedelta(days=1), periods=fcPeriod)
        
        # Confidence interval
        yL = [y - i * rmse for i, y in enumerate(fcY, start=1)]
        yU = [y + i * rmse for i, y in enumerate(fcY, start=1)]
        
        fcY[0] = (fcY[0] + 1) * data['close'][-1]
        yL[0] = (yL[0] + 1) * data['close'][-1]
        yU[0] = (yU[0] + 1) * data['close'][-1]
        for i in range(1, fcPeriod):
            fcY[i] = (fcY[i] + 1) * fcY[i-1]
            yL[i] = (yL[i] + 1) * yL[i-1]
            yU[i] = (yU[i] + 1) * yU[i-1]
            
        yHat = pd.Series(fcY, index=fcDate)
    
    # Accuracy interval
    yL = pd.Series(yL, index=yHat.index)
    yU = pd.Series(yU, index=yHat.index)
    
    return yHat, yL, yU

# LSTM model creator
def lstmMdl(inShp, hl=(128, 128), dl=(32,1), 
            act='elu', keInit='he_uniform'):
    '''
    Parameters
    ----------
    inShp : tuple
        (time_steps, features)
    hl : tuple, optional
        Neurons in hidden layers. The default is (128, 128).
    dl : tuple, optional
        Neurons in dense layers. The default is (32,1).
    act : str, optional
        Activation function. The default is 'elu'.
    keInit : str, optional
        Kernel initializer. The default is 'he_uniform'.
        
        Kernel initializers:
            random_normal/uniform
            lecun_normal/uniform
            glorot_normal/uniform
            he_normal/uniform
            truncated_normal
            orthogonal
            identity

    Returns
    -------
    model : Sequential
    '''

    model = Sequential()
    
    # 1st layer
    model.add(
        LSTM(
            units=hl[0],
            input_shape=inShp,  # input_shape=(time_steps, features)
            return_sequences=True if len(hl) > 1 else False
        )
    )    
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    # Hidden layers
    if len(hl) > 1:
        for i in range(1, len(hl)-1):
            rs = True if i < len(hl)-1 else False
            model.add(LSTM(hl[i], return_sequences=rs))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
    
    # Dense layers
    for d in dl:
        model.add(Dense(units=d, kernel_initializer=keInit, activation=act))
    
    return model

# LSTM
def lstm(data, fcPeriod, seqLen=60, valSplit=0.1, batch=100, epoch=10, 
         trend=False, valid=False):
    shft = 1
    
    # Preprocess
    data.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    
    if trend:
        data['hpf'] = hpf(data['close'] , l=1600)[1]
        data.drop('close', axis=1, inplace=True)
        data['label'] = data['hpf'].shift(-1)
    else:
        data['label'] = data['close'].shift(-shft)
        
    x = data.drop('label', axis=1).to_numpy()
    y = data.loc[:-shft, 'label'].to_numpy()
        
    # Normalize data
    scaleF = preprocessing.StandardScaler() # MinMaxScaler(feature_range=(0, 1)) scaler.inverse_transform()
    scaleL = preprocessing.StandardScaler() # MinMaxScaler(feature_range=(0, 1)) scaler.inverse_transform()
    
    # Features
    x = scaleF.fit_transform(x.reshape(len(x), 1))
    
    # Labels
    y = scaleL.fit_transform(y.reshape(len(y), 1))
    
    # Create sequences
    X = []
    seq = deque(maxlen=seqLen)
    for i in x[:-shft]:
        seq.append(i)
        if len(seq) == seqLen:
            X.append(np.array(seq))

    X = np.array(X)
    y = y[seqLen-1:]
    
    # Train/test split
    splitIndex = int(valSplit*X.shape[0])
    xTrain = X[:-splitIndex]
    xTest = X[-splitIndex:]
    yTrain = y[:-splitIndex]
    yTest = y[-splitIndex:]

    # LSTM model
    model = lstmMdl(xTrain.shape[1:], (128, 64), (32, 1))
            
    # Optimizer
    opt = Adam(lr=0.001, decay=1e-6)#RMSprop(learning_rate=0.001, rho=0.9)

    model.compile(loss='mse', #  mean_squared_error/sparse_categorical_crossentropy
        optimizer=opt,
        metrics=['mse'])
    
    # Fit model
    history = model.fit(
        xTrain, yTrain,
        batch_size=batch,
        epochs=epoch,
        validation_data=(xTest, yTest), #validation_split=valSplit,
    )
            
    # Validation
    # valid = None
    # if valid:
    #     validClose = model.predict(X)
        
    #     # Flatten list
    #     validflat = [item for sublist in validClose for item in sublist]
    #     validflat = scaleL.inverse_transform(validflatten)
        
    #     validDates = data.index[seqLen:] 
    #     valid = pd.Series(validflat, index=validDates)
        
    # Forecast
    score = model.evaluate(xTest, yTest, verbose=0)
    rmse = np.sqrt(score[0])
    yL, yU = []
    for i in range(fcPeriod):
        # Forecast sequences        
        fcX = x[-seqLen:]
        
        # Predict
        fcY = model.predict(fcX).flatten()
        #fcFlat = [item for sublist in fcY for item in sublist]
        
        yL.append(fcY - rmse * (i + 1))
        yU.append(fcY + rmse * (i + 1))
        
        fcY = scaleL.inverse_transform(fcY)
        fcDate = pd.bdate_range(start=data.index[-1] + relativedelta(days=1), periods=1)
        data.loc[fcDate[0], 'close'] = fcY

        # Rescale
        x = data.drop('label', axis=1).to_numpy()
        x = scaleF.fit_transform(x)
        
    yHat = data.loc[-fcPeriod:, 'close']
    
    # Accuracy interval
    yL = scaleL.inverse_transform(yL)
    yU = scaleL.inverse_transform(yU)
    
    yL = pd.Series(yL, index=yHat.index)
    yU = pd.Series(yU, index=yHat.index)
            
    return yHat, yL, yU
    
# Trained model forecast
def modelPredict(model, data, features, X, p, recurs=True, sims=1):
    scaleF = preprocessing.StandardScaler()
    if recurs:
        yList = []
        for s in range(1, sims):
            # Simulation
            featSims = pd.DataFrame(columns=features.columns)
            for c in features.columns:
                featSims[c] = gbm(features[c], p, arch=False)
            
            temp = data.copy()
            xTemp = X.copy()
            for i in range(p):
                # Forecast sequences        
                fcX = xTemp[-1].reshape(1, -1)

                # Predict
                fcY = model.predict(fcX).flatten()[-1]
                fcDate = pd.bdate_range(start=data.index[-1] + relativedelta(days=1), periods=1)

                #yL.append(fcY - rmse * (i + 1))
                #yU.append(fcY + rmse * (i + 1))

                #fcY = scaleL.inverse_transform([fcY])

                temp.loc[fcDate[0], 'close'] = temp['close'][-1] * (1 + fcY)
                temp.loc[fcDate[0], [c for c in featSims.columns]] = featSims.loc[fcDate[0], [c for c in featSims.columns]]
                temp.append(featSims.loc[fcDate[0]], sort=True)

                temp = techFeatures(temp, recurs)

                # Rescale
                xTemp = temp.drop('label', axis=1).to_numpy()
                xTemp = scaleF.fit_transform(xTemp)
            cols = ['close'] + [c for c in features.columns]
            mCols = pd.MultiIndex.from_product([[f's{s}'], cols])
            yTemp = temp[cols][-p:]
            yTemp.columns = mCols
            yList.append(yTemp)
            
        yHat = reduce(lambda x, y: pd.merge(x, y, on = 'date'), yList)
        yHat.to_csv('data/tst.csv', encoding='utf-8')
    else:
        fcX = X[-p:]
        fcY = model.predict(fcX).flatten()
        #fcY = scaleL.inverse_transform([fcY])
        fcDate = pd.bdate_range(start=data.index[-1] + relativedelta(days=1), periods=p)
        
        fcY[0] = (fcY[0] + 1)*data['close'][-1]
        for i in range(1, len(fcY)):
            fcY[i] = (fcY[i] + 1)*fcY[i-1]
        
        yHat = pd.Series(fcY, index=fcDate)
        #yL = yHat.apply(lambda x: x - (rmse * i)) for i in range(0, shft))
        
    # Accuracy interval
    #yL = scaleL.inverse_transform(yL)
    #yU = scaleL.inverse_transform(yU)
    
    #yL = pd.Series(yL, index=yHat.index)
    #yU = pd.Series(yU, index=yHat.index)
    
    return yHat


# Time Series Analysis
# Holt's Linear Trend
def hlt(x, fcPeriod, alfa=0.8, beta=0.2):
    model = Holt(x.to_numpy(), exponential=True, damped=False).fit(
        smoothing_level=alfa, 
        smoothing_slope=beta, 
        optimized=True
    )
    fcX = model.forecast(fcPeriod)
    fcDates = pd.bdate_range(start=x.index[-1] + relativedelta(days=1), 
                             periods=fcPeriod)
    fc = pd.Series(fcX, index=fcDates)
    return fc

# Vectorized Autoregression
def var(x, fcPeriod, pMax):
    model = VAR(x.to_numpy())
    p = model.select_order(pMax)
    fit = model.fit(p.selected_orders['aic'])
    #irf = fit.irf(fcPeriod)
    #irf.plot(orth = False)
    
    fc = fit.forecast(y=x.iloc[-fit.k_ar:].to_numpy(), steps=fcPeriod)
    fcDates = pd.bdate_range(start=x.index[-1] + relativedelta(days=1), periods=fcPeriod)
    yHat = pd.DataFrame(fc, index=fcDates, columns=x.columns)
    return yHat

# Hidden Markov Model
def hmm(x, roc=True, n=2):
    # x: returns (DataFrame/Series)
    if not roc:
        x = x.pct_change().dropna()
    
    x = x.to_numpy().reshape(-1, 1)
    
    model = GaussianHMM(
        n_components=n, covariance_type='full', n_iter=1000
    ).fit(x)
    
    # Predict hidden states
    hiddenStates = model.predict(x)
    return hiddenStates

# Auto ARIMA
# def autoArima(x, fcPeriod, s=False, D=0):
#     model = auto_arima(x.to_numpy(), start_p=1, start_q=1,
#         test='kpss', # adf
#         max_p=5, max_q=5, # maximum p and q
#         m=1, # 7: daily, 52: weekly, 12: monthly, 1: annual
#         d=None, # let model determine 'd'
#         seasonal=s, # For SARIMA set True 
#         start_P=0, 
#         D=D, # For SARIMA set D=1
#         trace=True,
#         scoring='mse',
#         error_action='ignore',  
#         suppress_warnings=True, 
#         stepwise=True,
#     )
    
#     fcX, confInt = model.predict(n_periods=fcPeriod, return_conf_int=True)
#     fcDates = pd.bdate_range(start=x.index[-1] + relativedelta(days=1), periods=fcPeriod)
#     fc = pd.Series(fcX, index=fcDates)
#     confL = pd.Series(confInt[:, 0], index=fcDates)
#     confH = pd.Series(confInt[:, 1], index=fcDates)
    
#     return fc, confL, confH

# Prophet
def fbProphet(x, fcPeriod, bc=True):
    x = x.reset_index()
    x = x[['Date', 'Close']]
    x.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    if bc:
        x['y'], lamb = boxcox(x['y'])
    
    # Define and fit model
    model = Prophet(yearly_seasonality=True, 
                    weekly_seasonality=True, 
                    daily_seasonality=True,
                    seasonality_prior_scale=0.5,
                    changepoint_range=0.9,
                    changepoint_prior_scale=0.5,
                    )
    model.fit(x)
    
    future = model.make_future_dataframe(periods=fcPeriod)
        
    # Forecast
    fc = model.predict(future)
    
    if bc:
        fc[['yhat','yhat_upper','yhat_lower']] = fc[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lamb))
    
    fc.set_index('ds', inplace=True)
    
    
    yHat = fc['yhat'][-fcPeriod:]
    yL = fc['yhat_lower'][-fcPeriod:]
    yU = fc['yhat_upper'][-fcPeriod:]
            
    return yHat, yL, yU

# Trend filter
# Hodrick-Prescott Filter
def hpf(x, l):
    '''
    Parameters
    ----------
    x : Pandas Series
    l : TYPE
        Smoothing parameter.
        Monthly data: l = 129600 (1600*3**4)
        Quarterly data: l = 1600
        Annual data: l = 6.25 (1600/4**4)
    '''
        
    cycle, trend = sm.tsa.filters.hpfilter(x, lamb=l)
    return cycle, trend

# Baxter-King Filter
def bkf(x, low=6, high=32, k=12):
    '''
    Parameters
    ----------
    x : Pandas Series
    low : Float
        Minimum period for oscillations.
        Quarterly data: 6
        Annual data: 1.5
    high : Float
        Maximum period for oscillations.
        Quarterly data: 32
        Annual data: 8
    k : Float
        Lead-lag length.
        Quarterly data: 12
        Annual data: 3
    '''

    cycle = sm.tsa.filters.bkfilter(x, low, high, k)
    return cycle

# Christiano Fitzgerald Filter
def cff(x, low, high, drift):
    '''
    Parameters
    ----------
    x : Pandas Series
    low : float
        Minimum period for oscillations.
        Quarterly data: 6
        Annual data: 1.5
    high : float
        Maximum period for oscillations.
        Quarterly data: 32
        Annual data: 8
    drift : bool
        Whether or not to remove a trend from the data.
    '''
    
    cycle, trend = sm.tsa.filters.cffilter(x, low, high, drift)
    return cycle, trend

# Fourier transform
def fourier(x, components):
    # x: Pandas Series
    xF = fft(x.to_numpy())
    
    xF[components:-components] = 0
    
    y = ifft(xF)
    
    return pd.Series(np.abs(y), index=x.index) #y.real
    

# Kalman Filter
def kalman(x):
    # x: Pandas Series
    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = x[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01
              )
    trend, _ = kf.filter(x.to_numpy())
    trend = trend.flatten()
    return pd.Series(trend, index=x.index)

# Portfolio analysis


class Portfolio:
    ''' Docstring '''
    def __init__(self, close, riskFreeReturn=0.03, timeFrame='a'):
        self.returns = close.pct_change()
        self.meanReturns = self.returns.mean()
        self.covMatrix = self.returns.cov()
        self.downCovMatrix = self.returns.clip(0).cov()
        self.rfr = riskFreeReturn/252 # Daily return
        self.tickers = close.columns.values
        self.numAssets = len(self.tickers)
        
        if timeFrame == 'm':
            self.tf = 5*4
        elif timeFrame == 'q':
            self.tf = 5*4*3
        elif timeFrame == 'a':
            self.tf = 252 # Annual trading days
        else:
            raise ValueError()
        
        # Optimisation 
        self._w = cp.Variable(self.numAssets)                
    
    def setRFR(self, riskFreeReturn):
        self.rfr = riskFreeReturn
    
    def performance(self, w, riskRatio):
        
        ret = self.meanReturns.dot(w)
                       
        if riskRatio == 'sharpe':
            risk = np.sqrt(cp.quad_form(w, self.covMatrix).value)
            rar = self.sharpe(w)
        elif riskRatio == 'sortino':
            risk = np.sqrt(cp.quad_form(w, self.downCovMatrix).value)
            rar = self.sortino(w)
        
        return np.array([ret * self.tf, risk * np.sqrt(self.tf), rar])
    
    def sharpe(self, w):
        
        ret = self.meanReturns.dot(w)
        
        risk = np.sqrt(cp.quad_form(w, self.covMatrix).value)
        sharpe = np.sqrt(self.tf) * (ret - self.rfr)/risk
        
        return sharpe
        
    def sortino(self, w):
        ret = self.meanReturns.dot(w)
        
        risk = np.sqrt(cp.quad_form(w, self.downCovMatrix).value)
        sortino = np.sqrt(self.tf) * (ret - self.rfr)/risk
        
        return sortino
    
    def optimize(self, problem, riskRatio):
        
        if riskRatio == 'sharpe':
            risk = cp.quad_form(self._w, self.covMatrix)
        elif riskRatio == 'sortino':
            risk = cp.quad_form(self._w, self.downCovMatrix)
          
        # Minimize risk
        if problem == 'minrisk':
            
            constraints = [
                cp.sum(self._w) == 1,
                self._w >= 0
            ]
        
        # Maximize risk
        elif problem == 'maxrar':
            k = cp.Variable()
                                
            constraints = [
                cp.sum((self.meanReturns.to_numpy() - self.rfr).T @ self._w) == 1,
                cp.sum(self._w) == k,
                k >= 0,
                self._w <= k,
                #(self._w/k) >= 0
            ]
             
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()
        
        if problem == 'maxrar':
            w = (self._w.value / k.value)
        else:
            w = self._w.value
        
        perf = self.performance(w, riskRatio)
        cols = np.concatenate((np.array(['Return', 'Risk', 'RAR']), 
                              self.tickers))
        data = np.concatenate((perf, w))
        
        result = pd.DataFrame([data], columns=cols)
                
        return result
                
    def efficientFrontier(self, nSamples, riskRatio):
        # Risk ratio
        
        gamma = cp.Parameter(nonneg=True)
        ret = self.meanReturns.to_numpy().T @ self._w  
        if riskRatio == 'sharpe':
            risk = cp.quad_form(self._w, self.covMatrix)
        elif riskRatio == 'sortino':
            risk = cp.quad_form(self._w, self.downCovMatrix)
                    
        constraints = [
            cp.sum(self._w) == 1, 
            self._w >= 0
        ]
        prob = cp.Problem(cp.Maximize(ret - gamma*risk), constraints)
        
        data = np.zeros((self.numAssets + 3, nSamples))
        
        gammaVals = np.linspace(0, 10, num=nSamples)
        for i in range(nSamples):
            gamma.value = gammaVals[i]
            prob.solve()
            
            data[0, i] = ret.value * self.tf
            data[1, i] = np.sqrt(risk.value * self.tf)
            data[2, i] = gamma.value
            
            for j in range(len(self._w.value)):
                data[j+3, i] = self._w.value[j]
            
        cols = np.concatenate((np.array(['Return', 'Risk', 'RAR']), 
                               self.tickers))    
        result = pd.DataFrame(data.T, columns=cols)
        
        return result
            
    def monteCarlo(self, nSims, riskRatio):
        def randWeights(n):
            ''' Produces n random weights that sum to 1 '''
            k = np.random.rand(n)
            return k / np.sum(k)
        
        data = np.zeros((self.numAssets + 3, nSims))
        
        # Monte Carlo
        for s in range(nSims):
            w = randWeights(self.numAssets)
            perf = self.performance(w, riskRatio)
    
            data[0, s] = perf[0] # Return
            data[1, s] = perf[1] # Risk
            data[2, s] = perf[2] # Return-to-risk

            for j in range(len(w)):
                data[j+3, s] = w[j]
        
        cols = np.concatenate((np.array(['Return', 'Risk', 'RAR']), 
                               self.tickers))  
        result = pd.DataFrame(data.T, columns=cols)

        return result
        
#%% Web scrapping
def mic():
    # Market identifier code
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    url = 'https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv'

    with requests.Session() as s:
        rc = s.post(url, headers=headers)
        csv = rc.text
        
    buff = io.StringIO(csv) # Parse csv
    df = pd.read_csv(buff, delimiter=',')
    buff.close()

    df.rename(columns={'ISO COUNTRY CODE (ISO 3166)': 'COUNTRY CODE'}, inplace=True)

    return df

def searchTickers(src, instr, search):

    # SQLite database
    dbPath = Path().cwd() / 'data' / 'ticker.db'
    engine = sqla.create_engine(f'sqlite:///{dbPath}')
    tbl = f'{src}{instr.capitalize()}' # Table name

    if instr == 'stock':

        if src in {'yahoo', 'investing', 'marketwatch'}:

            # CTE for split exchange names
            xchgSplit = textwrap.dedent(f'''WITH split(mic, xchg, str) AS (
                    SELECT e.morningstarMic, "", e.{src}Exchange||"/" FROM exchange AS e 
                        WHERE e.{src}Exchange IS NOT NULL
                    UNION ALL SELECT
                    mic,
                    substr(str, 0, instr(str, "/")),
                    substr(str, instr(str, "/")+1)
                    FROM split WHERE str != ""
                ) 
            ''')

            value = f'{tbl}.ticker || "|" || split.mic'
            if src == 'investing':
                value += f' || "|" || {tbl}.name || "|" {src}Link || "|" || {src}Id'

            elif src == 'marketwatch':
                value = f' || "|" || {tbl}.exchange || "|" || {tbl}.country'

            value += ' AS value'

            query = textwrap.dedent(f'''
                {xchgSplit}
                SELECT {tbl}.name || " (" || {tbl}.ticker || ") - "  || split.mic AS label, {value}
                FROM {tbl}
                JOIN split
                    ON {tbl}.exchange = split.xchg
            ''')

        elif src == 'morningstar':
            query = textwrap.dedent(f'''
                SELECT {tbl}.name || " (" || {tbl}.ticker || ") - "  || morningstarMic AS label,
                {src}Id as value
                FROM {tbl}
            ''')

    query += f'WHERE label LIKE "%{search}%"'

    df = pd.read_sql(query, con=engine) 
    return df

def tickerList(src, instr, mic=''):

    # SQLite database
    dbPath = Path().cwd() / 'data' / 'ticker.db'
    engine = sqla.create_engine(f'sqlite:///{dbPath}')
    tbl = f'{src}{instr.capitalize()}' # Table name

    if instr == 'stock':

        if src in {'yahoo', 'investing', 'marketwatch'}:

            # CTE for split exchange names
            xchgSplit = textwrap.dedent(f'''WITH split(mic, xchg, str) AS (
                    SELECT e.morningstarMic, "", e.{src}Exchange||"/" FROM exchange AS e 
                        WHERE e.{src}Exchange IS NOT NULL
                    UNION ALL SELECT
                    mic,
                    substr(str, 0, instr(str, "/")),
                    substr(str, instr(str, "/")+1)
                    FROM split WHERE str != ""
                ) 
            ''')

            sel = ''
            if src == 'investing':
                sel += f', {src}Link AS link, {src}Id'

            elif src == 'marketwatch':
                sel = f', {tbl}.exchange, {tbl}.country'

            query = textwrap.dedent(f'''
                {xchgSplit}
                SELECT {tbl}.name, {tbl}.ticker, split.mic AS mic {sel}
                FROM {tbl}
                JOIN split
                    ON {tbl}.exchange = split.xchg
            ''')

            if mic:
                query += f'WHERE split.mic = "{mic}"'

        elif src == 'morningstar':
            query = textwrap.dedent(f'''
                SELECT {tbl}.name, {tbl}.ticker, morningstarMic AS mic, {src}Id as value
                FROM {tbl}
            ''')

            if mic:
                query += f'WHERE mic = "{mic}"'

    df = pd.read_sql(query, con=engine) 
    return df

#    cond = ''
#    if where:
#        cond = 'WHERE '
#        for i, (k, v) in enumerate(where.items()):
#            
#            if isinstance(v, str):
#                v = f'"{v}"'
#                
#            temp = f'{k} = {v}'
#            if i == 0:
#                cond += temp
#            else:
#                cond += ' AND ' + temp
#
#    if isinstance(sel, dict):
#
#        selQuery = ''
#        for k, v in sel.items():
#            selQuery += f'{k} AS {v}, '
#
#        selQuery = selQuery[:-2]
#
#    elif isinstance(sel, list):
#        selQuery = ', '.join(sel)
#
#    if sel:
#        selQuery = ', '.join(sel)
#        
#    tbl = f'{src}{security.capitalize()}'
#    query = f'SELECT {selQuery} FROM {tbl} ' 
#    
#    query += cond
#    tickers = pd.read_sql(query, con=engine) 
#    
#    return tickers

def tickersToDb(df, src, db):

    if db == 'sqlite':
        engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\ticker.db')

        for t in df['type'].unique(): # Store stock and etf tickers separately
            mask = df['type'] == t
            df.loc[mask, df.columns != 'type'].to_sql(
                f'{src}{t.capitalize()}', con=engine, index=False, if_exists='replace')

    elif db == 'mongo':
        client = MongoClient('mongodb://localhost:27017/')
        coll = client['finly']['tickers']

        records = df.to_dict('records')
        operations = []
        for r in tqdm(records):
            
            query = {
                '$and': [
                    {'type': 'exchange'}
                ]
            }
            if src in ['yahoo', 'fmp']:
                query['$and'].append(
                    {'$or': [
                        {f'{src}Exchange': r['exchange']},
                        {f'yahooSuffix': r['ticker'].split('.')[-1]} 
                    ]}
                )
            else:
                query['$and'].append({f'{src}Exchange': r['exchange']})

            mics = coll.find_one(query, {'mic': 1, 'morningstarMic': 1})
            query = {
                '$and': [
                    {'type': r['type']},
                    {'ticker.trim': r['tickerTrim']}
                ]
            }
            if r['type'] == 'stock':
                query['$and'].append({'exchange.morningstarMic': mics['morningstarMic']},)

            if coll.count_documents(query) == 0: # Fuzzy name match

                query = {
                    '$and': [
                        {'type': r['type']},
                        {f'ticker.{src}': {'$exists': False}},
                    ]
                }
                if r['type'] == 'stock':
                    query['$and'].append({'exchange.morningstarMic': mics['morningstarMic']})

                insertNew = False
                if coll.count_documents(query) == 0:
                    insertNew = True

                else:
                    stored = pd.DataFrame.from_dict(
                        coll.find(query, {'name': 1})
                    )
                    fuzzName = process.extractOne(
                        r['name'], stored['name'].tolist(), 
                        scorer=fuzz.token_set_ratio,
                        score_cutoff=100
                    )
                    if fuzzName is not None:
                        query = {
                            '$and': [
                                {'type': r['type']},
                                {'name': fuzzName[0]}
                            ]
                        }
                        if r['type'] == 'stock':
                            query['$and'].append({'exchange.morningstarMic': mics['morningstarMic']})

                        updt = {
                            '$set': {
                                f'ticker.{src}': r['ticker'],
                                f'name.{src}': r['name'],
                                'exchange.mic': mics['mic']
                            }
                        }
                        if r['type'] == 'etf':
                            updt['$set']['exchange.morningstarMic'] = mics['morningstarMic']

                        if src in ['barrons', 'investing']:
                            updt['$set'][f'link.{src}'] = r[f'{src}Link']

                        if src == 'investing':
                            updt['$set'][f'id.{src}'] = r[f'{src}Id']

                        if src == 'fmp':
                            updt['$set']['fmpFinStatement'] = r['fmpFinStatement']

                        operations.append(UpdateOne(query, updt, upsert=True)) #coll.update_one(query, updt)
                    else:
                        insertNew = True

                if insertNew: # Insert new document
                    query = {
                        'type': r['type'],
                        'ticker': {
                            f'{src}': r['ticker'],
                            'trim': r['tickerTrim']
                        },
                        f'name.{src}': r['name'],
                        'exchange': {
                            'mic': mics['mic'],
                            'morningstar': mics['morningstarMic']
                        }
                    }
                    if src in ['barrons', 'investing']:
                        query['link'] = {f'{src}': r[f'{src}Link']} 

                    if src == 'investing':
                        query['id'] = {f'{src}': r[f'{src}Id']}
                        query['sector'] = r['sector']
                        query['industry'] = r['industry']

                    operations.append(InsertOne(query)) #coll.insert_one(query)
            else: # Ticker match found

                query = {
                    '$and': [
                        {'type': r['type']},
                        {'$or': [
                            {'ticker.morningstar': r['ticker'].split('.')[0]},
                            {'ticker.trim': r['tickerTrim']}
                        ]}
                ]}
                if r['type'] == 'stock':
                    query['$and'].append({'exchange.morningstarMic': mics['morningstarMic']})

                updt = {
                    '$set': {
                        f'ticker.{src}': r['ticker'],
                        'exchange.mic': mics['mic']
                    },
                    '$setOnInsert': {
                        'ticker.trim': r['tickerTrim'],
                        'name': r['name'],
                    }
                }
                if src in ['barrons', 'investing']: # if f'{src}Link' in r
                    updt['$set'][f'link.{src}'] = r[f'{src}Link']
                
                if src == 'investing': # if f'{src}Id' in r
                    updt['$set'][f'id.{src}'] = r[f'{src}Id']
                    updt['$setOnInsert']['sector'] = r['sector']
                    updt['$setOnInsert']['industry'] = r['industry']
                
                if src == 'fmp':
                    updt['$set']['fmpFinStatement'] = r['fmpFinStatement']

                if r['type'] == 'stock':
                    updt['$setOnInsert']['exchange.morningstarMic'] = mics['morningstarMic']
            
                elif r['type'] == 'etf':
                    updt['$set']['exchange.morningstarMic'] = mics['morningstarMic']

                operations.append(UpdateOne(query, updt, upsert=True)) #coll.update_one(query, updt, upsert=True)
            
            if len(operations) == 1000:
                coll.bulk_write(operations, ordered=False)
                operations = []

        if len(operations) > 0 :
            coll.bulk_write(operations,ordered=False)

def getMorningstarId(src, ticker, mic):
    engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\ticker.db')

    where = f'WHERE stock.{src}Ticker = "{ticker}" AND stock.morningstarMic = "{mic}"'
    query = f'SELECT stock.morningstarId FROM stock {where}'
    
    mId = pd.read_sql(query, con=engine)
    if not mId.empty:
        return mId.squeeze()
    
    else:
        return None

def yfOhlc(ticker, startDate, store=True):
    
    def parser(ticker, startDate):
        
        endDate = dt.now() + relativedelta(days=1)
        
        df = yf.download(
            tickers=ticker,
            start=startDate,
            end=endDate.strftime('%Y-%m-%d'),
            group_by='column',
            auto_adjust=True,
            treads=True,
        )
        
        # Lower case columns
        df.index.names = ['date']
        df.columns = map(str.lower, df.columns)
        
        return df
    
    if store:
        engine = sqla.create_engine(r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\tickerData.db')
        
        if not isinstance(ticker, list):
            ticker = [ticker]
        
        dfs = [None] * len(ticker)
        for i, t in enumerate(ticker):
        
            if not engine.dialect.has_table(engine, t):
                
                parse = parser(t, startDate)
                parse.to_sql(t, con=engine, index=True, if_exists='replace')
                
            else:
                query = f'SELECT * FROM {t}'
                parse = pd.read_sql(query, con=engine)
                parse['date'] = pd.to_datetime(parse['date'], 
                                               format='%Y-%m-%d %H:%M:%S.%f')
                parse.set_index('date', inplace=True)
                
                if parse.index[-1].date != dt.now().date():
                    startDate = parse.index[-1]
                    temp = parser(t, startDate)
                    mask = temp.index > startDate
                    parse = pd.concat([parse, temp.loc[mask]])
                    #parse.append(temp.loc[mask])
                    parse.to_sql(t, con=engine, index=True, if_exists='replace')
            
            if len(ticker) > 1:
                mCols = pd.MultiIndex.from_product([[t], parse.columns])
                parse.columns = mCols
            dfs[i] = parse
                
        if len(ticker) == 1:
            data = dfs[0]
            
        else:
            data = pd.concat(dfs, axis=1)
            
    else:
        data = parser(ticker, startDate)
    
    return data

# Features
def quandlData(ticker, startDate, valName=None):
    # API key: KH1x3LUCNac4ExyLpNWm
    quandl.ApiConfig.api_key = 'KH1x3LUCNac4ExyLpNWm'
    data = quandl.get(ticker, start_date=startDate)
    data.index.names = ['date']
    
    if valName is not None:
        if isinstance(valName, list):            
            rename = {f'{i} - Value': j for i, j in zip(ticker, valName)} 
            
        else:
            rename = {'Value': valName}
        
        data.rename(columns=rename, inplace=True)
   
    return data

def buffet(country='US', startDate='1947-01-01'):
    
    if country == 'US':
        fredSym = 'GDPC1'
        ixTicker = '^W5000' # 'FRED/WILL5000PRFC'
    elif country == 'NOR':
        fredSym = 'CPMNACSCAB1GQNO'
        ixTicker = '^OSEAX'

    gdp = fredTicker(fredSym).timeSeries(startDate, 'gdp')
    gdp = gdp.resample('D').ffill()

    smc = yhTicker(ixTicker).ohlcv(startDate)['close']    
    buffet = pd.merge(smc, gdp, on='date', how='left')
    
    return buffet['close']*1e3 / buffet['gdp']

def mTwoPerCapita():
    
    # US population
    url = ('https://api.census.gov/data/1990/pep/int_natrespop'
           '?get=YEAR,MONTH,TOT_POP') # 1990-2000
    
    with requests.Session() as s:
            rc = s.get(url)
            parse = json.loads(rc.text)
    
    scrap = []
    
    for m in parse[1:]:
        date = dt.strptime(f'1-{m[1]}-{m[0]}', '%d-%B-%Y')
        
        scrap.append({
            'date': date,
            'population': int(m[2]),
        })
    
    # 2000-
    url = [
        ('https://api.census.gov/data/2000/pep/int_natmonthly'
         '?get=POP,MONTHLY_DESC&for=us:1'),
        ('https://api.census.gov/data/2019/pep/natmonthly'
         '?get=POP,MONTHLY_DESC&for=us:1')
    ]
    
    for u in url:
        
        with requests.Session() as s:
            rc = s.get(u)
            parse = json.loads(rc.text)
            
        for m in parse[1:]:
            
            pattern = '[1-9][0-2]?/1/[1-2][0-9]{3}'
            date = re.search(pattern, m[1]).group()
            date = dt.strptime(date, '%m/%d/%Y')
            
            scrap.append({
                'date': date,
                'population': int(m[0]),
            })
    
    pop = pd.DataFrame.from_records(scrap)
    pop.set_index('date', inplace=True)
    pop = pop[~pop.index.duplicated(keep='first')]
    pop = pop.resample('D').ffill()
    
    # Money supply
    startDate = dt.strftime(pop.index.min(), '%Y-%m-%d')
    mBase = fredTicker('BOGMBASE').timeSeries(startDate)
    mRes = fredTicker('BOGMBBM').timeSeries(startDate)
    dxy = fredTicker('DTWEXBGS').timeSeries(startDate)
    
    m = pd.merge(mBase, mRes, on='date', how='left')
    m = pd.merge(m, dxy, on='date', how='left')
    
    m['m1'] = (m['BOGMBASE'] - m['BOGMBBM'])*1e6 / m['DTWEXBGS']
    m = pd.merge(m, pop, on='date', how='left')
    
    return m['m1'] / m['population']
    
# Features
def macroFeatures(startDate, roc=False):
    features = pd.DataFrame()
    
    # CAPE
    cape = quandlData(ticker='MULTPL/SHILLER_PE_RATIO_MONTH',
        startDate=startDate,
        valName='CAPE',                     
    )
    cape.resample('D').ffill()
    
    # Buffet indicator (Wilshire 5000)
    gdp = quandlData(ticker='FRED/GDP',
        startDate=startDate,
        valName='gpd'
    )
    gdp.resample('D').ffill()
    smc = yhTicker('^W5000').ohlcv(startDate)['close']
    
    buffet = pd.merge(smc, gdp, on='date', how='left')
    buffet['buffet'] = buffet['close'] / buffet['gdp']
    
    # Quandl data (oil, gold, forex, PMI)
    quandlQuery = [
        'FRED/DCOILBRENTEU',
        'WGC/GOLD_DAILY_USD',
        'LBMA/SILVER', #WORLDBANK/WLD_SILVER',
        'FRED/DFF', # Federal-Funds-Rate
        'ECB/EURUSD',
        'FRED/DEXUSUK',
        'FRED/DEXCHUS',
        'FRED/DEXINUS',
        'FRED/DEXJPUS',
        'ISM/MAN_PMI'
    ]
    
    # FRED/LNS14000024 (unemployment rate)
    
    for q in quandlQuery:
        val = q.split('/')[-1]
        features = quandlData(ticker=q,
            startDate=startDate,
            valName=val,
            df=features,
            merge=True
        )
        
    # Major indices
    indices = ['^GSPC', '^DJI', '^IXIC', '^N225', '^HSI', '^GDAXI', '^FTSE', '^VIX']

    indexData = yf.download(
        tickers=indices,
        start=startDate.strftime('%Y-%m-%d'),
        end=dt.now().strftime('%Y-%m-%d'),
        group_by = 'column',
        auto_adjust = True,
        treads = True,
    )['Close']

    indexData.index.names = ['date']
    
    features = pd.merge(features, indexData, on='date', how='outer')
    features.sort_index(inplace=True)
    
    # Interpolate NaN
    features.fillna(method='ffill', inplace=True)
    features.dropna(inplace=True)
        
    # ROC
    if roc:
        features = features.pct_change().dropna()
        
        # Market Regime
        features['hmm'] = hmm(features[indices], roc=False)
    else:
        features['hmm'] = hmm(features[indices], roc=True)
    
    # Save
    path = 'data/Features.csv'
    if os.path.exists(path):
        os.remove(path)
    
    features.to_csv(path, encoding='utf-8-sig')
    
    return features


def techFeatures(ohlc, rec=True):
    # Seasonality
    ohlc.reset_index(inplace=True)
    #add_datepart(ohlc, 'date', drop=False)
    ohlc.set_index('date', inplace=True)
    ohlc.drop('Elapsed', axis=1, inplace=True) # don't need this
    ohlc.drop('Is_year_start', axis=1, inplace=True) # never a trading day
    
    #ohlc['weekday'] = ohlc.index.weekday
    #ohlc['day'] = ohlc.index.day
    #ohlc['month'] = ohlc.index.month
    #ohlc['quarter'] = (ohlc['month']-1)//3

    
    if not rec: # Features dependent on open/high/low
        # Intraday Volatility
        ohlc['gap'] = ohlc['open'] - ohlc['close'].shift(1)
        ohlc['volatility'] = ohlc['high'] - ohlc['low']
        ohlc['relVol'] = ohlc['volatility'].pct_change()
        
        ohlc['adx'] = ADX(ohlc) # Average directional movement index
        ohlc['cci'] = CCI(ohlc) # Commodity channel index
        ohlc['mi'] = MI(ohlc) # Mass index
        ohlc['uo'] = UO(ohlc) # Ultimate oscillator
        ohlc = SO(ohlc) # Stochastic oscillator
        ohlc['VI+'], ohlc['VI-'] = VI(ohlc) # Vortex indicator
        ohlc['VIx'] = ohlc['VI+'] - ohlc['VI-']

    # Technical indicators
    for p in [10, 20, 60]:
        ohlc[f'zlema{p}'] = zlema(ohlc['close'], p)
        ohlc[f'hma{p}'] = hma(ohlc['close'], p)
    ohlc['maX'] = ohlc['zlema20'] - ohlc['zlema60']
    ohlc['kama'] = kama(ohlc['close'])
    ohlc['std'] = ohlc['close'].rolling(window=20, min_periods=0).std() # Standard deviation
    
    ohlc['rsi'] = RSI(ohlc['close']) # Relative strength index
    ohlc['trix'] = TRIX(ohlc['close']) # Triple exponential
    ohlc['tsi'] = TSI(ohlc['close']) # True strength index

    # Bollinger
    lB, uB = bollinger(ohlc['close'], 20)
    ohlc['bb'] = uB - lB
    
    # Hodrick-Prescot Filter
    ohlc['hpfC'], ohlc['hpfT'] = hpf(ohlc['close'] , 1600)
    ohlc['kalman'] = kalman(ohlc['close'])
        
    # Remove nullvalues
    ohlc.fillna(method='ffill', inplace=True)
    ohlc.replace([np.inf, -np.inf], np.nan, inplace=True)
    ohlc.dropna(inplace=True)
    #ohlc.fillna(0, inplace=True)

    return ohlc


# Currency rates
def getNorgesBankData(what, startDate, currency, bondType, df, merge=False):
    url = 'https://data.norges-bank.no/api/data/'
    
    # URL
    if what == 'currency':
        url += f'EXR/B.{currency}.NOK.SP?StartPeriod={startDate}&format=csv-:-comma-false-y'
    elif what == 'keyrate':
        url += f'IR/B.KPRA.SD.R?StartPeriod={startDate}&format=csv-:-comma-false-y'
    elif what == 'bond':
        url += f'IR/B.GBON.{bondType}.R?StartPeriod={startDate}&format=csv-:-comma-false-y'
    
    with requests.Session() as s:
        resp = s.get(url)
        csv = resp.text
    buff = io.StringIO(csv)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d') 
    data = pd.read_csv(buff, delimiter=',', index_col=0, parse_dates=True, date_parser=dateparse)
    data.index.names = ['date']
    
    # Merge data with parent dataframe
    if merge:
        if what == 'currency':
            df = pd.merge(df, data[f'B:{currency}:NOK:SP'], on='date', how='left')
        elif what == 'keyrate':
            df = pd.merge(df, data['B:KPRA:SD:R'], on='date', how='left')
        elif what == 'bond':
            df = pd.merge(df, data[f'B:GBON:{bondType}:R'], on='date', how='left')

# def finNews(query):
#     api = NewsApiClient(api_key='4155fba99b97448b80a2b1a9da429bba')
    
#     parse = api.get_everything(q=query)
        
#     scrap = []
#     for n in parse['articles']:
#         date = n.get('publishedAt').replace('T', ' ')
#         date = dt.strptime(date, '%Y-%m-%d %H:%M:%S%Z')
        
#         scrap.append({
#             'date': date,
#             'source': n['source'].get('name'),
#             'headline': n.get('title'),
#             'summary': n.get('description')
#         })
    
#     df = pd.DataFrame.from_records(scrap)
    
#     df['date'] = df['date'].dt.normalize()
    
#     analyzer = SentimentIntensityAnalyzer()
#     df['sentiment'] = df['summary'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
#     return df

def googleRelatedQueries(term):
    '''
    Parameters
    ----------
    term : string
        Search term.

    Returns
    -------
    DataFrames 
    '''
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=[term])
    
    relQueriesDict = pytrend.related_queries()
    topQueries = relQueriesDict['top']
    risingQueries = relQueriesDict['rising']
    
    return topQueries, risingQueries
    
def bitcoinData():
    
    # Ethereum: https://ethereumprice.org/defi-charts/
    # CFTS bitcoin futures
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Sec-GPC': '1',
        'If-None-Match': 'W/12e39-uVIO+VDOkcPKK6HdhVy63VQlLcQ',
        'Cache-Control': 'max-age=0',
        'TE': 'Trailers',
    }
    
    # Metrics
    metrics = {
        'daily-transactions': 'tx',
        'transactions-per-block': 'txBlk',
        'fee-percentage': 'feePct',
        'block-size': 'blkSz',
        'blocks-daily': 'blkDly',
        'blockchain-size': 'blkChnSz',
        'metcalfe-tx': 'metTx',
        'metcalfe-utxo': 'metUtxo',
        'money-supply': 'mnySply',
        'hash-rate': 'hashRate',
        'difficulty': 'dfclty',
        'miner-revenue': 'mineRevBtc',
        'miner-revenue-value': 'mineRevUsd',
        'inflation': 'infl',
    }
    
    dfs = [None] * len(metrics)
    for i, k in enumerate(metrics):
        
        url = f'https://charts.bitcoin.com/btc/api/chart/{k}'
    
        with requests.Session() as s:
            rc = s.get(url, headers=headers)
            parse = json.loads(rc.text)
            
        scrap = []
        for j in parse:
            
            scrap.append({
                'date': dt.fromtimestamp(i[0]),
                metrics[k]: i[1]
            })
        
        temp = pd.from_records(scrap)
        temp.set_index('date', inplace=True)
        dfs[i] = temp
    
    df = pd.concat(dfs, axis=1)
    
    return df

# World Bank data
def worldBankData(tickers, startYear=''):
    '''
    Parameters
    ----------
    tickers : dict
        Dictionary with keys representing World Bank API indicators and values
        corresponding column labels.
    startYear : str, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    DataFrame

    '''
    
    def parser(tickers, startYear):
    
        if startYear:
            endYear = dt.now().year
            period = startYear + f':{endYear}'
            
        else:
            period = ''
        
        dfs = [None] * len(tickers)
        for i, k in enumerate(tickers):
            
            tmp = wb.get_series(k, date=period, id_or_value='id', 
                                simplify_index=True)
            tmp.rename(tickers[k], inplace=True) 
            dfs[i] = tmp
        
        df = pd.concat(dfs, axis=1)
        df.index.rename(['country', 'year'], inplace=True)
        
        return df
    
    engine = sqla.create_engine(
        r'sqlite:///C:\Users\danfy\OneDrive\FinAnly\data\macroData.db')
        
    if not engine.dialect.has_table(engine, 'wb'):
        
        parse = parser(tickers, startYear)
        parse.to_sql('wb', con=engine, index=True, if_exists='replace')
                                    
    else:
        query = 'SELECT * FROM wb'
        parse = pd.read_sql(query, con=engine)
        lastYear = parse['date'].iloc[-1]
        
        parse.index = pd.MultiIndex.from_frame(parse[['country', 'year']], 
                                               names=['name', 'date'])
        
        parse.drop(['country', 'name'], axis=1, inplace=True)
        
                
        if dt.now().year > lastYear + 3:
            parse = parser(tickers, startYear)
            parse.to_sql('wb', con=engine, index=True, if_exists='replace')
                
    return parse
    
def finraMarginDebt():
    
    def dateParse(strDate):
        
        strDate = strDate.replace('Sept', 'Sep')
        
        for fmt in ('%b-%y', '%B-%y'):
            try:
                return dt.strptime(strDate, fmt)
            except ValueError:
                pass
    
    url = 'https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics'
    
    with requests.Session() as s:
        rc = s.get(url)
        parse = bs.BeautifulSoup(rc.text, 'lxml')
        
    tbls = parse.findAll('table', {'class': 'rwd-table width100'})
    tbls = tbls[:-15] + tbls[-14:]
    
    scrap = []
    
    for tbl in tbls:
        
        for tr in tbl.findAll('tr')[1:]:
            
            td = tr.findAll('td')
            if len(td) < 2:
                continue
            
            scrap.append({
                'date': dateParse(td[0].text),
                'mrgDbt': int(td[1].text.replace(',', '')) * 1e6
            })
        
    df = pd.DataFrame.from_records(scrap)
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    return df


# def ssbMacroData():
    
#     url = 'https://data.ssb.no/api/v0/dataset/59012.json?lang=en'
    
#     with requests.Session() as s:
#         rc = s.get(url)
#         parse = json.loads(rc.text)
        
#     cols = list(parse['dataset']['dimension']['Makrost']['category']['label'].values())
    
#     rows = list(parse['dataset']['dimension']['Tid']['Category']['label'])
