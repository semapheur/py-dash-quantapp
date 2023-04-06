import pandas as pd

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

# Web scrapping
import requests
import json

# Databasing
import sqlalchemy as sqla
from pymongo import MongoClient, DESCENDING

# Utils
from pathlib import Path

# Local
from lib.foos import updateDict
from lib.finlib import finItemRenameDict

class Ticker():

    # Class variable
    _apiKey = 'f32bfcc6073ae29d5b18158e8360288cf4a91a463ca41100e2c8aaac00fde67a'
    
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    def __init__(self, ticker):
        self._ticker = ticker

    def filings(self, startDate='2000-01-01', formType='10-Q'):
        '''
        Returns list of SEC filings for a given ticker

        '''

        # Edgar query
        endDate = dt.now().strftime('%Y-%m-%d')
        
        qFilter = (
            f'ticker:{self._ticker} AND filedAt:{{{startDate} TO {endDate}}} ' 
            f'AND formType:\"{formType}\" ' 
            f'AND dataFiles.description:\"EXTRACTED XBRL INSTANCE DOCUMENT\"'
        )
        start = 0 # Batch intervall
        size = 10000 # Batch size
        
        # Sort in descending order by filedAt
        sort = [{ 'filedAt': { 'order': 'desc' } }]
        
        data = {
            'query': { 'query_string': { 'query': qFilter } },
            'from': start,
            'size': size,
            'sort': sort
        }
        url = f'https://api.sec-api.io?token={self._apiKey}'
        
        with requests.Session() as s:
            rc = s.post(url, headers=self._headers, json=data)
            parse = json.loads(rc.text)
        
        return parse

    def financials(self, startDate='2000-01-01'):
    
        # MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['finly']
        coll = db['secSheets']
        
        for formType in ['10-K', '10-Q']:
            
            # Get list of financial statements
            lstSheets = self.filings(self._ticker, startDate=startDate, formType=formType)
            sheets = []
            
            for f in lstSheets['filings']:
                
                sheet = self.financialsToJson(f['accessionNo'])
                fltr = {
                    'EntityFileNumber': sheets['EntityFileNumber'],
                    'EntityCentralIndexKey': sheets['EntityCentralIndexKey']
                }
                coll.update_one(fltr, sheet, upsert=True)

    def loadFinancials(self):
        
        period = {'10-Q': 'q', '10-K': 'a'}

        # MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['finly']
        coll = db['secSheets']
        
        dfs = []
        
        fltr = {'Metadata.TradingSymbol': self._ticker}
        query = coll.find(fltr, {'_id': False}).sort(
            'Metadata.DocumentPeriodEndDate', DESCENDING)

        record = next(query)
        if record is None:
            self.financials()

        lastDate = record['Metadata']['DocumentPeriodEndDate']

        if relativedelta(dt.now(), lastDate).months > 3:
            startDate = dt.strftime(lastDate, '%Y-%m-%d')
            self.financials(startDate + relativedelta(days=1))
         
        for x in query:
            temp = financialsToFrame(x)
            formType = x['Metadata']['DocumentType']
            temp['period'] = period[formType]
            dfs.append(temp)
        
        df = dfs.pop(0)
        for i in dfs:
            df = df.combine_first(i)
            diffCols = i.columns.difference(df.columns)

            if diffCols:
                df = df.join(i[diffCols], how='outer')

        return df

    @classmethod
    def financialsToJson(cls, accession):
        '''
        Gets EDGAR financial statement in XBRL format and parses it to JSON

        '''
        
        def dateparse(x):
            try:
                return dt.strptime(x, '%Y-%m-%d')
            except:
                return x

        def fixJson(parse):
                        
            fixParse = {}
        
            keys = {
                'CoverPage': 'Metadata', 
                'StatementsOfIncome': 'StatementsOfIncome',
                'StatementsOfComprehensiveIncome': 'StatementsOfComprehensiveIncome',
                'BalanceSheets': 'BalanceSheets', 
                'StatementsOfCashFlows': 'StatementsOfCashFlows',
                'IncomeTaxesAdditionalInformationDetails': 'IncomeTaxes',
                'FinancialInstrumentsAdditionalInformationDetails': 'FinancialInstruments',
                'DebtSummaryofTermDebtDetails': 'Debt'
            }
            
            for key in set(parse.keys()).intersection(set(keys.keys())):
                fixParse[keys[key]] = parse[key] 
        
            # Metadata
            delItems = [
                'DocumentQuarterlyReport',
                'DocumentAnnualReport',
                'DocumentTransitionReport',
                'DocumentFiscalPeriodFocus',
                'DocumentFiscalYearFocus',
                'EntityIncorporationStateCountryCode',
                'EntityTaxIdentificationNumber',
                'EntityAddressAddressLine1',
                'EntityAddressCityOrTown',
                'EntityAddressStateOrProvince',
                'EntityAddressPostalZipCode',
                'EntityWellKnownSeasonedIssuer',
                'EntityVoluntaryFilers',
                'IcfrAuditorAttestationFlag',
                'EntityPublicFloat',
                'CityAreaCode',
                'LocalPhoneNumber',
                'Security12bTitle',
                'NoTradingSymbolFlag',
                'EntityCurrentReportingStatus',
                'EntityInteractiveDataCurrent',
                'EntityFilerCategory',
                'EntitySmallBusiness',
                'EntityEmergingGrowthCompany',
                'EntityShellCompany',
                'EntityCommonStockSharesOutstanding',
                'AmendmentFlag',
            ]
        
            temp = fixParse['Metadata']
            temp['TradingSymbol'] = parse['CoverPage']['TradingSymbol']
            if isinstance(temp['TradingSymbol'], dict):
                 temp['TradingSymbol'] = temp['TradingSymbol']['value']
            
            try:
                temp['SecurityExchangeName'] = parse['CoverPage']['SecurityExchangeName']
                if isinstance(temp['SecurityExchangeName'], list):
                     temp['SecurityExchangeName'] = temp['SecurityExchangeName'][0]['value']
                        
            except:
                print(json.dumps(parse, indent=4))

            for di in set(temp.keys()).intersection(set(delItems)):
                temp.pop(di)
            
            fixParse['Metadata'] = temp
        
            # Merge items
            sheets = {
                'StatementsOfIncome': [
                    'CondensedConsolidatedFinancialStatementDetailsOtherIncomeExpenseNetDetails',
                    'SegmentInformationandGeographicDataReconciliationofSegmentOperatingIncometotheCondensedConsolidatedStatementsofOperationsDetails',
                ],
                'BalanceSheets': [
                    'CondensedConsolidatedFinancialStatementDetailsPropertyPlantandEquipmentNetDetails',
                    'CondensedConsolidatedFinancialStatementDetailsOtherNonCurrentLiabilitiesDetails',
                    'FinancialInstrumentsRestrictedCashDetails',
                ],
                'FinancialInstruments': [
                    'FinancialInstrumentsCashCashEquivalentsandMarketableSecuritiesDetails',
                    'FinancialInstrumentsNonCurrentMarketableDebtSecuritiesbyContractualMaturityDetails',
                    #'FinancialInstrumentsRestrictedCashDetails',
                    'FinancialInstrumentsDerivativeInstrumentsatGrossFairValueDetails',
                    'FinancialInstrumentsPreTaxGainsandLossesofDerivativeandNonDerivativeInstrumentsDesignatedasCashFlowandNetInvestmentHedgesDetails',
                    'FinancialInstrumentsDerivativeInstrumentsDesignatedasFairValueHedgesandRelatedHedgedItemsDetails',
                    'FinancialInstrumentsNotionalAmountsandCreditRiskAmountsAssociatedwithDerivativeInstrumentsDetails'            
                ],
                'Debt': [
                    'DebtAdditionalInformationDetails',
                    'DebtSummaryofCashFlowsAssociatedwithCommercialPaperDetails'
                ],
                'StatementsOfShareholdersEquity': [
                    'StockholdersEquity'
                ]
            }
            
            for sheet in set(fixParse.keys()).intersection(set(sheets.keys())):
                
                for itm in set(parse.keys()).intersection(set(sheets[sheet])):
                
                        for k, v in parse[itm].items():
                            if v not in fixParse[sheet].values():
                                if k in fixParse[sheet].keys():
                                    fixParse[sheet][k].update(v)
                                else:
                                    fixParse[sheet][k] = v
                                                   
            return fixParse
        
        url = f'https://api.sec-api.io/xbrl-to-json?accession-no={accession}&token={cls._apiKey}'

        with requests.Session() as s:
            rc = s.get(url, headers=cls._headers)
            parse = json.loads(rc.text)
            
        parse = fixJson(parse)

        # Parse dates
        func = lambda x: dateparse(x)
        updateDict(parse, 'date', func)
        
        return parse    
    
def financialsToFrame(dctRaw):
    
    def insertValue(dct, col, val, endDate, period):
        if (endDate, period) not in dfData:
            dct[(endDate, period)] = {}
        
        if col not in dfData[(endDate, period)]:
            dfData[(endDate, period)][col] = val

        return dct

    formType = dctRaw['Metadata']['DocumentType']

    sheets = [
        'StatementsOfIncome',
        'BalanceSheets',
        'StatementsOfCashFlows',
    ]
    
    dfData = {}
    
    for sheet in sheets:
        for k, v in dctRaw[sheet].items():
            for i in v:
                
                if (not 'segment' in i) and ('value' in i):
                    
                    if 'instant' in i['period']:

                        if isinstance(i['period']['instant'], str):
                            endDate = dt.strptime(i['period']['instant'], '%Y-%m-%d')
                        else:
                            endDate = i['period']['instant']

                        dfData = insertValue(dfData, k, float(i['value']), endDate, 'a')

                        if formType == '10-Q':
                            dfData = insertValue(dfData, k, float(i['value']), endDate, 'q')
                            
                    else:
                        if isinstance(i['period']['startDate'], str):
                            startDate = dt.strptime(i['period']['startDate'], '%Y-%m-%d')
                        else:
                            startDate = i['period']['startDate']

                        if isinstance(i['period']['startDate'], str):
                            endDate = dt.strptime(i['period']['endDate'], '%Y-%m-%d')
                        else:
                            endDate = i['period']['startDate']
                        
                        if relativedelta(endDate, startDate).months > 4:
                            dfData = insertValue(dfData, k, float(i['value']), endDate, 'a')

                        else:
                            dfData = insertValue(dfData, k, float(i['value']), endDate, 'q')
    
    # Construct dataframe
    df = pd.DataFrame.from_dict(dfData, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ['date', 'period']
    rnm = finItemRenameDict('GAAP')
    df.rename(columns=rnm, inplace=True)
    df.sort_index(level=0, ascending=True, inplace=True)
    
    return df
