import numpy as np
import pandas as pd

import io
import re

# Date
from datetime import datetime as dt
from datetime import timezone as tz

# Pdf scrapping
import tabula #, camelot
import PyPdf2
from pdfminer.pdfinterp import PdfResourceManager, PdfPageInterpreter
from pdfminer.pdfpage import PdfPage
from pdfminer.converter import TextConverter #, XMLConverter, HTMLConverter
from pdfminer.layout import LAParams
#from pdfminer.pdfdocument import PdfDocument
#from pdfminer.pdfparser import PdfParser

def pdfExtractTbl(path):
    
    def setInterpreter():
        rm = PdfResourceManager()
        extract = io.BytesIO() #StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rm, extract, codec=codec, laparams=laparams)
        interpreter = PdfPageInterpreter(rm, device)
        return {'extract': extract, 'device': device, 'interpreter': interpreter}
    
    def fixTbl(df, sheet, valFactor, repDate):
        
        # Remove nan rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Delete note column
        for c in df.columns:
            if c == 'Note':
                df.drop(c, axis=1, inplace=True)
                break
            
            elif df[c].dtype == object:
                if df[c].str.contains('^Note$', regex=True).any():
                    df.drop(c, axis=1, inplace=True)
                    break

        # Fix multi-line rows
        c0 = df.columns[0]

        if df[c0].str.islower().any():

            for i in range(len(df[c0])):

                if not pd.isnull(df[c0].iloc[i]):
                    if df[c0].iloc[i].islower():
                        df[c0].iloc[i] = df[c0].iloc[i-1] + ' ' + df[c0].iloc[i]
                        #df.drop(i-1, inplace=True)

        # Remove remaining empty rows
        df.dropna(inplace=True)
        
        # Remove bad rows
        pattern = '\(?(unaudited, )?(in )?[A-Z]{3}( in)? (thousand|million|billion)\)?'
        mask = df[c0].str.contains(pattern, regex=True)
        df = df[~mask] #.copy()
        
        # Remove footnote markers
        df[c0] = df[c0].str.replace('\d\)( \d\))*', '', regex=True)
        
        # Set posts as index
        pattern = '(-|[(]\s?)?(0|[1-9]\d{0,2}([, ]\d{3}){0,3})([.]\d{1,3})?(\s?[)])?'
        if not df[c0].str.contains('\s' + pattern, regex=True).any():
            df.set_index(df.columns[0], inplace=True)
            df.index.rename('Post', inplace=True)
            c0 = df.columns[0]
            
            # Check for bad column parse

            if df[c0].astype(str).str.contains('(' + pattern + ' )+').any():

                # Reduce to single whitespaces to allow easy split
                df[c0] = df[c0].str.replace(' +', ' ', regex=True)

                # Count whitespaces
                spaces = df[c0].str.count(' ').min()

                # Split columns
                df = df[c0].str.rsplit(' ', spaces, expand=True).iloc[:,0]
                df = df.to_frame(repDate)

                # Remove leading notes
                df[repDate] = df[repDate].str.replace('^(\d, )?\d ', '', regex=True)
                
            else:
                df = df[df.columns[0]].to_frame(repDate)
        
        # Bad post column parse
        else: 
            # Extract values from post column
            ePattern = f'(?P<extract>{pattern})'
            
            temp = df[c0].str.extract(ePattern, expand=False)
            post = df[c0].str.replace(pattern, '', regex=True)
            
            df = pd.DataFrame({'Post': post, 'extract': temp['extract']})
            df.dropna(inplace=True)
            df.rename(columns={'extract': repDate}, inplace=True)
            # Set posts as index
            df.set_index('Post', inplace=True)
            
        return df
            
    
    def extractTblData(sheet, page, lookup, txt, repDate, valFactor):
        
        def rpl(x):
            rpl = {' ': '', '*': '', ',': '', '(': '-', ')': ''}
    
            for i, j in rpl.items():
                x = x.replace(i, j)
    
            return x
        
        complete = True
        completeCond = re.search('total equity', txt, re.I)
        if re.search(lookup, txt, re.I):
            
            if sheet == 'balance' and not completeCond:
                complete = False
            
            # Extract value factor and currency
            pattern = '[A-Z]{3}( in)? (thousand|million|billion)'
            match = re.search(pattern, txt)
            
            if match:
                match = match.group().split()
                curr = match[0]
                #curr = re.search('[A-Z]{3}', match.group()).group()
            
                # Factor
                if match[-1] == 'thousand': #re.search('thousand', match.group()):
                    valFactor = 1e3

                elif match[-1] == 'million': #re.search('million', match.group()):
                    valFactor = 1e6

                elif match[-1] == 'billion': #re.search('billion', match.group()):
                    valFactor = 1e9
            
            else:
                valFactor = valFactor
                curr = None
            
            # Extract items and values
            
            vPattern = (
                r'\b(-|[(]\s?)?'
                r'(0|[1-9]\d{0,2}(,\d{3}){0,3})([.]\d{1,3})?'
                r'(\s?[)])?\b'     
            )
            
            pPattern = (
                r'(?<! Condensed )'
                r'(((Revenues|Inventor(y|ies)|Receivables)|'
                r'TOTAL (ASSETS|(EQUITY|LIABILITIES) AND (EQUITY|LIABILITIES))) |'
                r'\(?[A-Z][a-z\-’]+\)?,?( (\(|\[)?[a-z/\-*]+( [A-Z]{3})?(\)|\])?)+ )'
                #'([(](Gains|Increase)[)]|[A-Z][a-z,\-]+)( [A-Za-z/\-()\[\]*]+)+ )'
                r'(?<!ended )(?<!note )(?<!on )(?<!of )(?<!of [A-Z]{3} )(?<!At )'
                r'(?<!year )(?<!half )(?<!quarter )'
            )
            
            cond = (
                (sheet == 'income' and re.search(vPattern + ' Revenues', txt)) or
                (sheet == 'cash' and re.search(vPattern + ' Income(/[(]loss[)])? before tax', txt))
            )
            
            if cond:
                pattern = '(' + vPattern + ' ){2,3}' + pPattern
                vSub = vPattern + '\s'
                
            else:
                vSub = '\s' + vPattern
                pattern = pPattern + vPattern
            
            if len(re.findall(pattern, txt)) > 10: #and not re.search('(' + vPattern + ' ){10}', txt):

                matches = re.finditer(pattern, txt)
                post = []
                val = []
                for m in matches:
                    if not re.search('statement', m.group(), re.I):
                        post.append(re.sub(vSub, '', m.group()))
                        val.append(re.search(vPattern, m.group()).group())
            
                df = pd.DataFrame(val, index=post, columns=[repDate])
                
            else:

                if sheet == 'income':
                    posts = [
                        'shares outstanding',
                        'earnings per share|eps'
                    ]
                    for p in posts:
                        if re.search(p, txt, re.I):
                            bPost = p
                    
                elif sheet == 'balance':
                    
                    if complete:
                        bPost = 'total (equity|liabilities) and (liabilities|equity)'
                        
                    else:
                        bPost = 'total (assets|liabilities)'
                        
                elif sheet == 'cash': 
                    bPost = 'cash equivalents at the end'
                
                
                df = tabula.read_pdf(path, pages=page+1, multiple_tables=False)
                
                # Check for bad table area width
                wOffset = 0
                if len(df.columns) > 10:
                    wOffset = 5
                    tblJson = tabula.read_pdf(path, pages=page+1, 
                                              multiple_tables=False, 
                                              output_format='json')

                    area = [
                        tblJson[0]['top'], 
                        tblJson[0]['left'], 
                        tblJson[0]['top'] + tblJson[0]['height'], 
                        tblJson[0]['left'] + tblJson[0]['width'] + wOffset,
                    ]

                    df = tabula.read_pdf(path, pages=page+1, multiple_tables=False, area=area)
                
                # Check for bad table area height
                postCol = 0
                offset = 20
                tblLen = len(df)
                adjustable = True
                badRegion = True
                while badRegion and adjustable:
                    
                    # Remove nan rows and columns
                    df.dropna(how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    
                    for i, c in enumerate(df.columns):
                        
                        if df[c].str.contains(bPost, regex=True, flags=re.I).any():
                            postCol = i
                            badRegion = False
                            break
                            
                    if badRegion:
                        tblJson = tabula.read_pdf(path, pages=page+1, 
                                                  multiple_tables=False, 
                                                  output_format='json')
                    
                        area = [
                            tblJson[0]['top'], 
                            tblJson[0]['left'], 
                            tblJson[0]['top'] + tblJson[0]['height'] + offset, 
                            tblJson[0]['left'] + tblJson[0]['width'] + wOffset,
                        ]

                        df = tabula.read_pdf(path, pages=page+1, area=area, 
                                             multiple_tables=False)
                    
                        offset += 20
                        
                        if tblLen == len(df):
                            adjustable = False
                
                if postCol != 0:
                    # Rearrange columns
                    cols = df.columns.tolist()                    
                    df = df[[cols[postCol]] + cols[:postCol] + cols[postCol+1:]]
                
                df = fixTbl(df, sheet, valFactor, repDate)
            
            #  Convert numbers to float
            if df[repDate].dtype == object:
                df[repDate] = df[repDate].str.replace('^-$', '0', regex=True)
                df[repDate] = df[repDate].str.replace('^\d\)(?=\(?\d+)', '', regex=True) # Footnotes
                df[repDate] = df[repDate].apply(rpl).astype(float)  
            
            # Factorize
            pattern = '((per( ordinary)? )?share|EPS|shares outstanding|outstanding shares)'
            mask = df.index.str.contains(pattern, regex=True)
            
            # Currency factor
            df.loc[~mask, repDate] *= valFactor
            
            # Share number factor
            pattern = '(shares outstanding|outstanding shares)'
            mask = df.index.str.contains(pattern, regex=True)
            
            sharePost = df.index[mask]
            
            if len(sharePost) > 0:
                
                if re.search('billion', sharePost[0]):
                    shareFactor = 1e9
                
                elif re.search('million', sharePost[0]):
                    shareFactor = 1e6
                    
                elif re.search('thousand', sharePost[0]):
                    shareFactor = 1e3
                    
                else:
                    shareFactor = 1
                    
            else:
                shareFactor = 1
                    
            df.loc[mask, repDate] *= shareFactor
            #df.loc['Currency', repDate] = curr
            
            # Drop duplicate rows
            df.drop_duplicates(inplace=True)

            return df, complete, curr, valFactor
        
        else:
            return None, complete, None, valFactor
        
    # Report date
    date = re.search('\d{4}-(0[1-9]|1[1-2])-([0-2][1-9]|3[0-1])', str(path)).group()
    repDate = dt.strptime(date, '%Y-%m-%d')
    
    # Find sheets
    
    lookup = {
        'income': '(consolidated )?statement[s]? of (comprehensive )?income(?!(-IFRS| for further details))',
        'balance': '((consolidated )?balance sheet[s]?|statement of financial position)(?=.+total current (assets|liabilities))',
        'cash': '(?<!condensed )(cash flow statement|(consolidated )?statement[s]? of cash flows)(?=.+operating activities)'
    }
    
    valFactor = 1
    currs = []
    tbls = []
    miner = False
    for sheet in lookup:
        
    # Open the pdf file   
        with open(path, 'rb') as f:
            
            # PyPdf2
            pdf = PyPdf2.PdfFileReader(f)
            if pdf.isEncrypted:
                miner = True
            
            elif not pdf.isEncrypted and not miner:
    
                # Get number of pages
                pages = pdf.getNumPages()
    
                # Extract text and do the search
                for i in range(pages):
                    page = pdf.getPage(i)
                    txt = page.extractText()
                    txt = re.sub('\s+' , ' ', txt)
                    
                    # Check if PyPdf works
                    if not txt:
                        miner = True
                        break

                    tbl, complete, curr, valFactor = extractTblData(sheet, i, lookup[sheet], txt, repDate, valFactor)
                    
                    if (tbl is not None) and complete:
                        tbls.append(tbl)
      
                        if curr is not None:
                            currs.append(curr)
                        
                        break
            
            # PdfMiner
            if miner:
                
                si = setInterpreter()
                extract = si['extract']
                device = si['device']
                interpreter = si['interpreter']
                
                for i, page in enumerate(PdfPage.get_pages(f, set(), caching=True, check_extractable=True)):
                    interpreter.process_page(page)
                    txt = extract.getvalue().decode('utf-8')
                    txt = re.sub('\s+' , ' ', txt)
                    
                    tbl, complete, curr, valFactor = extractTblData(sheet, i, lookup[sheet], txt, repDate, valFactor)
                    
                    if (tbl is not None) and complete:
                        tbls.append(tbl)
                        
                        if curr is not None:
                            currs.append(curr)
                        
                        break
    
                    si = setInterpreter()
                    extract = si['extract']
                    device = si['device']
                    interpreter = si['interpreter']
        
    result = pd.concat(tbls)
    result = result.T
    result.index.names = ['Date']
    result.loc[repDate, 'Currency'] = currs[0]
            
    return result

def pdfFindPage(path, sheet, startPage=0):
    
    def setInterpreter():
        rm = PdfResourceManager()
        extract = io.BytesIO() #StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rm, extract, codec=codec, laparams=laparams)
        interpreter = PdfPageInterpreter(rm, device)
        return {'extract': extract, 'device': device, 'interpreter': interpreter}
    
    def extractPageFactor(lookup, txt):
        
        if re.search(lookup, txt, re.I | re.M):
            pMatch = []       
            pMatch.append(i+1)
            # Check for splitted balance sheet
            if sheet == 'balance' and not re.search('total liabilities', txt, re.I):
                pMatch.append(i+2)
                
            # Extract value factor and currency
            pattern = '[A-Z]{3}( in )?(thousand|million|billion)'
            vMatch = re.search(pattern, txt).group().split
            curr = vMatch[0]
            
            # Factor
            if vMatch[-1] == 'thousand':
                valFactor = 1e3
            
            elif vMatch[-1] == 'million':
                valFactor = 1e6
                
            elif vMatch[-1] == 'billion':
                valFactor = 1e9
                
            return pMatch, valFactor, curr
        
        else:
            return [], 1
        
    # Lookups
    if sheet == 'income':
        #lookup = '^(?!.*(IFRS|for further details))(consolidated )?statement[s]? of (comprehensive )?income'
        lookup = '(consolidated )?statement[s]? of (comprehensive )?income(?!(-IFRS| for further details))'
        
    elif sheet == 'balance':
        lookup = '((consolidated )?balance sheet[s]?|statement of financial position)(?=.*total assets)'
        
    elif sheet == 'cash':
        lookup = '(?<!condensed )(consolidated )?statement[s]? of cash flow[s]?'
    
    # Open the pdf file   
    with open(path, 'rb') as f:
        
        pMatch = []
        valFactor = 1
        curr = None
        
        # PyPdf2
        pdf = PyPdf2.PdfFileReader(f)
        
        if not pdf.isEncrypted:

            # Get number of pages
            pages = pdf.getNumPages()

            # Extract text and do the search
            for i in range(startPage, pages):
                page = pdf.getPage(i)
                txt = page.extractText()
                txt = re.sub('\s+' , ' ', txt)
                
                pMatch, valFactor, curr = extractPageFactor(lookup, txt)
                
                if pMatch:
                    break
        
        # PdfMiner
        if not pMatch:

            si = setInterpreter()
            extract = si['extract']
            device = si['device']
            interpreter = si['interpreter']
            
            for i, page in enumerate(PdfPage.get_pages(f, set(), caching=True, check_extractable=True)):
                interpreter.process_page(page)
                txt = extract.getvalue().decode('utf-8')
                txt = re.sub('\s+' , ' ', txt)
                
                pMatch, valFactor, curr = extractPageFactor(lookup, txt)
                
                if pMatch:
                    break

                si = setInterpreter()
                extract = si['extract']
                device = si['device']
                interpreter = si['interpreter']

    return pMatch, valFactor, curr

def pdfParseTbl(path):
    
    def fixTbl(df, sheet, valFactor, repDate):
        
        def rpl(x):
            rpl = {' ': '', '*': '', ',': '', '(': '-', ')': ''}

            for i, j in rpl.items():
                x = x.replace(i, j)

            return x
                
        # Remove nan rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
             
        # Fix multi-line rows
        c0 = df.columns[0]
        
        if df[c0].str.islower().any():

            for i in range(len(df[c0])):

                if not pd.isnull(df[c0].iloc[i]):
                    if df[c0].iloc[i].islower():
                        df[c0].iloc[i] = df[c0].iloc[i-1] + ' ' + df[c0].iloc[i]
                        #df.drop(i-1, inplace=True)

        # Remove remaining empty rows
        df.dropna(thresh=len(df.columns[1:]), inplace=True)

        # Remove nan from posts
        mask = df[c0].isna()
        df = df[~mask] #.copy()
        
        # Check for bad column parse in first column
        pattern = '\(?[A-Z][A-Za-z *-,)]+\s+(-|\(\s?)?(\d{1,3}[, ]?)+(\.\d+)?(\s?\))?'
        if df[c0].str.contains(pattern, regex=True).any():
            pattern = '(?P<extract>(-|\(\s?)?(\d{1,3}[, ]?)+(\.\d+)?(\s?\))?)'
            temp = df[c0].str.extract(pattern, expand=False)
            pattern = '(-|\(\s?)?(\d{1,3}[, ]?)+(\.\d+)?(\s?\))?'
            post = df[c0].str.replace(pattern, '', regex=True)
            df = pd.DataFrame({'Post': post, 'extract': temp['extract']})
            df.dropna(inplace=True)
            df.rename(columns={'extract': repDate}, inplace=True)
            # Set posts as index
            df.set_index('Post', inplace=True)
        
        else:
            
            # Set posts as index
            df.set_index(df.columns[0], inplace=True)
            df.index.rename('Post', inplace=True)

            # Delete note column
            c0 = df.columns[0]
            if df[c0].str.contains('Note', regex=True).any():
                df.drop(c0, axis=1, inplace=True)
                c0 = df.columns[0]
        
            # Remove non-number row
            pattern = '(-|\(\s?)?(\d{1,3}[, ]?)+(\.\d+)?(\s?\))?'
            mask = df[c0].str.contains(pattern, regex=True)
            df = df[mask]
        
            # Check bad column parse
            pattern += '\s+'
            dfSplit = None
            if df[c0].str.contains(pattern, regex=True).any():

                # Reduce to single whitespaces to allow easy split
                df.loc[:, c0] = df[c0].str.replace(' +', ' ', regex=True)

                # Count whitespaces
                spaces = df[c0].str.count(' ').min()

                # Split columns
                dfSplit = df[c0].str.rsplit(' ', spaces, expand=True)
        
            # Keep only latest quarter figures
            if dfSplit is not None:

                df = dfSplit.iloc[:, 0]
                #df = dfSplit[dfSplit.columns[0]]

            else:
                df = df.iloc[:, 0]
                
            # Convert to dataframe
            df = df.to_frame(repDate)
        
        #  Convert numbers to float
        if df[repDate].dtype != np.float64:
            df[repDate] = df[repDate].str.replace('^-$', '0', regex=True)
            df[repDate] = df[repDate].apply(rpl).astype(float)  
        
        # Factorize
        pattern = '(earnings per share|EPS|shares outstanding|outstanding shares)'
        mask = df.index.str.contains(pattern, regex=True)
        
        # Currency factor
        df.loc[~mask, repDate] *= valFactor
        
        # Share number factor
        pattern = '(shares outstanding|outstanding shares)'
        mask = df.index.str.contains(pattern, regex=True)
        
        sharePost = df.index[mask]
        
        if len(sharePost) > 0:
            
            if re.search('billion', sharePost[0]):
                shareFactor = 1e9
            
            elif re.search('million', sharePost[0]):
                shareFactor = 1e6
                
            elif re.search('thousand', sharePost[0]):
                shareFactor = 1e3
                
            else:
                shareFactor = 1
                
        else:
            shareFactor = 1
                
        df.loc[mask, repDate] *= shareFactor
        
        return df
        
    # Report date
    date = re.search('\d{4}-(0[1-9]|1[1-2])-([0-2][1-9]|3[0-1])', str(path)).group()
    repDate = dt.strptime(date, '%Y-%m-%d')
    
    # Parse tables
    dfs = [None] * 3
    for i, sheet in enumerate(['income', 'balance', 'cash']):
        
        pages, valFactor, curr = pdfFindPage(path, sheet)
        
        if pages:
            
            # Parse table
            df = tabula.read_pdf(path, pages=pages, multiple_tables=False)
            
            # Check for bad area
            goodParse = False
            pattern = '\(?(unaudited, )?(in )?[A-Z]{3}( in)? (thousand|million|billion)\)?'
            offset = 10
            top = None
            postCol = 0
            while not goodParse:
                
                # Remove nan rows and columns
                df.dropna(how='all', inplace=True)
                df.dropna(axis=1, how='all', inplace=True)
                                
                for j, c in enumerate(df.columns):
                                        
                    cond = (
                        re.search(pattern, c) or 
                        re.search(pattern, df[c].dropna().astype(str).iloc[0])
                    )
                    if cond:
                        
                        postCol = j
                        goodParse = True
                        break
                    
                if not goodParse:
                    
                    tbl = tabula.read_pdf(path, pages=pages, 
                                          multiple_tables=False, 
                                          output_format='json')
                    top = tbl[0]['top'] - offset
                    area = [
                        top, 
                        tbl[0]['left'], 
                        tbl[0]['top'] + tbl[0]['height'] + offset, 
                        tbl[0]['left'] + tbl[0]['width']
                    ]
                    
                    df = tabula.read_pdf(path, pages=pages, area=area, 
                                         multiple_tables=False)
                    
                    offset += 10
            
            # Check if balance sheet is complete
            if sheet == 'balance':
                
                c0 = df.columns[0]
                offset = 20
                while not df[c0].dropna().str.contains('total (equity|liabilities) and (liabilities|equity)', regex=True, flags=re.I).any():
                    
                    if top is not None:
                        
                        area = [
                            top, 
                            tbl[0]['left'], 
                            top + tbl[0]['height'] + offset, 
                            tbl[0]['left'] + tbl[0]['width']
                        ]
                        
                    else: 
                        tbl = tabula.read_pdf(path, pages=pages, 
                                              multiple_tables=False, 
                                              output_format='json')
                        area = [
                            tbl[0]['top'], 
                            tbl[0]['left'], 
                            tbl[0]['top'] + tbl[0]['height'] + offset, 
                            tbl[0]['left'] + tbl[0]['width']
                        ]
                        
                    df = tabula.read_pdf(path, pages=pages, area=area, 
                                         multiple_tables=False)
                    offset += 20
            
            # Check if sheet posts are not in leftmost column
            if postCol != 0:
                
                df.columns = ([df.columns[postCol]] + df.columns[:postCol-1] + 
                              df.columns[postCol:])
                        
            dfs[i] = fixTbl(df, sheet, valFactor, repDate)
            
        else:
            
            dfs[i] = None
    
    financials = pd.concat(dfs)
    #financials.drop_duplicates(inplace=True)
    financials = financials.T
    financials.index.names = ['Date']
    financials.loc[repDate, 'Currency'] = curr 
     
    return financials