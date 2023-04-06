from datetime import datetime as dt

import pandas as pd
import wbgapi as wb

from lib.db import read_sqlite, upsert_sqlite

SERIES = {
	'NY.GDP.MKTP.CD': 'gdp_cd', # GDP (current &)
	'NY.GDP.PCAP.PP.CD': 'gdpc_ppp_cd', # GDP per capita, PPP (current $)
	'NY.GDP.PCAP.KD.ZG': 'gdpcg', # GDP per capita growth (annual %)
	'NY.GDP.MKTP.KD.ZG': 'gdpg', # GDP growth (annual %)
	'SP.POP.TOTL': 'pop', # Population (total)
	'FP.CPI.TOTL.ZG': 'cpi' # CPI (annual %)
}

def fetch_wdi(end_year=0, start_year=1960) -> pd.DataFrame:
	if not end_year:
		end_year = dt.today().year

	df = wb.data.DataFrame(
		list(SERIES.keys()), 
		time=range(start_year, end_year)
	)
	df.index.rename(['country', 'series'], inplace=True)
	df = df.T
	df.index.set_names('year', inplace=True)
	df.index = df.index.str.replace('YR', '').astype(int)
	df = df.stack(level=0)
	df.rename(columns=SERIES, inplace=True)

	upsert_sqlite(df, 'macro.db', 'wdi')

	return df

def load_wdi(cols: str|list[str]='*', years=0, delta=2) -> pd.DataFrame:
	
	if isinstance(cols, list):
		cols = ','.join(cols)

	where = ''
	if years:
		where = f' WHERE year > (SELECT MAX(year) FROM wdi) - {years}'

	query = f'SELECT {cols} FROM wdi' + where
	df = read_sqlite(query, 'macro.db', ['year', 'country'])

	this_year = dt.today().year

	if df is None:
		df = fetch_wdi()
		return df.unstack(level=1)

	df = df.unstack(level=1)

	end_year = df.index[-1]
	if this_year - end_year > delta:
		df = fetch_wdi(this_year)
		df = df.unstack(level=1)

	return df