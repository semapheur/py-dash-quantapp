from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
import json
from typing import Awaitable, Optional, TypedDict

import pandas as pd
from sqlalchemy import create_engine, text

from lib.const import DB_DIR
from lib.db.lite import check_table, read_sqlite, upsert_sqlite
from lib.edgar.company import Company
from lib.fin.calculation import calculate_items

db_path = DB_DIR / 'ticker.db'
ENGINE = create_engine(f'sqlite+pysqlite:///{db_path}')

class Stock(TypedDict, total=False):
  id: str
  ticker: str
  name: str
  mic: str
  currency: str
  sector: str
  industry: str

def stock_label(id: str) -> str:
  if not check_table('stock', ENGINE):
    return None

  query = text('''
    SELECT name || " (" || ticker || ":" exchange || ")" AS label 
    FROM stock WHERE id = ":id"
  ''').bindparams(id=id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)
  
  return fetch.first()[0]

def fetch_stock(id: str, cols: Optional[set] = None) -> Optional[Stock]:
  
  if not check_table('stock', ENGINE):
    return None

  cols = (
    set(Stock.__optional_keys__) if cols is None 
    else set(Stock.__optional_keys__).intersection(cols)
  )
  if not cols:
    raise Exception(f'Columns must be from {Stock.__optional_keys__}')

  query = text(f'SELECT {",".join(cols)} FROM stock WHERE id = ":id"').bindparams(id=id)

  with ENGINE.begin() as con:
    cursor = con.execute(query)
    fetch = cursor.first()
  
  if fetch:
    return {c: f for c, f in zip(cols, fetch)}

def find_cik(id: str) -> Optional[int]:
  if not check_table({'stock', 'edgar'}, ENGINE):
    return None

  query = text('''
    SELECT  
      edgar.cik AS cik FROM stock, edgar
    WHERE 
      stock.id = :id AND 
      REPLACE(edgar.ticker, "-", "") = REPLACE(stock.ticker, ".", "")
  ''').bindparams(id=id)

  with ENGINE.begin() as con:
    cursor = con.execute(query)
    fetch = cursor.first()

  if fetch:
    return fetch[0]

def search_tickers(
  security: str, 
  search: str, 
  href: bool = True, 
  limit: int = 10
) -> pd.DataFrame:
  
  if security == 'stock':
    value = (f'"/{security}/" || id AS href' if href 
      else 'id || "|" || currency AS value'
    )
    query = text(f'''
      SELECT 
        name || " (" || ticker || ") - "  || mic AS label,
        {value}
      FROM {security} WHERE label LIKE :search
      LIMIT {limit}
    ''').bindparams(search=f'%{search}%')

  with ENGINE.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(query, con=con)
    
  return df

def get_ohlcv(
  id: str, 
  security: str,
  ohlcv_fetcher: partial[pd.DataFrame | None],
  delta: int = 1,
  cols: Optional[set[str]] = None
) -> pd.DataFrame:
  
  if not cols:
    cols = {'open', 'high', 'low', 'close', 'volume'}
  
  query = f'SELECT {", ".join(cols.add("date"))} FROM "{id}"'
  ohlcv = read_sqlite(f'{security}_quote.db', query, 
    index_col='date', 
    parse_dates=True
  )

  if ohlcv is None:
    ohlcv = ohlcv_fetcher()
    upsert_sqlite(ohlcv, f'{security}_quote.db', id)

    return ohlcv[list(cols)]
  
  last_date = ohlcv.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return ohlcv

  new_ohlcv = ohlcv_fetcher(last_date.strftime('%Y-%m-%d'))

  if new_ohlcv is None:
    return ohlcv

  upsert_sqlite(ohlcv, f'{security}_quote.db', id)
  ohlcv = read_sqlite(f'{security}_quote.db', query, 
    index_col='date', 
    parse_dates=True
  )

  return ohlcv

def load_schema(query: Optional[str] = None) -> dict[str, dict]:
  if query is None:
    query = '''
      SELECT item, calculation FROM items
      WHERE calculation IS NOT NULL
    '''
  
  df = read_sqlite('taxonomy.db', query)
  df.loc[:,'calculation'] = (
    df['calculation'].apply(lambda x: json.loads(x))
  )
  schema = {k: v for k, v in zip(df['item'], df['calculation'])}
  return schema


def calculate_fundamentals(
  financials: pd.DataFrame,
  ohlcv_fetcher: partial[pd.DataFrame]
) -> pd.DataFrame:
  
  price = get_ohlcv(id, 'stock', ohlcv_fetcher, cols={'date', 'close'})
  price.rename(columns={'close': 'share_price'}, inplace=True)
  price = price.resample('D').ffill()

  financials.reset_index(inplace=True)
  financials = (financials
    .reset_index()
    .merge(price, how='left', on='date')
    .set_index(['date', 'period', 'months']) 
  )
  schema = load_schema()
  financials = calculate_items(financials, schema)

  return financials

async def get_fundamentals(
  id: str,
  financials_fetcher: partial[Awaitable[pd.DataFrame]],
  ohlcv_fetcher: partial[pd.DataFrame],
  cols: Optional[set[str]] = None,
  delta: int = 120
) -> pd.DataFrame:

  col_text = '*'
  index_col = {'date', 'period', 'months'}
  if cols is not None:
    col_text = ", ".join(cols.union(index_col))

  query = f'SELECT {col_text} FROM "{id}"'
  df = read_sqlite('fundamentals.db', query, 
    index_col=list(index_col), 
    parse_dates=True
  )

  if df is None:
    df = await financials_fetcher()
    df = calculate_fundamentals(df, ohlcv_fetcher)
    upsert_sqlite(df, 'fundamentals.db', id)

  last_date = df.index.get_level_values('date').max()
  if relativedelta(dt.now(), last_date).days <= delta:
    return df

  _df = financials_fetcher(last_date.strftime('%Y-%m-%d'))

  if _df is None:
    return _df
  
  _df = calculate_fundamentals(_df, ohlcv_fetcher)
  upsert_sqlite(_df, 'fundamentals.db', id)
  df = read_sqlite('fundamentals.db', query, 
    index_col=index_col, 
    parse_dates=True
  )

  return df