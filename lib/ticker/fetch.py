from typing import Optional, TypedDict

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from lib.const import DB_DIR

class Stock(TypedDict, total=False):
  id: str
  ticker: str
  name: str
  mic: str
  currency: str
  sector: str
  industry: str

db_path = DB_DIR / 'ticker.db'
engine = create_engine(f'sqlite+pysqlite:///{db_path}')

def check_table(tables: str|set[str], engine) -> bool:
  db_tables = inspect(engine).get_table_names()
  if not db_tables:
    return False
  
  return set(tables).issubset(db_tables)

def stock_label(id: str) -> str:
  if not check_table('stock', engine):
    return None

  query = f'''
    SELECT name || " (" || ticker || ":" exchange || ")" AS label 
    FROM stock WHERE id = "{id}"
  '''

  with engine.begin() as con:
    fetch = con.execute(text(query))
    return fetch.first()[0]

def fetch_stock(id: str, cols: Optional[set] = None) -> Optional[Stock]:
  
  if not check_table('stock', engine):
    return None

  cols = (
    set(Stock.__optional_keys__) if cols is None 
    else set(Stock.__optional_keys__).intersection(cols)
  )
  if not cols:
    raise Exception(f'Columns must be from {Stock.__optional_keys__}')

  query = f'SELECT {",".join(cols)} FROM stock WHERE id = "{id}"'

  with engine.begin() as con:
    cursor = con.execute(text(query))
    fetch = cursor.first()
  
  if fetch:
    return {c: f for c, f in zip(cols, fetch)}

def find_cik(id: str) -> Optional[int]:
  if not check_table({'stock', 'cik'}, engine):
    return None

  query = f'''
    SELECT  
      cik.cik AS cik FROM stock, cik
    WHERE stock.id = "{id}" AND cik.ticker = stock.ticker
  '''

  with engine.begin() as con:
    cursor = con.execute(text(query))
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

  with engine.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(query, con=con)
    
  return df