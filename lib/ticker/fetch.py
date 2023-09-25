from typing import Optional, TypedDict

import pandas as pd
from sqlalchemy import create_engine, text

from lib.const import DB_DIR
from lib.db.lite import check_table

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
    FROM stock WHERE id = :id
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

  query = text(f'SELECT {",".join(cols)} FROM stock WHERE id = :id').bindparams(id=id)

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