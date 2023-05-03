import pandas as pd
from sqlalchemy import create_engine, text

from lib.const import DB_DIR

db_path = DB_DIR / 'ticker.db'
engine = create_engine(f'sqlite+pysqlite:///{db_path}')

def search(ticker):
  pass

def search_tickers(
  security: str, 
  search: str, 
  href: bool = True, 
  limit: int = 10
) -> pd.DataFrame:
  db_path = DB_DIR / 'ticker.db'
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  if security == 'stock':
    value = (f'"/{security}/" || id AS href' if href 
      else 'id || "|" || currency AS value'
    )

    query = f'''
      SELECT 
        name || " (" || ticker || ") - "  || mic AS label,
        {value}
      FROM {security} WHERE label LIKE "%{search}%"
      LIMIT {limit}
    '''

  with engine.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(text(query), con=con)
    
  return df