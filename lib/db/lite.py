from typing import Literal, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text, Engine, TextClause #event

from lib.const import DB_DIR

def check_table(tables: str|set[str], engine: Engine) -> bool:
  db_tables = inspect(engine).get_table_names()

  if not db_tables:
    return False
  
  if isinstance(tables, str):
    tables = set(tables)

  return tables.issubset(set(db_tables))

def sqlite_name(db_name: str) -> str:
  if not db_name.endswith('.db'):
    return db_name + '.db'

  return db_name

def sqlite_vacuum(db_name: str):
  db_path = DB_DIR / sqlite_name(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  with engine.connect().execution_options(autocommit=True) as con:
    con.execute(text('VACUUM'))

#@event.listens_for(Engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute('PRAGMA optimize')
  cursor.execute('PRAGMA journal_mode=WAL')
  cursor.execute('PRAGMA synchronous=normal')
  cursor.execute('PRAGMA auto_vacuum=INCREMENTAL')
  #cursor.execute('PRAGMA mmap_size=30000000000')
  #cursor.execute('PRAGMA page_size=32768')
  cursor.close()

def read_sqlite(
  db_name: str,
  query: str|TextClause,
  params: Optional[dict[str, str]] = None,
  index_col: Optional[str | list[str]] = None,
  parse_dates=False,
) -> pd.DataFrame | None:
  
  db_path = DB_DIR / sqlite_name(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  if not insp.get_table_names():
    return None
  
  parser = None
  if parse_dates:
    parser = {'date': {'format': '%Y-%m-%d %H:%M:%S.%f'}}

  if isinstance(query, str):
    query = text(query)

    if params:
      query = query.bindparams(**params) 

  with engine.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(query, con=con, parse_dates=parser, index_col=index_col)

  return df

def insert_sqlite(
  df: pd.DataFrame, 
  db_name: str, 
  tbl_name: str,
  action: Literal['merge', 'replace'] = 'merge',
  index: bool = True
) -> None:

  db_path = DB_DIR / sqlite_name(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  if not insp.has_table(tbl_name):
    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql(tbl_name, con=con, index=index)
    return

  if action == 'replace':
    df.to_sql(tbl_name, con=engine, if_exists='replace', index=index)
    return

  query = text(f'SELECT * FROM "{tbl_name}"')
  ix = list(df.index.names)

  with engine.connect().execution_options(autocommit=True) as con:
    df_old = pd.read_sql(
      query, con=engine, 
      parse_dates={'date': {'format': '%Y-%m-%d'}}, 
      index_col=ix
    )

  df_old = df_old.combine_first(df)
  diff_cols = df.columns.difference(df_old.columns).tolist()

  if diff_cols:
    df_old = df_old.join(df[diff_cols], how='outer')

  df_old.to_sql(tbl_name, con=engine, if_exists='replace', index=index)

def upsert_sqlite(df: pd.DataFrame, db_name: str, tbl_name: str):

  db_path = DB_DIR / sqlite_name(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  # SQL index headers
  ix_cols = list(df.index.names)
  ix_cols_text = ', '.join(f'"{i}"' for i in ix_cols)

  if not insp.has_table(tbl_name):
    df.to_sql(tbl_name, con=engine, index=True)

    # Create index
    with engine.begin() as con:
      con.execute(text(f'CREATE UNIQUE INDEX ix ON {tbl_name} ({ix_cols_text})'))

    return

  with engine.begin() as con:
      
    # SQL header query text
    cols = list(df.columns.tolist())
    headers = ix_cols + cols
    headers_text = ', '.join(f'"{i}"' for i in headers)
    update_text = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in cols])

    # Store data in temporary table
    #con.execute(text(f'CREATE TEMP TABLE temp({headers_text})'))

    df.to_sql('temp', con=con, if_exists='replace', index=True)
    
    # Check if new columns must be inserted
    result = con.execute(f'SELECT * FROM "{tbl_name}"')
    old_cols = set([c for c in result.keys()]).difference(set(ix_cols))
    new_cols = list(set(cols).difference(old_cols))
    if new_cols:
      for c in new_cols:
        con.execute(text(f'ALTER TABLE "{tbl_name}" ADD COLUMN {c}'))

    # Upsert data to main table
    query = text(f'''
      INSERT INTO "{tbl_name}" ({headers_text})
      SELECT {headers_text} FROM temp WHERE true
      ON CONFLICT ({ix_cols_text}) DO UPDATE 
      SET {update_text}
    ''')

    con.execute(text(
      f'CREATE UNIQUE INDEX IF NOT EXISTS ix ON "{tbl_name}" ({ix_cols_text})'
    ))
    con.execute(query)
    #con.execute(text('DROP TABLE temp')) # Delete temporary table

def replace_sqlite(col: str, replacements: dict[str, str]) -> str:
  old, new = replacements.popitem()
  query = f'REPLACE({col}, "{old}", "{new}")'

  for old, new in replacements.items():
    query = f'REPLACE({query}, "{old}", "{new}")'

  return query