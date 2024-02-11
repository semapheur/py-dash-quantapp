from pathlib import Path
import re
import sqlite3
from typing import cast, Literal, Optional

import pandas as pd
from pandas._typing import DtypeArg
from pandera.typing import DataFrame
from sqlalchemy import (
  create_engine,
  inspect,
  text,
  Engine,
  TextClause,
)  # event

from lib.const import DB_DIR


def sqlite_path(db_name: str) -> Path:
  if not db_name.endswith('.db'):
    db_name += '.db'

  db_path: Path = DB_DIR / db_name

  return db_path


def empty_tables(db_name: str) -> list[str]:
  db_path = sqlite_path(db_name)

  con = sqlite3.connect(db_path)
  cur = con.cursor()

  cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
  tables = cur.fetchall()

  result: list[str] = []
  for table in tables:
    table_name = table[0]
    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    row_count = cur.fetchone()[0]

    if row_count == 0:
      result.append(table[0])

  con.close()

  return result


def sql_table(query: str):
  pattern = r'\bFROM (\'|")?(\w+?)(\'|")?\b'
  m = re.search(pattern, query)
  if m is None:
    raise ValueError(f'Could not parse table name from the query: {query}')

  return m.group(2)


def get_tables(db_name: str) -> list[str]:
  db_path = sqlite_path(db_name)

  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  return insp.get_table_names()


def check_table(tables: str | set[str], engine: Engine) -> bool:
  db_tables = inspect(engine).get_table_names()

  if not db_tables:
    return False

  if isinstance(tables, str):
    tables = {tables}

  return tables.issubset(set(db_tables))


def sqlite_vacuum(db_name: str):
  db_path = sqlite_path(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  with engine.connect().execution_options(autocommit=True) as con:
    con.execute(text('VACUUM'))


# @event.listens_for(Engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute('PRAGMA optimize')
  cursor.execute('PRAGMA journal_mode=WAL')
  cursor.execute('PRAGMA synchronous=normal')
  cursor.execute('PRAGMA auto_vacuum=INCREMENTAL')
  # cursor.execute('PRAGMA mmap_size=30000000000')
  # cursor.execute('PRAGMA page_size=32768')
  cursor.close()


def read_sqlite(
  db_name: str,
  query: str | TextClause,
  params: Optional[dict[str, str]] = None,
  index_col: Optional[str | list[str]] = None,
  dtype: Optional[DtypeArg] = None,
  date_parser: Optional[dict[str, dict[str, str]]] = None,
) -> DataFrame | DataFrame[T] | None:
  db_path = sqlite_path(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  table = sql_table(str(query))
  tables = insp.get_table_names()

  if tables == [] or table not in set(tables):
    return None

  if isinstance(query, str):
    query = text(query)

    if params:
      query = query.bindparams(**params)

  with engine.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(
      query, con=con, parse_dates=date_parser, index_col=index_col, dtype=dtype
    )

  return None if df.empty else cast(DataFrame, df)


def insert_sqlite(
  df: pd.DataFrame,
  db_name: str,
  tbl_name: str,
  action: Literal['merge', 'replace'] = 'merge',
  index: bool = True,
) -> None:
  db_path = sqlite_path(db_name)
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
      query, con=engine, parse_dates={'date': {'format': '%Y-%m-%d'}}, index_col=ix
    )

  df_old = df_old.combine_first(df)
  diff_cols = df.columns.difference(df_old.columns).tolist()

  if diff_cols:
    df_old = df_old.join(df[diff_cols], how='outer')

  df_old.to_sql(tbl_name, con=engine, if_exists='replace', index=index)


def upsert_sqlite(df: pd.DataFrame, db_name: str, tbl_name: str):
  db_path = sqlite_path(db_name)
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')
  insp = inspect(engine)

  # SQL index headers
  ix_cols = list(df.index.names)
  ix_cols_text = ', '.join(f'"{i}"' for i in ix_cols)

  if not insp.has_table(tbl_name):
    df.to_sql(tbl_name, con=engine, index=True)

    # Create index
    with engine.begin() as con:
      con.execute(text(f'CREATE UNIQUE INDEX ix ON "{tbl_name}" ({ix_cols_text})'))

    return

  with engine.begin() as con:
    # SQL header query text
    cols = list(df.columns.tolist())
    headers = ix_cols + cols
    headers_text = ', '.join(f'"{i}"' for i in headers)
    update_text = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in cols])

    # Store data in temporary table
    # con.execute(text(f'CREATE TEMP TABLE temp({headers_text})'))

    df.to_sql('temp', con=con, if_exists='replace', index=True)

    # Check if new columns must be inserted
    result = con.execute(text(f'SELECT * FROM "{tbl_name}"'))
    old_cols = set([c for c in result.keys()]).difference(set(ix_cols))
    new_cols = list(set(cols).difference(old_cols))
    if new_cols:
      for c in new_cols:
        con.execute(text(f'ALTER TABLE "{tbl_name}" ADD COLUMN {c}'))

    # Upsert data to main table
    query = text(
      f"""
      INSERT INTO "{tbl_name}" ({headers_text})
      SELECT {headers_text} FROM temp WHERE true
      ON CONFLICT ({ix_cols_text}) DO UPDATE 
      SET {update_text}
    """
    )

    con.execute(
      text(f'CREATE UNIQUE INDEX IF NOT EXISTS ix ON "{tbl_name}" ({ix_cols_text})')
    )
    con.execute(query)
    # con.execute(text('DROP TABLE temp')) # Delete temporary table


def replace_sqlite(col: str, replacements: dict[str, str]) -> str:
  old, new = replacements.popitem()
  query = f'REPLACE({col}, "{old}", "{new}")'

  for old, new in replacements.items():
    query = f'REPLACE({query}, "{old}", "{new}")'

  return query


def json_query(schema: dict[str, Literal['patch', 'unique']]) -> str:
  query: list[str] = []
  for col, fn in schema.items():
    if fn == 'patch':
      query.append(f'"{col}"=json_patch("{col}", excluded."{col}")')
    elif fn == 'unique':
      query.append(
        f""""{col}"=(
        SELECT json_group_array(value)
        FROM (
          SELECT json_each.value
          FROM json_each("{col}")
          WHERE json_each.value IN (SELECT json_each.value FROM json_each(excluded."{col}"))
        )
      )"""
      )

  return ','.join(query)


def upsert_json(
  db_name: str,
  table: str,
  fields: dict[str, str],
  ix: list[str],
  json_cols: dict[str, Literal['patch', 'unique']],
  records: list[tuple],
):
  db_path = sqlite_path(db_name)

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    fields_text = ','.join([' '.join((k, v)) for k, v in fields.items()])

    cur.execute(f"""CREATE TABLE IF NOT EXISTS "{table}"({fields_text})""")

    ix_text = ','.join(ix)
    cur.execute(
      f"""CREATE UNIQUE INDEX IF NOT EXISTS ix 
      ON "{table}"({ix_text})"""
    )

    columns = ','.join(tuple(fields.keys()))
    values = ','.join(['?'] * len(columns))

    upsert_text = json_query(json_cols)

    query = f"""INSERT INTO "{table}"({columns}) VALUES ({values})
      ON CONFLICT ({ix_text}) DO UPDATE SET 
        {upsert_text}
    """
    cur.executemany(query, records)
