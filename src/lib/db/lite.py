from contextlib import closing
from pathlib import Path
import re
import sqlite3
from typing import Sequence, cast, Any, Literal, Optional

from ordered_set import OrderedSet
import pandas as pd
from pandas._typing import DtypeArg
from pandera.typing import DataFrame
import polars as pl
from sqlalchemy import (
  create_engine,
  inspect,
  text,
  TextClause,
)
from sqlalchemy.types import Integer, Text, Float, DateTime

from lib.const import DB_DIR

_POLARS_TO_SQLITE: dict[pl.DataType, str] = {
  pl.Int8: "INTEGER",
  pl.Int16: "INTEGER",
  pl.Int32: "INTEGER",
  pl.Int64: "INTEGER",
  pl.UInt8: "INTEGER",
  pl.UInt16: "INTEGER",
  pl.UInt32: "INTEGER",
  pl.UInt64: "INTEGER",
  pl.Float32: "REAL",
  pl.Float64: "REAL",
  pl.Boolean: "INTEGER",  # stored as 0/1
  pl.Utf8: "TEXT",
  pl.Date: "TEXT",
  pl.Datetime: "TEXT",
  pl.List: "BLOB",
  pl.Struct: "BLOB",
}


def sqlite_path(db_name: str) -> Path:
  if not db_name.endswith(".db"):
    db_name += ".db"

  db_path: Path = DB_DIR / db_name

  return db_path


def dict_factory(cursor, row):
  fields = [column[0] for column in cursor.description]
  return {key: value for key, value in zip(fields, row)}


def empty_tables(db_name: str) -> list[str]:
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
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

  return result


def sql_table(query: str):
  pattern = r'\bFROM (\'|")?(\w+?)(\'|")?\b'
  m = re.search(pattern, query)
  if m is None:
    raise ValueError(f"Could not parse table name from the query: {query}")

  return m.group(2)


def get_tables(db_name: str) -> list[str]:
  db_path = sqlite_path(db_name)

  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  insp = inspect(engine)

  return insp.get_table_names()


def check_table(tables: set[str], db_name: str) -> bool:
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  db_tables = inspect(engine).get_table_names()

  if not db_tables:
    return False

  return tables.issubset(set(db_tables))


def sqlite_vacuum(db_name: str, backup_file: str | None = None):
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")

  query = "VACUUM"
  if backup_file is not None:
    query += f" INTO '{backup_file}'"

  with engine.connect().execution_options(autocommit=True) as con:
    con.execute(text(query))


# @event.listens_for(Engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA optimize")
  cursor.execute("PRAGMA journal_mode=WAL")
  cursor.execute("PRAGMA synchronous=normal")
  cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
  # cursor.execute('PRAGMA mmap_size=30000000000')
  # cursor.execute('PRAGMA page_size=32768')
  cursor.close()


def sqlite_dtypes(df: pd.DataFrame) -> dict[str, Any]:
  dtype_mapping = {
    "int64": Integer,
    "float64": Float,
    "bool": Text,
    "datetime64[ns]": DateTime,
    "timedelta64[ns]": Integer,
    "object": Text,
  }

  result: dict[str, Any] = {}

  if isinstance(df.index, pd.MultiIndex):
    for ix, dtype in zip(df.index.names, df.index.dtypes):
      result[ix] = dtype_mapping[str(dtype)]
  else:
    index_name = df.index.name if df.index.name is not None else "index"
    sqlite_dtype = dtype_mapping[str(df.index.dtype)]
    # if df.index.is_unique:
    #  sqlite_dtype += ' PRIMARY KEY'

    result[index_name] = sqlite_dtype

  for col, dtype in zip(df.columns, df.dtypes):
    result[col] = dtype_mapping[str(dtype)]

  return result


def fetch_sqlite(db_name: str, query: str, params: Optional[dict[str, str]] = None):
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  insp = inspect(engine)

  table = sql_table(query)
  tables = insp.get_table_names()

  if tables == [] or table not in set(tables):
    return None

  query_text = text(query)
  if params:
    query_text = query_text.bindparams(**params)

  with engine.begin() as con:
    fetch = con.execute(query_text).fetchall()

  return fetch


def read_sqlite(
  db_name: str,
  query: str | TextClause,
  params: dict[str, str] | None = None,
  index_col: str | list[str] | None = None,
  dtype: DtypeArg | None = None,
  date_parser: dict[str, dict[str, str]] | None = None,
) -> DataFrame | None:
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  insp = inspect(engine)

  table = sql_table(str(query))
  tables = insp.get_table_names()

  if tables == [] or table not in set(tables):
    return None

  if params:
    if isinstance(query, str):
      query = text(query)

    query = query.bindparams(**params)

  with engine.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(
      query, con=con, parse_dates=date_parser, index_col=index_col, dtype=dtype
    )

  return None if df.empty else cast(DataFrame, df)


def polars_from_sqlite(
  db_name: str,
  query: str | TextClause,
  params: dict[str, str] | None = None,
  column_transform: list[pl.Expr] | None = None,
) -> pl.DataFrame | None:
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  insp = inspect(engine)

  table = sql_table(str(query))
  tables = insp.get_table_names()

  if tables == [] or table not in set(tables):
    return None

  if params:
    if isinstance(query, str):
      query = text(query)

    query = query.bindparams(**params)

  df = pl.read_database(query=query, connection=engine.connect())

  if column_transform:
    df = df.with_columns(column_transform)

  return df


def select_sqlite(
  db_name: str,
  table: str,
  columns: OrderedSet[str] | None = None,
  index_columns: list[str] = [],
  where: str = "",
) -> DataFrame | None:
  if columns is None:
    query = f"SELECT * FROM '{table}' {where}".strip()
    df = read_sqlite(
      db_name,
      query,
      index_col=index_columns if index_columns else None,
      date_parser={"date": {"format": "%Y-%m-%d"}},
    )
    return df

  table_columns = get_table_columns(db_name, [table])
  select_columns = columns.intersection(table_columns[table])
  if not select_columns:
    return None

  column_text = ", ".join(select_columns.union(index_columns))

  query = f"SELECT {column_text} FROM '{table}' {where}".strip()
  df = read_sqlite(
    db_name,
    query,
    index_col=index_columns if index_columns else None,
    date_parser={"date": {"format": "%Y-%m-%d"}},
  )
  return df


def get_table_columns(
  db_name: str, tables: Optional[list[str]] = None
) -> dict[str, set[str]]:
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")

  inspector = inspect(engine)

  if tables is None:
    tables = inspector.get_table_names()

  table_columns = {
    table: {col["name"] for col in inspector.get_columns(table)} for table in tables
  }
  return table_columns


def insert_sqlite(
  df: pd.DataFrame,
  db_name: str,
  table: str,
  action: Literal["merge", "replace"] = "merge",
  index: bool = True,
  dtype: Optional[DtypeArg] = None,
) -> None:
  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  inspector = inspect(engine)

  if not inspector.has_table(table):
    with engine.connect().execution_options(autocommit=True) as con:
      df.to_sql(table, con=con, index=index, dtype=dtype)
    return

  if action == "replace":
    df.to_sql(table, con=engine, if_exists="replace", index=index, dtype=dtype)
    return

  query = text(f'SELECT * FROM "{table}"')
  ix = list(df.index.names)

  with engine.connect().execution_options(autocommit=True) as con:
    df_old = pd.read_sql(
      query, con=engine, parse_dates={"date": {"format": "%Y-%m-%d"}}, index_col=ix
    )

  df_old = df_old.combine_first(df)
  diff_cols = df.columns.difference(df_old.columns).tolist()

  if diff_cols:
    df_old = df_old.join(df[diff_cols], how="outer")

  df_old.to_sql(table, con=engine, if_exists="replace", index=index)


def polars_sqlite_table(
  df: pl.DataFrame, table: str, primary_keys: Sequence[str] | None = None
) -> str:
  if primary_keys is not None:
    missing = set(primary_keys).difference(df.columns)
    if missing:
      raise ValueError(f"Index columns {missing} not found in DataFrame")

  col_defs: list[str] = []
  for name, dtype in zip(df.columns, df.dtypes):
    sqlite_type = _POLARS_TO_SQLITE.get(dtype, "BLOB")
    col_defs.append(f'"{name}" {sqlite_type}')

  if primary_keys is not None:
    cols = ", ".join(f"'{c}'" for c in primary_keys)
    col_defs.append(f"PRIMARY KEY ({cols})")

  return f"CREATE TABLE '{table}' ({', '.join(col_defs)});"


def combine_polars(
  df1: pl.DataFrame, df2: pl.DataFrame, index_cols: Sequence[str]
) -> pl.DataFrame:
  if not index_cols:
    raise ValueError("Index columns required for combining DataFrames")

  index_set = set(index_cols)
  for i, df in enumerate([df1, df2]):
    missing = index_set.difference(df.columns)
    if missing:
      raise ValueError(f"Missing index columns in DataFrame {i}: {missing}")

  joined = df1.join(df2, on=list(index_cols), how="outer", suffix="_new").lazy()
  df1_cols = set(df1.columns)
  df2_cols = set(df2.columns)

  expressions: list[pl.Expr] = []
  for col in joined.columns:
    if col.endswith("_new"):
      base_col = col.removesuffix("_new")
      if base_col in df1_cols:
        expressions.append(pl.coalesce(pl.col(col), pl.col(base_col)).alias(base_col))
      else:
        expressions.append(pl.col(col).alias(base_col))
    elif col in index_cols or (col in df1_cols and f"{col}_new" not in df2_cols):
      expressions.append(pl.col(col))

  return joined.select(expressions).collect()


def polars_to_sqlite(
  df: pl.DataFrame,
  db_name: str,
  table: str,
  action: Literal["merge", "replace"] = "merge",
  index_cols: Sequence[str] | None = None,
) -> None:
  if action == "merge" and not index_cols:
    raise ValueError("Index columns required for merging DataFrames")

  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  inspector = inspect(engine)

  with engine.connect() as con:
    if action == "replace":
      con.execute(text(f"DROP TABLE IF EXISTS '{table}'"))
      con.commit()

    if not inspector.has_table(table):
      ddl = polars_sqlite_table(df, table, index_cols)
      con.execute(text(ddl))

      df.write_database(
        table_name=table,
        connection=engine,
        engine="sqlalchemy",
        if_table_exists="append",
      )
      return

    query = text(f"SELECT * FROM '{table}'")
    df_old = pl.read_database(query, connection=con)
    merged = combine_polars(df, df_old, cast(Sequence[str], index_cols))

    ddl = polars_sqlite_table(merged, table, index_cols)
    con.execute(text(f"DROP TABLE IF EXISTS '{table}'"))
    con.execute(text(ddl))
    con.commit()

    merged.write_database(
      table_name=table,
      connection=engine,
      engine="sqlalchemy",
      if_table_exists="append",
    )


def upsert_sqlite(
  df: pd.DataFrame, db_name: str, tbl_name: str, dtype: DtypeArg | None = None
) -> None:
  if not df.index.is_unique:
    raise ValueError("DataFrame index is not unique. Failed to upsert into SQLite!")

  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  inspector = inspect(engine)

  # SQL index headers
  ix_cols = list(df.index.names)
  ix_cols_text = ", ".join(f'"{i}"' for i in ix_cols)

  if not inspector.has_table(tbl_name):
    df.to_sql(tbl_name, con=engine, index=True, dtype=dtype)

    # Create index
    with engine.begin() as con:
      con.execute(
        text(f'CREATE UNIQUE INDEX "ix_{tbl_name}" ON "{tbl_name}" ({ix_cols_text})')
      )

    return

  with engine.begin() as con:
    # SQL header query text
    cols = list(df.columns.tolist())
    headers = ix_cols + cols
    headers_text = ", ".join(f'"{i}"' for i in headers)
    update_text = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in cols])

    # Store data in temporary table
    # con.execute(text(f'CREATE TEMP TABLE temp({headers_text})'))

    df.to_sql("temp", con=con, if_exists="replace", index=True, dtype=dtype)

    # Check if new columns must be inserted
    result = con.execute(text(f'SELECT * FROM "{tbl_name}" LIMIT 1'))
    old_cols = set([c for c in result.keys()]).difference(set(ix_cols))
    new_cols = list(set(cols).difference(old_cols))
    if new_cols:
      for c in new_cols:
        con.execute(text(f'ALTER TABLE "{tbl_name}" ADD COLUMN {c}'))

    # Upsert data to main table
    query = f"""
      INSERT INTO "{tbl_name}" ({headers_text})
      SELECT {headers_text} FROM temp WHERE true
      ON CONFLICT ({ix_cols_text}) DO UPDATE 
      SET {update_text}
    """

    con.execute(
      text(
        f'CREATE UNIQUE INDEX IF NOT EXISTS "ix_{tbl_name}" ON "{tbl_name}" ({ix_cols_text})'
      )
    )
    con.execute(text(query))
    # con.execute(text('DROP TABLE temp')) # Delete temporary table


def polars_to_sqlite_upsert(
  df: pl.DataFrame, db_name: str, table: str, index_cols: Sequence[str]
) -> None:
  missing = set(index_cols).difference(df.columns)
  if missing:
    raise ValueError(f"Index columns missing from DataFrame: {missing}")

  if df.select(pl.col(index_cols)).n_unique() != df.height:
    raise ValueError("DataFrame index is not unique. Failed to upsert into SQLite!")

  db_path = sqlite_path(db_name)
  engine = create_engine(f"sqlite+pysqlite:///{db_path}")
  inspector = inspect(engine)

  with engine.connect() as con:
    if not inspector.has_table(table):
      ddl = polars_sqlite_table(df, table, index_cols)
      con.execute(text(ddl))
      con.commit()

      df.write_database(
        table_name=table,
        connection=engine,
        engine="sqlalchemy",
        if_table_exists="append",
      )

      return

    table_cols = [c["name"] for c in inspector.get_columns(table)]
    new_cols = list(set(df.columns).difference(table_cols))
    if new_cols:
      for c in new_cols:
        sql_type = _POLARS_TO_SQLITE.get(df[c].dtype, "BLOB")
        con.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {c} {sql_type}'))
        con.commit()

    pk_cols = inspector.get_pk_constraint(table)["constrained_columns"]
    ix_sql = ", ".join(f'"{i}"' for i in index_cols)
    if set(pk_cols) != set(index_cols):
      con.execute(text(f'CREATE UNIQUE INDEX "ix_{table}" ON "{table}" ({ix_sql})'))
      con.commit()

    temp_table = "_temp_upsert"
    df.write_database(
      table_name=temp_table,
      connection=engine,
      engine="sqlalchemy",
      if_table_exists="replace",
    )

    nonindex_cols = set(df.columns).difference(index_cols)
    cols_sql = ", ".join(f'"{c}"' for c in df.columns)
    update_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in nonindex_cols)

    upsert_query = f"""
      INSERT INTO "{table}" ({cols_sql})
      SELECT {cols_sql} FROM "{temp_table}"
      ON CONFLICT ({ix_sql}) DO UPDATE SET
        {update_sql}
    """

    con.execute(text(upsert_query))
    con.execute(text(f'DROP TABLE "{temp_table}"'))
    con.commit()


def replace_sqlite(col: str, replacements: dict[str, str]) -> str:
  old, new = replacements.popitem()
  query = f'REPLACE({col}, "{old}", "{new}")'

  for old, new in replacements.items():
    query = f'REPLACE({query}, "{old}", "{new}")'

  return query


def json_query(schema: dict[str, Literal["patch", "unique"]]) -> str:
  query: list[str] = []
  for col, fn in schema.items():
    if fn == "patch":
      query.append(f'"{col}"=json_patch("{col}", excluded."{col}")')
    elif fn == "unique":
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

  return ",".join(query)


def upsert_json(
  db_name: str,
  table: str,
  fields: dict[str, str],
  ix: list[str],
  json_cols: dict[str, Literal["patch", "unique"]],
  records: list[tuple],
):
  db_path = sqlite_path(db_name)

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    fields_text = ",".join([" ".join((k, v)) for k, v in fields.items()])

    cur.execute(f"""CREATE TABLE IF NOT EXISTS "{table}"({fields_text})""")

    ix_text = ",".join(ix)
    cur.execute(
      f"""CREATE UNIQUE INDEX IF NOT EXISTS "ix_{table}" 
      ON "{table}"({ix_text})"""
    )

    columns = ",".join(tuple(fields.keys()))
    values = ",".join(["?"] * len(columns))

    upsert_text = json_query(json_cols)

    query = f"""INSERT INTO "{table}"({columns}) VALUES ({values})
      ON CONFLICT ({ix_text}) DO UPDATE SET 
        {upsert_text}
    """
    cur.executemany(query, records)


def upsert_strings(db_name: str, table: str, column: str, values: list[str]):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cursor = con.cursor()

    cursor.execute(f"""
      CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {column} TEXT UNIQUE
      )
    """)

    insert_query = f"""
      INSERT INTO {table} ({column})
      VALUES (?)
      ON CONFLICT({column}) DO NOTHING
    """

    cursor.executemany(insert_query, [(s,) for s in values])
    con.commit()


def create_fts_table(
  db_name: str, table: str, columns: list[str], tokenizer: str = "porter"
):
  db_path = sqlite_path(db_name)

  query = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {table}
    USING fts5({",".join(columns)}, tokenize={tokenizer})
  """

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()
    cur.execute(query)
    con.commit()


def drop_fts_table(db_name: str, table: str):
  db_path = sqlite_path(db_name)

  suffixes = ["_config", "_content", "_data", "_docsize", "_idx"]

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()
    cur.execute(f"DROP TABLE IF EXISTS '{table}'")

    for suffix in suffixes:
      aux_table = table + suffix
      cur.execute(f"DROP TABLE IF EXISTS '{aux_table}'")

    con.commit()


def add_column(db_name: str, table: str, column: str, dtype: str):
  db_path = sqlite_path(db_name)

  with closing(sqlite3.connect(db_path)) as con:
    cur = con.cursor()

    cur.execute(f"PRAGMA table_info({table})")
    columns = cur.fetchall()

    if column in [c[1] for c in columns]:
      return

    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")

    con.commit()
