from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial
from typing import cast, Optional, TypedDict

import pandas as pd
from sqlalchemy import create_engine, text

from lib.const import DB_DIR
from lib.db.lite import check_table, read_sqlite, upsert_sqlite

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


def stock_label(_id: str) -> str | None:
  if not check_table('stock', ENGINE):
    return None

  query = text(
    """
    SELECT name || " (" || ticker || ":" || mic || ")" AS label 
    FROM stock WHERE id = :id
  """
  ).bindparams(id=_id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)

  if (label := fetch.first()) is None:
    return None

  return label[0]


def fetch_stock(id: str, cols: Optional[set[str]] = None) -> Stock | None:
  if not check_table('stock', ENGINE):
    return None

  cols = (
    set(Stock.__optional_keys__)
    if cols is None
    else set(Stock.__optional_keys__).intersection(cols)
  )
  if not cols:
    raise Exception(f'Columns must be from {Stock.__optional_keys__}')

  query = text(f'SELECT {",".join(cols)} FROM stock WHERE id = ":id"').bindparams(id=id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)

  if (stock := fetch.first()) is not None:
    return cast(Stock, {c: f for c, f in zip(cols, stock)})

  return None


def find_cik(id: str) -> Optional[int]:
  if not check_table({'stock', 'edgar'}, ENGINE):
    return None

  query = text(
    """
    SELECT  
      edgar.cik AS cik FROM stock, edgar
    WHERE 
      stock.id = :id AND 
      REPLACE(edgar.ticker, "-", "") = REPLACE(stock.ticker, ".", "")
  """
  ).bindparams(id=id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)

  if (cik := fetch.first()) is not None:
    return cik[0]

  return None


def search_tickers(
  security: str, search: str, href: bool = True, limit: int = 10
) -> pd.DataFrame:
  if security == 'stock':
    value = (
      f'"/{security}/" || id AS href' if href else 'id || "|" || currency AS value'
    )
    query = text(
      f"""
      SELECT 
        name || " (" || ticker || ") - "  || mic AS label,
        {value}
      FROM {security} WHERE label LIKE :search
      LIMIT {limit}
    """
    ).bindparams(search=f'%{search}%')

  with ENGINE.connect().execution_options(autocommit=True) as con:
    df = pd.read_sql(query, con=con)

  return df
