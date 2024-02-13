import logging
from functools import lru_cache
from typing import cast, Optional, TypedDict

from pandera import DataFrameModel
from pandera.typing import DataFrame
from sqlalchemy import create_engine, text

from lib.const import DB_DIR
from lib.db.lite import check_table, get_tables, read_sqlite

db_path = DB_DIR / 'ticker.db'
ENGINE = create_engine(f'sqlite+pysqlite:///{db_path}')

logger = logging.getLogger(__name__)


class TickerOptions(DataFrameModel):
  label: str
  value: str


class Stock(TypedDict, total=False):
  id: str
  ticker: str
  name: str
  mic: str
  currency: str
  sector: str
  industry: str


def stock_currency(id: str) -> str:
  query = text(
    """
    SELECT currency FROM stock WHERE id = :id
  """
  ).bindparams(id=id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)

  if (currency := fetch.first()) is None:
    logger.warning(f'Could not retrieve currency for security {id}', extra={'id': id})
    return 'USD'

  return currency[0]


def stock_label(id: str) -> str:
  if not check_table('stock', ENGINE):
    logger.warning('Stock tickers have not been seeded!')
    return ''

  query = text(
    """
    SELECT name || " (" || ticker || ":" || mic || ")" AS label 
    FROM stock WHERE id = :id
  """
  ).bindparams(id=id)

  with ENGINE.begin() as con:
    fetch = con.execute(query)

  if (label := fetch.first()) is None:
    logger.warning(f'Could not retrieve stock label for {id}', extra={'id': id})
    return ''

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
  security: str,
  search: str,
  href: bool = True,
  limit: int = 10,
  restriction: Optional[tuple[str, ...]] = None,
) -> DataFrame[TickerOptions]:
  membership = ''
  if restriction is not None:
    membership = f'AND id IN {str(restriction)}'

  if security == 'stock':
    value = f'"/{security}/" || id' if href else 'id || "|" || currency'
    query = f"""
      SELECT 
        name || " (" || ticker || ") - "  || mic AS label,
        {value} AS value
      FROM {security} WHERE label LIKE :search {membership}
      LIMIT {limit}
    """

  df = read_sqlite('ticker.db', query, {'search': f'%{search}%'})
  return cast(DataFrame[TickerOptions], df)


@lru_cache
def get_stored_fundamentals() -> tuple[str, ...]:
  return tuple(get_tables('fundamentals.db'))


def search_fundamentals(search: str, limit=10) -> DataFrame[TickerOptions] | None:
  stored_tickers = get_stored_fundamentals()
  if not stored_tickers:
    return None

  query = f"""
  WITH fundamentals AS (
    SELECT
      name || " (" || ticker || ") - "  || mic AS label,
      id || "|" || currency AS value
    FROM stock WHERE id IN {str(stored_tickers)}
  )
  SELECT label, value FROM fundamentals WHERE label LIKE :search

  """

  df = read_sqlite('ticker.db', query, {'search': f'%{search}%'})
  return cast(DataFrame[TickerOptions], df)
