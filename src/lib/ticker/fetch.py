import logging
from typing import cast

from pandera import DataFrameModel
from pandera.typing import DataFrame

from lib.db.lite import fetch_sqlite, read_sqlite

logger = logging.getLogger(__name__)


class TickerOptions(DataFrameModel):
  label: str
  value: str


def get_primary_securities(id: str) -> DataFrame:
  query = """
    SELECT s.security_id, s.ticker, s.mic, s.currency
    FROM stock s
    JOIN company c ON s.company_id = c.company_id
    JOIN json_each(c.primary_security) AS ps
    WHERE 
      s.company_id = :id AND 
      s.security_id = ps.value
  """

  securities = read_sqlite("ticker.db", query, {"id": id})

  if securities is None:
    raise ValueError(f"No primary securities found for {id}")

  return securities


def company_currency(id: str) -> list[str]:
  query = """
    SELECT DISTINCT s.currency
    FROM stock s
    JOIN company c ON s.company_id = c.company_id
    JOIN json_each(c.primary_security) AS ps
    WHERE 
      s.company_id = :id AND 
      s.security_id = ps.value
  """

  currency = read_sqlite("ticker.db", query, {"id": id})

  if currency is None:
    raise ValueError(f"Could not retrieve currency for security {id}")

  return currency["currency"].tolist()


def company_label(id: str) -> str:
  query = """ 
    SELECT
      name || " (" || group_concat(ticker, ", ") || ")" AS label
    FROM (
      SELECT
        c.name, 
        s.ticker
      FROM company c 
      JOIN json_each(c.primary_security) ON 1=1
      JOIN stock s ON json_each.value = s.security_id
      WHERE c.company_id = :id
    )
    GROUP BY name
  """

  fetch = fetch_sqlite("ticker.db", query, {"id": id})

  if (label := fetch[0]) is None:
    logger.warning(f"Could not retrieve stock label for {id}", extra={"id": id})
    return ""

  return label[0]


def get_cik(id: str) -> int | None:
  query = """SELECT DISTINCT edgar.cik AS cik
    FROM stock
    JOIN edgar ON stock.isin = edgar.isin
    WHERE stock.company_id = :id
  """

  fetch = fetch_sqlite("ticker.db", query, {"id": id})

  if (cik := fetch[0]) is not None:
    return cik[0]

  return None


def search_companies(
  search: str,
  stored: bool = True,
  limit: int = 10,
) -> DataFrame[TickerOptions]:
  query = f""" 
    SELECT
      name || " (" || group_concat(ticker || ":" || mic, ", ") || ")" AS label,
      company_id AS value
    FROM (
      SELECT 
        {"f" if stored else "c"}.company_id, 
        c.name, 
        t.ticker, 
        t.mic 
      {"FROM financials f JOIN company c ON f.company_id = c.company_id" if stored else "FROM company c"}
      JOIN json_each(c.primary_security) ON 1=1
      JOIN stock t ON json_each.value = t.security_id
    )
    GROUP BY company_id
    HAVING label LIKE :search
    LIMIT :limit
  """

  df = read_sqlite("ticker.db", query, {"search": f"%{search}%", "limit": str(limit)})
  return cast(DataFrame[TickerOptions], df)


def search_stocks(search: str, limit: int = 10) -> DataFrame[TickerOptions]:
  query = """ 
    SELECT
      name || " (" || ticker || ":" || mic || ")" AS label,
      security_id || "_" || currency AS value
    FROM stock
    WHERE label LIKE :search
    LIMIT :limit
  """

  df = read_sqlite("ticker.db", query, {"search": f"%{search}%", "limit": str(limit)})
  return cast(DataFrame[TickerOptions], df)


def get_currency(id: str) -> str | None:
  currency = fetch_sqlite(
    "ticker.db", "SELECT currency FROM company WHERE company_id = :id", {"id": f"{id}"}
  )

  if currency is None:
    return None

  return currency[0][0]
