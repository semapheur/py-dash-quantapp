from lib.db.lite import read_sqlite

query = """
  SELECT stock.id, edgar.cik FROM stock
  INNER JOIN edgar ON
    edgar.ticker = stock.ticker
  WHERE stock.mic = "XOSL" AND stock.ticker IN (SELECT ticker FROM edgar)
  """

df = read_sqlite('ticker.db', query)
