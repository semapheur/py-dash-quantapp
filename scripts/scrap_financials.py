from lib.db.lite import read_sqlite

query = """
  SELECT stock.id, edgar.cik FROM stock
  INNER JOIN edgar ON
    edgar.ticker = stock.ticker
  WHERE stock.mic = "XOSL" AND stock.ticker IN (SELECT ticker FROM edgar)
  """

df = read_sqlite('ticker.db', query)

faulty = [
  '0P0000BV6H',
  '0P0001691U',
  '0P0000BRJU',
  '0P0000C80Q',
  '0P0001M59M',
  '0P0000X9JZ',
]
