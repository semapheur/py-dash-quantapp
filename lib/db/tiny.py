from glom import glom
from tinydb import TinyDB, Query

from lib.const import DB_DIR

def tinydb_name(db_name):
  if not db_name.endswith('.json'):
    return db_name + '.json'

  return db_name

def insert_tinydb(
  data: list|dict,
  db_name: str,
  tbl: str=''
):
  db_path = DB_DIR / tinydb_name(db_name)
  db = TinyDB(db_path)

  if tbl:
    db = db.table(tbl)

  if isinstance(data, list):
    db.insert_multiple(data)
  elif isinstance(data, dict):
    db.insert(data)

def read_tinydb(
  db_name: str, 
  query=None, 
  tbl: str=''
) -> list|dict :
  db_path = DB_DIR / tinydb_name(db_name)
  db = TinyDB(db_path)

  if tbl:
    if not tbl in db.tables():
      return []
    db = db.table(tbl)

  if not query:
    return db.all()
  
  return db.search(query)

def tinydb_field(
  db_name:str, 
  query, 
  field:str, 
  tbl:str='_default'
) -> list:
  result = read_tinydb(db_name, query, tbl)  
  return [glom(r, field) for r in result]