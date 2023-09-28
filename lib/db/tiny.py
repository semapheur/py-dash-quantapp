from pathlib import Path
from typing import Optional

from glom import glom
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer

serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')

def tinydb_name(db_name):
  if not db_name.endswith('.json'):
    return db_name + '.json'

  return db_name

def insert_tinydb(
  data: list|dict,
  db_path: str|Path,
  tbl: str='',
  dt_serialize: bool = False
):
  
  if dt_serialize:
    serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
  
  db = TinyDB(db_path, storage=serialization)

  if tbl:
    db = db.table(tbl)

  if isinstance(data, list):
    db.insert_multiple(data)
  elif isinstance(data, dict):
    db.insert(data)

def read_tinydb(
  db_path: str|Path, 
  query: Optional[Query] = None, 
  tbl: Optional[str] = None,
  dt_serialize: bool = False
) -> list|dict :
  
  if dt_serialize:
    serialization.register_serializer(DateTimeSerializer(), 'TinyDate')

  db = TinyDB(db_path, storage=serialization)

  if tbl:
    if tbl not in db.tables():
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