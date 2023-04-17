from dash import html, register_page
from sqlalchemy import create_engine, text

from lib.db import DB_DIR

def title(id=None):
  if not id:
    return '404'

  db_path = DB_DIR / 'ticker.db'
  engine = create_engine(f'sqlite+pysqlite:///{db_path}')

  query = f'SELECT name || " (" || ticker || ":" exchange || ")" AS label FROM stock WHERE id = "{id}"'
  with engine.begin() as con:
    fetch = con.execute(text(query))
    label = fetch.first()[0]

  return label

register_page(__name__, path_template='/stock/<id>', title=title)

def layout(id=None):
  return html.Div(
    html.H1(id)
  )