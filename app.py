# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from dash import Dash, dcc, html, page_container, Input, Output

from lib.log.logger import setup_queue_handler

from components.header import Header

# https://utxo.live/oracle/


def cleanup():
  temp = Path('temp')

  for file in temp.iterdir():
    file.unlink()


# atexit.register(cleanup)

external_stylesheets = ['https://cdn.tailwindcss.com']

app = Dash(
  __name__,
  use_pages=True,
  title='Gelter',
  suppress_callback_exceptions=True,
  external_stylesheets=external_stylesheets,
  # prevent_initial_callbacks='initial_duplicate'
)  # run with 'python app.py'

# print(page_registry)

app.layout = html.Div(
  id='app',
  children=[Header(), page_container, dcc.Location(id='location:app', refresh=False)],
)

app.clientside_callback(
  """
  function(theme) {
    document.body.dataset.theme = theme === undefined ? "light" : theme 
    return theme
  }
  """,
  Output('theme-toggle', 'value'),
  Input('theme-toggle', 'value'),
)

if __name__ == '__main__':
  # cleanup()

  queue_handler = setup_queue_handler()
  logger = logging.getLogger(__name__)
  logger.addHandler(queue_handler)

  app.run_server(
    debug=True,
    dev_tools_hot_reload=False,
  )  # , host='0.0.0.0', port=8080)
