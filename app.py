# -*- coding: utf-8 -*-
import atexit
import json
import logging.config
import logging.handlers
from pathlib import Path

from dash import Dash, dcc, html, page_container
from dash.dependencies import Input, Output

from components.header import Header

# https://utxo.live/oracle/


def setup_logging():
  config_file = Path('lib/log/logging_config.json')

  with open(config_file, 'r') as f:
    config = json.load(f)
  logging.config.dictConfig(config)
  queue_handler = logging.getHandlerByName('queue_handler')
  if queue_handler is not None:
    queue_handler.listener.start()
    atexit.register(queue_handler.listener.stop)


def cleanup():
  temp = Path('temp')

  for file in temp.iterdir():
    file.unlink()


# atexit.register(cleanup)

app = Dash(
  __name__,
  use_pages=True,
  title='Gelter',
  suppress_callback_exceptions=True,
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

  setup_logging()
  app.run_server(
    debug=True,
    dev_tools_hot_reload=False,
  )  # , host='0.0.0.0', port=8080)
