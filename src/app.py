import logging
import os
from pathlib import Path

from dash import Dash, dcc, html, page_container, Input, Output  # page_registry
# from flask_caching import Cache

from lib.log.setup import setup_queue_handler

from components.header import Header

# https://utxo.live/oracle/


def cleanup():
  temp = Path("temp")

  for file in temp.iterdir():
    file.unlink()


# atexit.register(cleanup)

if "REDIS_URL" in os.environ:
  from celery import Celery
  from dash import CeleryManager

  celery_app = Celery(
    __name__, broker=os.environ["REDIS_URL"], backend=os.environ["REDIS_URL"]
  )
  background_callback_manager = CeleryManager(celery_app)

else:
  import diskcache
  from dash import DiskcacheManager

  cache = diskcache.Cache(".cache")
  background_callback_manager = DiskcacheManager(cache)

external_stylesheets = ["https://cdn.tailwindcss.com"]

app = Dash(
  __name__,
  use_pages=True,
  title="Gelter",
  suppress_callback_exceptions=True,
  background_callback_manager=background_callback_manager,
  # external_stylesheets=external_stylesheets,
  # prevent_initial_callbacks='initial_duplicate'
)  # run with 'python app.py'

# cache = Cache(
#  app.server,
#  config={
#    "CACHE_TYPE": "FileSystemCache",
#    "CACHE_DIR": ".cache",
#    "CACHE_DEFAULT_TIMEOUT": 300,
#  },
# )

# print(page_registry)

app.layout = html.Div(
  id="app",
  children=[Header(), page_container, dcc.Location(id="location:app", refresh=False)],
)

app.clientside_callback(
  """
  function(theme) {
    document.body.dataset.theme = theme === undefined ? "light" : theme 
    return theme
  }
  """,
  Output("theme-toggle", "value"),
  Input("theme-toggle", "value"),
)

if __name__ == "__main__":
  # cleanup()

  queue_handler = setup_queue_handler()
  logger = logging.getLogger(__name__)
  logger.addHandler(queue_handler)

  app.run(
    debug=True,
    dev_tools_hot_reload=False,
  )  # , host='0.0.0.0', port=8080)
