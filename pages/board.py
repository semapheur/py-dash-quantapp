import asyncio
import threading

from dash import callback, ctx, html, no_update, register_page, Input, Output

# from components.map import choropleth_map
from lib.const import DB_DIR
from lib.db.lite import insert_sqlite
from lib.morningstar.fetch import get_tickers
from lib.edgar.parse import get_ciks

register_page(__name__, path="/board")

main_style = "h-full grid grid-cols-[1fr_1fr] gap-2 p-2"
layout = html.Main(
  className=main_style,
  children=[
    html.Div(
      id="board-div:ticker",
      className="flex flex-col",
      children=[
        html.H4("Update tickers"),
        html.Button("Stocks", id="board-button:stock", n_clicks=0),
        html.Button("SEC Edgar", id="board-button:cik", n_clicks=0),
      ],
    ),
  ],
)


@callback(
  Output("board-div:ticker", "className"),
  Input("board-button:stock", "n_clicks"),
  Input("board-button:cik", "n_clicks"),
)
def update_tickers(n1: int, n2: int):
  button_id = ctx.triggered_id if not None else ""

  if not button_id:
    return no_update

  # Run the asyncio coroutine in a separate thread
  def run_coroutine():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    df = loop.run_until_complete(get_tickers("stock"))
    df.set_index("id", inplace=True)
    df.sort_values("name", inplace=True)
    insert_sqlite(df, "ticker.db", "stock", "overwrite")

    loop.close()

  if button_id == "board-button:stock":
    thread = threading.Thread(target=run_coroutine)
    thread.start()

  elif button_id == "board-button:cik":
    df = get_ciks()
    insert_sqlite(df, "ticker.db", "edgar", "replace")

  return no_update
