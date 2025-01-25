from dash import (
  # callback,
  # clientside_callback,
  # ClientsideFunction,
  # ctx,
  html,
  # no_update,
  register_page,
  # Input,
  # Output,
  # State,
)

from components.ticker_select import TickerSelectAIO
from components.quote_graph import (
  QuoteGraphAIO,
  # quote_volume_graph,
  # quote_graph_relayout,
  # quote_graph_range,
)
from components.quote_graph_type import QuoteGraphTypeAIO
from components.quote_graph import QuoteDatePickerAIO

# from components.quote_store import QuoteStoreAIO
from components.macro_choropleth import MacroChoropleth
# from lib.fin.quote import load_ohlcv
# from lib.morningstar.ticker import Stock

register_page(__name__, path="/")

main_style = "h-full grid grid-rows-2 grid-cols-2 gap-2 p-2 bg-primary"

layout = html.Main(
  className=main_style,
  children=[
    MacroChoropleth(className="h-full rounded-sm shadow-sm bg-primary"),
    html.Div(
      className="h-full flex flex-col rounded-sm shadow-sm",
      children=[
        html.Form(
          className="grid grid-cols-[2fr_1fr_auto] gap-2 px-2 pt-2",
          children=[
            TickerSelectAIO(id="home"),
            QuoteGraphTypeAIO(id="home"),
            QuoteDatePickerAIO(id="home"),
          ],
        ),
        QuoteGraphAIO(id="home"),
      ],
    ),
    # QuoteStoreAIO(id="home"),
  ],
)


# @callback(
#  Output(QuoteGraphAIO.aio_id("home"), "figure", allow_duplicate=True),
#  Input(QuoteStoreAIO.aio_id("home"), "data"),
#  Input(QuoteGraphAIO.aio_id("home"), "relayoutData"),
#  Input(QuoteGraphTypeAIO.aio_id("home"), "value"),
#  Input(QuoteDatePickerAIO.aio_id("home"), "start_date"),
#  Input(QuoteDatePickerAIO.aio_id("home"), "end_date"),
#  prevent_initial_call=True,
# )
# def update_graph(data, relayout, plot_type, start_date, end_date):
#  if not data:
#    return no_update
#
#  triggered_id = ctx.triggered_id
#
#  if triggered_id.get("component", "") in ("QuoteStoreAIO", "QuoteGraphTypeAIO"):
#    return quote_volume_graph(
#      data, plot_type, rangeselector=("1M", "6M", "YTD", "1Y", "All"), rangeslider=False
#    )
#  elif triggered_id.get("component", "") == "QuoteGraphAIO" and relayout:
#   return quote_graph_relayout(relayout, data, ["close", "volume"])
#
#  elif triggered_id.get("component", "") == "QuoteDatePickerAIO":
#    return quote_graph_range(data, ["close", "volume"], start_date, end_date)
#
#  return no_update


# clientside_callback(
#  ClientsideFunction(namespace="clientside", function_name="updateQuoteGraph"),
#  Output(QuoteGraphAIO.aio_id("home"), "figure"),
#  Input(QuoteGraphAIO.aio_id("home"), "relayoutData"),
#  State(QuoteGraphAIO.aio_id("home"), "figure"),
#  prevent_initial_call=True,
# )
