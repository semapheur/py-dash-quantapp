from dash import callback, ctx, html, no_update, register_page, Input, Output, State

from components.ticker_select import TickerSelectAIO
from components.quote_graph import (
  QuoteGraphAIO,
  quote_volume_graph,
  quote_graph_relayout,
  quote_graph_range,
)
from components.quote_graph_type import QuoteGraphTypeAIO
from components.quote_datepicker import QuoteDatePickerAIO
from components.quote_store import QuoteStoreAIO
from components.macro_choropleth import MacroChoropleth

register_page(__name__, path='/')

main_style = 'h-full grid grid-rows-2 grid-cols-2 gap-2 p-2 bg-primary'


layout = html.Main(
  className=main_style,
  children=[
    MacroChoropleth(className='h-full rounded shadow bg-primary'),
    html.Div(
      className='h-full flex flex-col rounded shadow',
      children=[
        html.Form(
          className='grid grid-cols-[2fr_1fr_auto] gap-2 px-2 pt-2',
          children=[
            TickerSelectAIO(aio_id='home'),
            QuoteGraphTypeAIO(aio_id='home'),
            QuoteDatePickerAIO(aio_id='home'),
          ],
        ),
        QuoteGraphAIO(aio_id='home'),
      ],
    ),
    QuoteStoreAIO(aio_id='home'),
  ],
)


@callback(
  Output(QuoteGraphAIO.id('home'), 'figure'),
  Input(QuoteStoreAIO.id('home'), 'data'),
  Input(QuoteGraphAIO.id('home'), 'relayoutData'),
  Input(QuoteGraphTypeAIO.id('home'), 'value'),
  Input(QuoteDatePickerAIO.id('home'), 'start_date'),
  Input(QuoteDatePickerAIO.id('home'), 'end_date'),
  State(QuoteGraphAIO.id('home'), 'figure'),
)
def update_graph(data, relayout, plot_type, start_date, end_date, fig):
  if not data:
    return no_update

  triggered_id = ctx.triggered_id

  if triggered_id.get('component', '') in ('QuoteStoreAIO', 'QuoteGraphTypeAIO'):
    return quote_volume_graph(
      data, plot_type, rangeselector=['1M', '6M', 'YTD', '1Y', 'All'], rangeslider=False
    )
  elif triggered_id.get('component', '') == 'QuoteGraphAIO' and relayout:
    return quote_graph_relayout(relayout, data, ['close', 'volume'], fig)

  elif triggered_id.get('component', '') == 'QuoteDatePickerAIO':
    return quote_graph_range(data, ['close', 'volume'], fig, start_date, end_date)

  return fig
