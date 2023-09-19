from dash import callback, ctx, html, no_update, register_page, Input, Output, State

register_page(__name__, path='/dupont')

wrapper_style = 'h-full grid place-items-center'
ol_style = (
  'relative grid grid-cols-3 items-center '
  'before:absolute before:content-[""] before:h-[2px] before:w-1/2 '
  'before:top-0, before:left-1/4 before:bg-black'
)

def card(top: str, bottom: float) -> html.Div:
  card_style = (
    'relative flex flex-col w-fit p-1 divide-y '
    'border border-secondary rounded shadow'
  )
  top_style = 'w-full font-bold text-center'
  bottom_style = 'w-full text-center'
  
  return html.Div(className=card_style, children=[
    html.Span(top, className=top_style),
    html.Span(f'{bottom:.2f}', className=bottom_style)
  ])

layout = html.Div(className=wrapper_style, children=[
  card('Return on Equity', 1.2991),
  html.Ol(className=ol_style, children=[
    html.Li(children=[
      card('Net Profit Margin', 0.2431),
      html.Ol(className='grid grid-cols-3', children=[
        html.Li(children=[
          card('Operating Margin', 0.2812)
        ]),
        html.Li(children=[
          card('Tax Burden', 0.8745)
        ]),
        html.Li(children=[
          card('Interest Burden', 0.9885)
        ])
      ])
    ]),
    html.Li(children=[
      card('Asset Turnover', 0.98),
      html.Ol(className='grid grid-cols-2', children=[
        html.Li(children=[
          card('Revenue', 327.2)
        ]),
        html.Li(children=[
          card('Average Assets', 333.6)
        ])
      ])
    ]),
    html.Li(children=[
      card('Equity Multiplier', 5.45),
      html.Ol(className='grid grid-cols-2', children=[
        html.Li(children=[
          card('Average Assets', 333.6)
        ]),
        html.Li(children=[
          card('Average Equity', 61.2)
        ])
      ])
    ]),
  ])
])