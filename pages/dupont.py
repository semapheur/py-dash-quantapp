from typing import Literal, Optional

from dash import callback, ctx, html, no_update, register_page, Input, Output, State

register_page(__name__, path='/dupont')

wrapper_style = 'h-full flex flex-col justify-center items-center gap-y-8'
h_branch = (
  'relative flex justify-center gap-x-4 '
  'before:absolute before:content-[""] before:h-[1px] '
  'before:-top-4 before:bg-black '
)
li_style = (
  'flex flex-col items-start gap-y-8 w-max ' 
)

v_branch = (
  'relative flex flex-col items-start gap-y-8 '
  'before:content-[""] before:absolute before:h-[calc(100%+0.25rem)] before:w-[1px] '
  'before:-top-8 before:left-4 before:bg-black'
)
h_node = 'pl-8'

def card(
  top: str, 
  bottom: float, 
  center: bool = False, 
  bud: tuple[Literal['top', 'bottom', 'left']] = tuple()
) -> html.Div:
  card_style = (
    'relative flex flex-col w-max p-1 divide-y '
    'border border-secondary rounded shadow '
  )
  if center:
    card_style += 'self-center '

  bud_style = {
    'top': (
      'before:content-[""] before:absolute before:-top-4 before:left-1/2 '
      'before:w-[1px] before:h-4 before:bg-black '
    ),
    'bottom': (
      'after:content-[""] after:absolute after:top-full after:left-1/2 '
      'after:w-[1px] after:h-4 after:bg-black '
    ),
    'left': (
      'before:content-[""] before:absolute before:top-1/2 before:-left-4 '
      'before:w-4 before:h-[1px] before:bg-black '
    )
  }

  for b in bud:
    card_style += bud_style[b]
    
  top_style = 'w-full font-bold text-center'
  bottom_style = 'w-full text-center'
  
  return html.Div(className=card_style, children=[
    html.Span(top, className=top_style),
    html.Span(f'{bottom:.2f}', className=bottom_style)
  ])

layout = html.Div(className=wrapper_style, children=[
  card('Return on Equity', 1.2991, False, ('bottom',)),
  html.Ol(className=h_branch + 'before:left-[16.8rem] before:w-[40.3rem]', children=[
    html.Li(className=li_style, children=[
      card('Net Profit Margin', 0.2431, True, ('top', 'bottom')),
      html.Ol(className=h_branch + 'before:left-[4.5rem] before:w-[21.98rem]', children=[
        html.Li(className=li_style, children=[
          card('Operating Margin', 0.2812, False, ('top',)),
          html.Ol(className=v_branch, children=[
            html.Li(className=h_node, children=[
              card('Operating Income', 92.0, False, ('left',))
            ]),
            html.Li(className=h_node, children=[
              card('Revenue', 327.2, False, ('left',))
            ])
          ])
        ]),
        html.Li(className=li_style, children=[
          card('Tax Burden', 0.8745, False, ('top',)),
          html.Ol(className=v_branch, children=[
            html.Li(className=h_node, children=[
              card('Net Income', 79.5, False, ('left',))
            ]),
            html.Li(className=h_node, children=[
              card('Pretax Income', 90.9, False, ('left',))
            ])
          ])
        ]),
        html.Li(className=li_style, children=[
          card('Interest Burden', 0.9885, False, ('top',)),
          html.Ol(className=v_branch, children=[
            html.Li(className=h_node, children=[
              card('Pretax Income', 90.9, False, ('left',))
            ]),
            html.Li(className=h_node, children=[
              card('Operating Income', 92.0, False, ('left',))
            ])
          ])
        ])
      ])
    ]),
    html.Li(className=li_style, children=[
      card('Asset Turnover', 0.98, True, ('top', 'bottom')),
      html.Ol(className=h_branch + 'before:left-[2.3rem] before:w-[7.2rem]', children=[
        html.Li(className=li_style, children=[
          card('Revenue', 327.2, False, ('top',))
        ]),
        html.Li(className=li_style, children=[
          card('Average Assets', 333.6, False, ('top',))
        ])
      ])
    ]),
    html.Li(className=li_style, children=[
      card('Equity Multiplier', 5.45, True, ('top', 'bottom')),
      html.Ol(className=h_branch + 'before:left-[3.8rem] before:w-[8.75rem]', children=[
        html.Li(className=li_style, children=[
          card('Average Assets', 333.6, False, ('top',))
        ]),
        html.Li(className=li_style, children=[
          card('Average Equity', 61.2, False, ('top',))
        ])
      ])
    ]),
  ])
])