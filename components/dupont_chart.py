from typing import Literal

from dash import html

wrapper_style = 'h-full flex flex-col justify-center items-center gap-y-8'
h_branch = (
  'relative flex justify-center gap-x-3 '
  'before:absolute before:content-[""] before:h-[1px] '
  'before:-top-4 before:bg-black '
)
li_style = (
  'flex flex-col items-center gap-y-8 w-max ' 
)

v_branch = (
  'relative flex flex-col items-start gap-y-1 '
  'before:content-[""] before:absolute before:h-[calc(100%+0.25rem)] before:w-[1px] '
  'before:-top-8 before:left-2 before:bg-black'
)
h_node = 'translate-x-6'
v_sign = 'self-center text-2xl translate-x-6'
h_sign = 'self-start text-2xl translate-y-2'

def card(
  top_text: str, 
  bottom_text: float = 0.,
  bottom_id: str = '',
  center: bool = False, 
  buds: tuple[Literal['top', 'bottom', 'left']] = tuple()
) -> html.Div:
  card_style = (
    'relative flex flex-col w-40 p-1 divide-y '
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

  for b in buds:
    card_style += bud_style[b]
    
  top_style = 'w-full font-bold text-center'
  bottom_style = 'w-full text-center'
  
  return html.Div(className=card_style, children=[
    html.Span(top_text, className=top_style),
    html.Span(f'{bottom_text:.3G}', id=bottom_id, className=bottom_style)
  ])

def DupontChart(id_prefix: str = 'span:dupont-chart:'): 
  return html.Div(className=wrapper_style, children=[
    card(
      top_text='Return on Equity',
      bottom_id=id_prefix + 'return_on_equity',  
      buds=('bottom',)),
    html.Ul(className=h_branch + 'before:left-[17.55rem] before:w-[56.4rem]', children=[
      html.Li(className=li_style, children=[
        card(
          top_text='Net Profit Margin', 
          bottom_id=id_prefix + 'net_profit_margin',
          center=True, 
          buds=('top', 'bottom')),
        html.Ul(className=h_branch + 'before:left-20 before:w-[25.1rem]', children=[
          html.Li(className=li_style, children=[
            card(
              top_text='Operating Margin', 
              bottom_id=id_prefix + 'operating_profit_margin',
              buds=('top',)),
            html.Ul(className=v_branch, children=[
              html.Li(className=h_node, children=[
                card(
                  top_text='Operating Income',
                  bottom_id=id_prefix + 'operating_margin:operating_income_loss',
                  buds=('left',))
              ]),
              html.Span('÷', className=v_sign),
              html.Li(className=h_node, children=[
                card(
                  top_text='Revenue', 
  
                  bottom_id=id_prefix + 'operating_margin:revenue',
                  buds=('left',))
              ])
            ])
          ]),
          html.Span('×', className=h_sign),
          html.Li(className=li_style, children=[
            card(
              top_text='Tax Burden',
              bottom_id=id_prefix + 'tax_burden',
              buds=('top',)),
            html.Ul(className=v_branch, children=[
              html.Li(className=h_node, children=[
                card(
                  top_text='Net Income',
  
                  bottom_id=id_prefix + 'net_income_loss',
                  buds=('left',))
              ]),
              html.Span('÷', className=v_sign),
              html.Li(className=h_node, children=[
                card(
                  top_text='Pretax Income',
  
                  bottom_id=id_prefix + 'tax_burden:pretax_income_loss',
                  buds=('left',))
              ])
            ])
          ]),
          html.Span('×', className=h_sign),
          html.Li(className=li_style, children=[
            card(
              top_text='Interest Burden',
              bottom_id=id_prefix + 'interest_burden',
              buds=('top',)),
            html.Ul(className=v_branch, children=[
              html.Li(className=h_node, children=[
                card(
                  top_text='Pretax Income',
  
                  bottom_id=id_prefix + 'interest_burden:pretax_income_loss',
                  buds=('left',))
              ]),
              html.Span('÷', className=v_sign),
              html.Li(className=h_node, children=[
                card(
                  top_text='Operating Income',
  
                  bottom_id=id_prefix + 'interest_burden:operating_income_loss',
                  buds=('left',))
              ])
            ])
          ])
        ])
      ]),
      html.Span('×', className=h_sign),
      html.Li(className=li_style, children=[
        card(
          top_text='Asset Turnover',
          bottom_id=id_prefix + 'asset_turnover',
          center=True,
          buds=('top', 'bottom')),
        html.Ul(className=h_branch + 'before:left-20 before:w-[12.6rem]', children=[
          html.Li(className=li_style, children=[
            card(
              top_text='Revenue', 
              bottom_id=id_prefix + 'asset_turnover:revenue',
              buds=('top',))
          ]),
          html.Span('÷', className=h_sign),
          html.Li(className=li_style, children=[
            card(
              top_text='Average Assets', 
              bottom_id=id_prefix + 'asset_turnover:average_assets',
              buds=('top',))
          ])
        ])
      ]),
      html.Span('×', className=h_sign),
      html.Li(className=li_style, children=[
        card(
          top_text='Equity Multiplier',
          bottom_id=id_prefix + 'equity_multiplier', 
          center=True,
          buds=('top', 'bottom')),
        html.Ul(className=h_branch + 'before:left-20 before:w-[12.6rem]', children=[
          html.Li(className=li_style, children=[
            card(
              top_text='Average Assets', 
              bottom_id=id_prefix + 'equity_multiplier:average_assets',
              buds=('top',))
          ]),
          html.Span('÷', className=h_sign),
          html.Li(className=li_style, children=[
            card(
              top_text='Average Equity',
              bottom_id=id_prefix + 'average_equity',
              buds=('top',))
          ])
        ])
      ]),
    ])
  ])