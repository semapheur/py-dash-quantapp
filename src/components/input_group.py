from dash import dcc, html


class InputGroupAIO(html.Form):
  def __init__(self, input_props: list[dict]):
    form_style = (
      'flex divide-x h-full p-1 bg-primary'
      'rounded border border border-text/10'
      'hover:border-text/50 focus:border-secondary'
    )

    super().__init__(
      className=form_style,
      action='',
      children=[dcc.Input(**props) for props in input_props],
    )
