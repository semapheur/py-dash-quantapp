from dash import callback, html, dcc, no_update, Input, Output, State, MATCH


def StockHeader(_id: str):
  return html.Nav(
    className='flex gap-4',
    children=[
      dcc.Link(
        'Overview',
        href=f'/stock/{_id}/overview',
        id={'type': 'link:stock', 'index': 'overview'},
      ),
      dcc.Link(
        'Financials',
        href=f'/stock/{_id}/financials',
        id={'type': 'link:stock', 'index': 'financials'},
      ),
      dcc.Link(
        'Fundamentals',
        href=f'/stock/{_id}/fundamentals',
        id={'type': 'link:stock', 'index': 'fundamentals'},
      ),
      dcc.Link(
        'Valuation',
        href=f'/stock/{_id}/valuation',
        id={'type': 'link:stock', 'index': 'valuation'},
      ),
    ],
  )


@callback(
  Output({'type': 'link:stock', 'id': MATCH}, 'className'),
  Input('location:app', 'pathname'),
  State({'type': 'link:stock', 'id': MATCH}, 'href'),
)
def update_link(path: str, href: str):
  if path == href:
    return 'text-secondary'
  else:
    return no_update
