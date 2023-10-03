from dash import callback, html, dcc, no_update, Input, Output, State

def StockHeader(_id: str):
  return html.Nav(className='flex gap-4', children=[
    dcc.Link('Overview', href=f'/stock/{_id}', id='link:stock:overview'),
    dcc.Link('Financials', href=f'/stock/{_id}/financials'),
    dcc.Link('Fundamentals', href=f'/stock/{_id}/fundamentals'),
    dcc.Link('Valuation', href=f'/stock/{_id}/valuation')
  ])

@callback(
  Output('link:stock:overview', 'className'),
  Input('location:app', 'pathname'),
  State('link:stock:overview', 'href')
)
def update_link(path: str, href: str):

  if path == href:
    return 'text-secondary'
  else:
    return no_update