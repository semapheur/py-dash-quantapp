from dash import callback, html, dcc, no_update, Input, Output, State

def StockHeader(id: str):
  return html.Nav(className='flex gap-4', children=[
    dcc.Link('Overview', href=f'/stock/{id}', id='link:stock:overview'),
    dcc.Link('Financials', href=f'/stock/{id}/financials'),
    dcc.Link('Fundamentals', href=f'/stock/{id}/fundamentals')
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