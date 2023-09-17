from dash import html, dcc

def StockHeader(id: str):
  return html.Nav(className='flex gap-4', children=[
    dcc.Link('Overview', href=f'/stock/{id}'),
    dcc.Link('Financials', href=f'/stock/{id}/financials'),
    dcc.Link('Fundamentals', href=f'/stock/{id}/fundamentals')
  ])