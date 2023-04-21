from dash import html, dcc, page_registry

from components.ticker_search import TickerSearch
from components.theme_toggle import ThemeToggle

header_style = 'grid grid-cols-[auto_1fr_auto_auto] justify-items-center content-start gap-4 p-2 bg-primary border-b border-b-secondary'
logo_style = 'text-4xl text-text hover:text-secondary leading-none'
link_style = 'text-2xl text-text hover:text-secondary'

def Header():
  return html.Header(className=header_style, children=[
    dcc.Link('Γ', className=logo_style, href='/'),
    html.Nav(className='flex gap-4', children=[
			dcc.Link(page['name'], href=page['relative_path'], className=link_style) 
      for page in page_registry.values() if page['location'] == 'header'
    ]),
    ThemeToggle(),
    TickerSearch()
	])