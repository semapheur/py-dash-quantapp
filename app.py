# -*- coding: utf-8 -*-
from dash import Dash, html, page_container, page_registry
from dash.dependencies import Input, Output

from components.header import Header

app = Dash(__name__, use_pages=True, title='Gelter') # run with 'python app.py'

#print(page_registry)

app.layout = html.Div(id='app', className='h-full flex flex-col', children=[
  Header(),
  page_container,
])

app.clientside_callback(
  '''
  function(theme) {
    document.body.dataset.theme = theme === undefined ? "light" : theme 
    return theme
  }
  ''',
  Output('theme-toggle', 'value'),
  Input('theme-toggle', 'value'),
)

if __name__ == '__main__':
  app.run_server(debug=True, dev_tools_hot_reload=False) #, host='0.0.0.0', port=8080) 