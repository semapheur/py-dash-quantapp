from dash import callback, html, no_update, register_page, Input, Output

from components.map import choropleth_map, MAP_PATH

register_page(__name__, path='/real_estate')

choro_path = MAP_PATH / 'choropleth.html'

if not choro_path.exists():
  choropleth_map()

layout = html.Main(id='real-estate', className='h-full', children=[
  html.Iframe(id='real-estate-iframe:choropleth', 
    src='assets/choropleth.html',
    width='100%', height='100%'
  )
])

#@callback(
#  Output('real-estate-iframe:choropleth', 'srcDoc'),
#  Input('url', 'pathname')
#)
#def update(url):
#  page = url.split('/')[-1]
#  if not page == 'real_estate':
#    return no_update
#
#  return   