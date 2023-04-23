from dash import callback, html, no_update, register_page, Input, Output

from components.map import choropleth_map, MAP_PATH

register_page(__name__, path='/real_estate')

choro_path = MAP_PATH / 'choropleth.html'

if not choro_path.exists():
  choropleth_map()

map_frame = html.Iframe(id='real-estate-iframe:choropleth', 
  srcDoc=open(choro_path, 'r').read(),
  width='100%', height='100%'
)

layout = html.Main(className='h-full', children=[
  map_frame
])