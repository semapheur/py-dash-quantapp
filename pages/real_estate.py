from dash import callback, html, no_update, register_page, Input, Output

from components.map import choropleth_map, MAP_PATH
from lib.virdi import price_choropleth

register_page(__name__, path='/real_estate')

choro_path = MAP_PATH / 'choropleth.html'

if not choro_path.exists():
  choropleth_map()

mapFrame = html.Iframe(
  id='realestate-map-choropleth', srcDoc=open(choro_path, 'r').read(),
  width='100%', height='800px'
)

layout = html.Main(className='h-full', children=[
  html.Div(id='real-estate-div:map-frame')
])