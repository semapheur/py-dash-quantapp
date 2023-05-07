import branca.colormap as cm
from dash import callback, html, no_update, register_page, Input, Output, State
from dash_extensions.javascript import arrow_function, assign
import dash_leaflet as dl
import dash_leaflet.express as dlx
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

#from components.map import choropleth_map, MAP_PATH
from lib.color import rgba_to_hex
from lib.const import DB_DIR, STATIC_DIR
from lib.utils import load_json
from lib.virdi import choropleth_polys

register_page(__name__, path='/real_estate')

# Map tiles
tiles = {
  'Stadia Maps': {
    'url': 'https://tiles.stadiamaps.com/tiles/{}/{{z}}/{{x}}/{{y}}{{r}}.png',
    'themes': {
      'Alidade': 'alidade_smooth',
      'Alidade Dark': 'alidade_smooth_dark',
      'OSM Bright': 'osm_bright'
    },
    'attr': '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '
  },
  'Stamen': {
    'url': 'http://{{s}}.tile.stamen.com/{}/{{z}}/{{x}}/{{y}}.png',
    'themes': {
      'Toner': 'toner',
      'Terrain': 'terrain'
    },
    'attr': (
      'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under '
      '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
      'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under '
      '<a href="http://www.openstreetmap.org/copyright">ODbL</a>. '
    )
  }
}

def base_layer(default_theme: str ='Alidade'):
  layers = []
  for k, v in tiles.items():
    for t in v['themes']:
      layers.append(dl.BaseLayer(
        dl.TileLayer(
          url=v['url'].format(v['themes'][t]),
          attribution=v['attr']
        ),
        name=t,
        checked=t == default_theme
      ))

  return layers

def make_hideout(unit: str, prop: str, style: dict, classes: int=5) -> tuple:
  path = DB_DIR / 'colorbar_values.json'
  vmin, vmax = load_json(path)[unit]

  ctg = [0, *np.linspace(vmin, vmax, classes)]

  colormap = cm.LinearColormap(
    ['gray', 'green', 'yellow', 'red'],
    vmin=vmin, vmax=vmax,
    index=[0.0, vmin, (vmax - vmin)/2, vmax]
  ).to_step(classes + 1)
  colorscale = [rgba_to_hex(c) for c in colormap.colors]

  hideout = dict(
    classes=ctg,
    colorscale=colorscale,
    style=style,
    colorProp=prop
  )
  return hideout

for unit in ('municipality', 'postal_code'):
  path = STATIC_DIR / f'realestate_choro_{unit}.json'
  if not path.exists():
    choropleth_polys(unit)

style = dict(weight=1, opacity=1, color='black', fillOpacity=0.5)
hideout = make_hideout('municipality', 'price_per_area', style)

style_handle = assign('''function(feature, context) {
  const {classes, colorscale, style, colorProp} = context.props.hideout
  const value = feature.properties[colorProp]
  
  if (value === null) {
    style.fillColor = colorscale[0]
    return style
  }

  for (let i=0; i < classes.length; i++) {
    if (value > classes[i]) {
      style.fillColor = colorscale[i]
    }
  }
  return style
}''')

#path = STATIC_DIR / 'realestate_choro_municipality.json'
#data = gpd.read_file(path)

layout = [
  dl.Map(id='realestate-map', className='h-full',
    zoom=9, center=(59.90, 10.75), 
    children=[
      dl.LayersControl(children=base_layer()),
      dl.GeoJSON(id='realestate-geojson:choropleth',
        #data=data.to_json(),
        url='/assets/realestate_choro_municipality.json',
        #format='geobuf',
        options=dict(style=style_handle),
        hideout=hideout,
        hoverStyle=arrow_function(dict(weight=2, color='white')),
        zoomToBoundsOnClick=True
      ),
      dlx.categorical_colorbar(
        id='realestate-colorbar',
        categories=[f'{c:.2E}' for c in hideout['classes']],
        colorscale=hideout['colorscale'],
        unit='/m2',
        width=500, height=10,
        position='bottomleft'
      )
    ]
  ),
]

@callback(
  Output('realestate-geojson:choropleth', 'url'),
  Output('realestate-geojson:choropleth', 'hideout'),
  Output('realestate-colorbar', 'tickText'),
  Output('realestate-colorbar', 'colorscale'),
  Input('realestate-map', 'zoom'),
  State('realestate-geojson:choropleth', 'url'),
  prevent_initial_call=True
)
def update_geojson(
  zoom: int, 
  url: str
) -> tuple[str, dict, list['str'], list['str']]:

  unit = url.split('/')[-1].split('.')[0]
  if zoom > 11:
    if unit == 'postal_code':
      return no_update
    
    unit = 'postal_code'

  else:
    if unit == 'municipality':
      return no_update
      
    unit = 'municipality'
  
  url = f'/assets/realestate_choro_{unit}.json'
  hideout = make_hideout(unit, 'price_per_area', style)
  ctg = [f'{c:.2E}' for c in hideout['classes']]
  
  return url, hideout, ctg, hideout['colorscale'] 

#def update_geojson(zoom: int, bounds: list[list[float, float]]):
  #mask = Polygon((
  #  (bounds[0][1], bounds[0][0]),
  #  (bounds[1][0], bounds[0][0]),
  #  (bounds[1][0], bounds[1][1]),
  #  (bounds[0][0], bounds[1][1])
  #))
#
  #path = lambda x: STATIC_DIR / f'realestate_choro_{x}.json'
  #if zoom > 9:
  #  gdf = gpd.read_file(path('postal_code'))
  #else:
  #  gdf = gpd.read_file(path('municipality'))
  #
  #gdf = gdf[gdf.intersects(mask)]
#
  #return gdf.to_json()

#choro_path = MAP_PATH / 'choropleth.html'
#if not choro_path.exists():
#  choropleth_map()

#layout = html.Main(id='real-estate', className='h-full', children=[
#  html.Iframe(id='real-estate-iframe:choropleth', 
#    src='assets/choropleth.html',
#    width='100%', height='100%'
#  )
#])