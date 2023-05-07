import branca.colormap as cm
from dash import callback, html, no_update, register_page, Input, Output
from dash_extensions.javascript import arrow_function, assign
import dash_leaflet as dl
import dash_leaflet.express as dlx
import numpy as np

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

def colorbar_hideout(unit: str, prop: str, style: dict, classes: int=5) -> tuple:
  path = DB_DIR / 'colorbar_values.json'
  vmin, vmax = load_json(path)[unit]

  ctg = [0, *np.linspace(vmin, vmax, classes)]

  colormap = cm.LinearColormap(
    ['gray', 'green', 'yellow', 'red'],
    vmin=vmin, vmax=vmax,
    index=[0.0, vmin, (vmax - vmin)/2, vmax]
  ).to_step(classes + 1)
  colorscale = [rgba_to_hex(c) for c in colormap.colors]

  colorbar = dlx.categorical_colorbar(
    categories=[f'{c:.2E}' for c in ctg],
    colorscale=colorscale,
    unit='/m2',
    width=500, height=10,
    position='bottomleft'
  )
  hideout = dict(
    classes=ctg,
    colorscale=colorscale,
    style=style,
    colorProp=prop
  )
  return colorbar, hideout

for unit in ('municipality', 'postal_code'):
  path = STATIC_DIR / f'realestate_choro_{unit}.json'
  if not path.exists():
    choropleth_polys(unit)

style = dict(weight=1, opacity=1, color='black', fillOpacity=0.5)
colorbar, hideout = colorbar_hideout('municipality', 'price_per_area', style)

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

layout = [
  dl.Map(id='realestate-map', className='h-full',
    zoom=10, center=(59.90, 10.75), 
    children=[
      dl.LayersControl(children=base_layer()),
      dl.GeoJSON(id='realestate-geojson:choropleth-postal',
        url='/assets/realestate_choro_municipality.json',
        #format='geobuf',
        options=dict(style=style_handle),
        hideout=hideout,
        hoverStyle=arrow_function(dict(weight=2, color='white')),
        zoomToBoundsOnClick=True
      ),
      colorbar
    ]
  ),
  html.Div(id='test')
]

@callback(
  Output('test', 'children'),
  Input('realestate-map', 'zoom'),
  Input('realestate-map', 'center'),
  Input('realestate-map', 'bounds'),
)
def test(zoom: int, center: tuple, bounds):
  print(bounds)

  return no_update

#choro_path = MAP_PATH / 'choropleth.html'
#if not choro_path.exists():
#  choropleth_map()

#layout = html.Main(id='real-estate', className='h-full', children=[
#  html.Iframe(id='real-estate-iframe:choropleth', 
#    src='assets/choropleth.html',
#    width='100%', height='100%'
#  )
#])