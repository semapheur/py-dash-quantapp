from pathlib import Path

import geopandas as gpd
import folium # MapQuest Open|Stamen Toner|Carto DB positron/dark_matter

from lib.db import DB_DIR
from lib.virdi import spatial_price_stats, postal_code_polys
from lib.geonorge import municipality_polys

MAP_PATH = Path(__file__).resolve().parent.parent / 'assets'

def load_geo_data(unit: str):
  path = DB_DIR / f'nor_{unit}.json'
  if not path.exists():
    if unit == 'municipality':
      gdf = municipality_polys()
    elif unit == 'postal_code':
      gdf = postal_code_polys()

    gdf.to_file(path, driver='GeoJSON', encoding='utf-8')
  
  else:
    gdf = gpd.read_file(path, driver='GeoJSON', encoding='utf-8')
  
  return gdf

def choropleth_map():

  # Folium map
  mp = folium.Map(location=[59.90, 10.75], zoom_start=10, tiles='Stamen Toner')

  # Choropleth
  fields = [
    {'key': 'municipality', 'label': 'Kommune'},
    {'key': 'postal_code', 'label': 'Poststed'}
  ]

  for field in fields: 
    df = spatial_price_stats(field['key'])
    gdf = load_geo_data(field['key'])

    choropleth = folium.Choropleth(
      name=f'choro_{field["key"]}',
      geo_data=gdf,
      data=df,
      columns=[field['key'], 'price_per_area', 'price_per_area_std'],
      key_on=f'feature.properties.{field["key"]}',
      fill_color='YlOrRd',
      fill_opacity=0.7,
      legend_name='Price/Area',
      highlight=True,
      line_color='',
      overlay=True
      #nan_fill_color='#0000'
    ).add_to(mp)
    print(field['key'])

    ## Tooltip
    #choropleth.geojson.add_child(
    #  folium.features.GeoJsonTooltip(
    #    fields=[field['key'], 'price_per_area', 'price_per_area_std'],
    #    aliases=[field['label'], 'Kvm.pris', 'Std.avvik'],
    #    style=(
    #      'background-color: white; color: #333333;'
    #      'fontfamily: arial; font-size: 12px; padding: 10px'
    #    )
    #  )
    #)

  # Tile layer
  folium.TileLayer('OpenStreetMap', control=True).add_to(mp)
  folium.TileLayer('CartoDb dark_matter', control=True).add_to(mp)

  # Layer control
  folium.LayerControl(collapsed=False).add_to(mp)

  mp.save(MAP_PATH / 'choropleth.html') # mp._repr_html_()