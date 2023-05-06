from pathlib import Path

import folium # MapQuest Open|Stamen Toner|Carto DB positron/dark_matter
import branca.colormap as cm

from lib.virdi import load_price_data, spatial_price_stats, load_geo_data

MAP_PATH = Path(__file__).resolve().parent.parent / 'assets'

def choropleth_map():

  # Folium map
  mp = folium.Map(
    location=(59.90, 10.75), 
    zoom_start=10, 
    tiles='Stamen Toner'
  )

  # Choropleth
  fields = [
    {'key': 'municipality', 'label': 'Kommune'},
    {'key': 'postal_code', 'label': 'Poststed'}
  ]

  price = load_price_data()

  for field in fields: 
    df = spatial_price_stats(price, field['key'])
    gdf = load_geo_data(field['key'])

    gdf = gdf.join(df, on=field['key'])

    vmin = df['price_per_area'].min()
    vmax = df['price_per_area'].max()
    colormap = cm.LinearColormap(
      ['gray', 'green', 'yellow', 'red'],
      vmin=0, vmax=vmax,
      index=[0, vmin, (vmax - vmin)/2, vmax]
    ).to_step(6)
    colormap.add_to(mp)

    tooltip = folium.features.GeoJsonTooltip(
      fields=[field['key'], 'price_per_area', 'price_per_area_std'],
      aliases=[field['label'], 'Kvm.pris', 'Std.avvik'],
      style=(
        'background-color: white; color: #333333;'
        'fontfamily: arial; font-size: 12px; padding: 10px'
      )
    )
    folium.GeoJson(gdf.fillna(0), 
      name=field['key'],
      style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['price_per_area']),
        'fillOpacity': 0.7,
        'color': 'black',
        'weight': 0.5
      },
      tooltip=tooltip,
      zoom_on_click=True
    ).add_to(mp)

    #choropleth = folium.Choropleth(
    #  name=f'choro_{field["key"]}',
    #  geo_data=gdf,
    #  data=df,
    #  columns=[field['key'], 'price_per_area', 'price_per_area_std'],
    #  key_on=f'feature.properties.{field["key"]}',
    #  fill_color='YlOrRd',
    #  fill_opacity=0.7,
    #  legend_name='Price/Area',
    #  highlight=True,
    #  line_color='',
    #  overlay=True
    #  #nan_fill_color='#0000'
    #).add_to(mp)
#
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