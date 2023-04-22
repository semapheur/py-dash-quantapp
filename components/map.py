import folium # MapQuest Open|Stamen Toner|Carto DB positron/dark_matter

MAP_PATH = Path(__file__).resolve().parent.parent / 'assets'

def choropleth_map():
  # Folium map
  mp = folium.Map(location=[59.90, 10.75], zoom_start=10, tiles='Stamen Toner')

  # Choropleth
  fields = {
    'municipality': {
      'key': 'municipality',
      'label': 'Kommune'
    },
    'postalarea': {
      'key': 'postalarea',
      'lebal': 'Poststed'
    }
  }

  for unit in fields.key(): 
    gdf = price_choropleth(unit)

    choropleth = folium.Choropleth(
      name=f'choro_{unit}',
      geo_data=gdf,
      data=gdf,
      columns=[fields[unit]['key'], f'price_{unit}', f'price_{unit}_std'],
      key_on=f'feature.properties.{fields[unit]["key"]}',
      fill_color='YlOrRd',
      fill_opacity=0.7,
      legend_name='Price/Area',
      highlight=True,
      line_color='',
      overlay=True
      #nan_fill_color='#0000'
    ).add_to(mp)

    # Tooltip
    choropleth.geojson.add_child(
      folium.features.GeoJsonTooltip(
        fields=[fields[unit]['key'], f'price_{unit}', f'price_{unit}_std'],
        aliases=[fields[unit]['alias'], 'Kvm.pris', 'Std.avvik'],
        style=(
          'background-color: white; color: #333333;'
          'fontfamily: arial; font-size: 12px; padding: 10px'
        )
      )
    )

  # Tile layer
  folium.TileLayer('OpenStreetMap', control=True).add_to(mp)
  folium.TileLayer('CartoDb dark_matter', control=True).add_to(mp)

  # Layer control
  folium.LayerControl(collapsed=False).add_to(mp)

  mp.save(MAP_PATH / 'choropleth.html') # mp._repr_html_()