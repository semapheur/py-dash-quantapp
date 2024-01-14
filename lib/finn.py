from dateutil.relativedelta import relativedelta
from functools import partial
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
from shapely.geometry import Point

from lib.const import HEADERS
from lib.virdi import choropleth_polys, load_geodata


def finn_data():
  def parse_json(docs: list[dict]):
    scrap = []
    for doc in docs:
      lat = doc['coordinates']['lat']
      lon = doc['coordinates']['lon']

      if int(lat) == 0 or int(lon) == 0:
        continue

      pnt = Point(lon, lat)

      area: dict = doc.get('area_range', doc.get('area'))
      area = area.get('size_from', area.get('size'))

      if area == 0:
        continue

      price = {}
      for sfx in ['total', 'suggestion']:
        temp = doc.get(f'price_{sfx}', doc.get(f'price_range_{sfx}'))
        price[f'price_{sfx}'] = temp.get('amount', temp.get('amount_from'))

      if 0.0 in price.values():
        continue

      shared_cost = doc.get('price_shared_cost', np.nan)
      if isinstance(shared_cost, dict):
        shared_cost = shared_cost.get('amount')

      beds = doc.get('number_of_bedrooms', doc.get('bedrooms_range'))
      if isinstance(beds, dict):
        beds = beds.get('start')

      # Municipality
      # mun = doc['location'].split(', ')[-1]
      # if mun not in setMun:
      #
      #    if mun not in dctMun:
      #        temp = gn.findMunicipality(pnt)['kommunenummer']
      #        dctMun[mun] = temp
      #        mun = temp
      #    else:
      #        mun = dctMun[mun]

      scrap.append(
        {
          'id': doc['ad_id'],
          'time_published': doc['timestamp'],
          'geometry': pnt,
          'address': doc['location'],
          #'municipality': mun,
          'price_total': price['price_total'],
          'price_ask': price['price_suggestion'],
          'shared_cost': shared_cost,
          'area': doc['area_range']['size_from'],
          'bedrooms': beds,
          'property_type': doc['property_type_description'],
          'owner_type': doc['owner_type_description'],
          'link': doc['ad_link'],
        }
      )
    return scrap

  def iterate_pages(scrap, params, startPage):
    for p in range(startPage, 51):
      params[-1] = ('page', str(p))

      with httpx.Client() as client:
        rs = client.get(
          'https://www.finn.no/api/search-qf', headers=HEADERS, params=params
        )
        parse: dict = rs.json()

      if 'docs' not in parse['docs']:
        continue

      scrap.extend(parse_json(parse['docs']))

    if parse['docs']:
      last: dict = parse['docs'][-1]
      price_to = last.get('price_suggestion', last.get('price_range_suggestion'))
      price_to = price_to.get('amount', price_to.get('amount_from'))

    else:
      priceTo = 0

    return scrap, priceTo

  params = {
    'searchkey': 'SEARCH_ID_REALESTATE_HOMES',
    'lifecycle': '1',
    'property_type': ['1', '2', '3', '4'],
    'sort': 'PRICE_ASKING_DESC',
    'price_to': '',
    'page': '1',
  }

  with httpx.Client() as client:
    rs = client.get('https://www.finn.no/api/search-qf', headers=HEADERS, params=params)
    parse = rs.json()

  nUnits = parse['metadata']['result_size']['match_count']

  scrap = []
  scrap.extend(parse_json(parse['docs']))

  scrap, price_to = iterate_pages(scrap, params, 2)

  while (price_to > 0) and (len(scrap) <= nUnits):
    params['price_to'] = str(price_to)
    scrap, price_to = iterate_pages(scrap, params, 1)

  gdf = gpd.GeoDataFrame(scrap, crs=4258)
  gdf.drop_duplicates(inplace=True)
  gdf['priceArea'] = gdf['priceTotal'] / gdf['area']

  # Additional data
  for scope in ('municipality', 'postal_code'):
    path = Path.cwd() / 'data' / 'dgi' / f'virdi_{scope}.json'
    parser = partial(choropleth_polys, scope)
    choro_polys = load_geodata(parser, path, relativedelta(months=6))

    gdf = gdf.sjoin(
      choro_polys[['geometry', scope, f'price_{scope}']], how='left', predicate='within'
    )
    gdf.drop('index_right', axis=1, inplace=True)

    # Price delta
    gdf[f'delta_{scope}'] = gdf['price_area'] - gdf[f'price_{scope}']
    gdf.drop(f'price_{scope}', axis=1, inplace=True)

  return gdf
