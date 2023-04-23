import requests
import json

import geopandas as gpd
from shapely.geometry import Point, Polygon

from lib.db import DB_DIR
from lib.geonorge import municipality_polys

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0',
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.5',
  'DNT': '1',
  'Connection': 'keep-alive',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-origin',
  'Sec-GPC': '1',
  'TE': 'trailers',
}

VIRDI_PATH = DB_DIR / 'hjemla.json'

def postal_code_polys():
  with requests.Session() as s:
    rs = s.get(
      'https://raw.githubusercontent.com/ivanhjel/postnummer/master/postnummeromrader.geojson', 
      headers=HEADERS
    )
    raw = rs.text
  
  rnm = {
    'kommune': 'municipality',
    'postnummer': 'postal_code',
    'poststedsnavn': 'postal_area'
  }
  gdf = gpd.read_file(raw, driver='GeoJSON', encoding='utf-8')
  gdf.rename(columns=rnm, inplace=True)
  return gdf

def real_estate_price_data(
  size_range=(30,150),
  sw_coord=(57.8, 4.3), 
  ne_coord=(71.2, 31.3),
):

  params = {
    'period': '12',
    'sizemin': '0',
    'sizemax': '500',
    'adstates': 'sold,forsale,comingforsale',
    'unittypes': 'apartment,semidetatchedhouse,house,serialhouse',
    'swLat': sw_coord[0],
    'swLng': sw_coord[1], 
    'neLat': ne_coord[0],
    'neLng': ne_coord[1],
  }
    
  scrap = []
  for size in range(size_range[0], size_range[1]):
    params['sizemin'] = str(size)
    params['sizemax'] = str(size)
    
    with requests.Session() as s:
      rs = s.get(
        'https://consumer-service.hjemla.no/public/maps/units', 
        headers=HEADERS, params=params
      )
      parse = json.loads(rs.text)

    if 'response' not in parse:
      continue

    for street in parse['response']:
      pnt = Point(street['coordinatesLng'], street['coordinatesLat'])  
      for unit in street['units']:
        scrap.append({
          'id': unit['id'],
          'geometry': pnt,
          'municipality': street['municipalityName'],
          'postal_code': street['postalCode'],
          'street_id': street['streetId'],
          'address': street['slug'],
          'flat': unit.get('floorCode', ''),
          'area': size,
          'estimated_price': unit['estimatedPrice'],
          'estimated_common_debt': unit['estimateCommonDebt'],
          'fixed_price': unit['fixedPrice'],
          'common_debt': unit['commonDebt'],
          'asking_price': unit['askingPrice']
        })
    
  df = gpd.GeoDataFrame(scrap, crs=4258)
  
  cols = ['estimated_price', 'fixed_price', 'asking_price']
  df.dropna(how='all', subset=cols, inplace=True)
  
  df['price_per_area'] = df[cols].max(axis=1) / df['area']
  
  return df

def spatial_price_stats(unit='municipality'):
  # unit: municipality / postal_code

  if unit not in {'municipality', 'postal_code'}:
    raise Exception('Unit only accepts values "municipality"/"postal_code"')

  if not VIRDI_PATH.exists():
    price = real_estate_price_data()
    price.to_file(VIRDI_PATH)
  else:
    price = real_estate_price_data()

  price = price[[unit, 'price_per_area']]
  stats = price.groupby([unit], as_index=False).agg(
    price_per_area=('price_per_area', 'mean'), 
    price_per_area_std=('price_per_area', 'std')
  )
  return stats