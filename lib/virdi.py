import requests

from typing import Literal, TypedDict

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from lib.const import DB_DIR, STATIC_DIR, HEADERS
from lib.geonorge import municipality_polys
from lib.utils import update_json

VIRDI_PATH = DB_DIR / 'hjemla.json'

class Virdi(TypedDict):
  id: int
  geometry: Point
  municipality: str
  postal_code: int
  street_id: int
  address: str
  flat: str
  area: float
  estimated_price: float
  estimated_common_debt: float
  fixed_price: float
  common_debt: float
  asking_price: float

def load_geo_data(unit: str) -> gpd.GeoDataFrame:
  path = DB_DIR / f'nor_{unit}.json'
  if not path.exists():
    if unit == 'municipality':
      gdf = municipality_polys(0.001)
    elif unit == 'postal_code':
      gdf = postal_code_polys()

    gdf.to_file(path, driver='GeoJSON', encoding='utf-8')
  
  else:
    gdf = gpd.read_file(path, driver='GeoJSON', encoding='utf-8')
  
  return gdf

def postal_code_polys() -> gpd.GeoDataFrame:
  url = (
    'https://raw.githubusercontent.com/'
    'ivanhjel/postnummer/master/postnummeromrader.geojson'
  )

  with requests.Session() as s:
    rs = s.get(url, headers=HEADERS)
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
  size_range: tuple = (30,150),
  sw_coord: tuple = (57.8, 4.3), 
  ne_coord: tuple = (71.2, 31.3),
) -> gpd.GeoDataFrame:

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
    
  scrap: list[Virdi] = []
  for size in range(size_range[0], size_range[1]):
    params['sizemin'] = str(size)
    params['sizemax'] = str(size)
    
    with requests.Session() as s:
      rs = s.get(
        'https://consumer-service.hjemla.no/public/maps/units', 
        headers=HEADERS, params=params
      )
      parse = rs.json()

    if 'response' not in parse:
      continue

    for street in parse['response']:
      pnt = Point(street['coordinatesLng'], street['coordinatesLat'])  
      for unit in street['units']:
        scrap.append({
          'id': int(unit['id']),
          'geometry': pnt,
          'municipality': street['municipalityName'],
          'postal_code': int(street.get('postalCode', np.nan)),
          'street_id': int(street['streetId']),
          'address': street['slug'],
          'flat': unit.get('floorCode', ''),
          'area': size,
          'estimated_price': unit['estimatedPrice'],
          'estimated_common_debt': unit['estimateCommonDebt'],
          'fixed_price': unit['fixedPrice'],
          'common_debt': unit['commonDebt'],
          'asking_price': unit['askingPrice']
        })
    
  gdf = gpd.GeoDataFrame(scrap, crs=4258)
  
  cols = ['estimated_price', 'fixed_price', 'asking_price']
  gdf.dropna(how='all', subset=cols, inplace=True)
  
  gdf['price_per_area'] = gdf[cols].max(axis=1) / gdf['area']
  
  return gdf

def load_price_data():
  if not VIRDI_PATH.exists():
    price = real_estate_price_data()
    price.to_file(VIRDI_PATH, driver='GeoJSON')
  else:
    price = gpd.read_file(VIRDI_PATH, encoding='utf-8')

  return price

def spatial_price_stats(
  price: gpd.GeoDataFrame, 
  unit: Literal['municipality', 'postal_code']
) -> pd.DataFrame:

  if unit not in {'municipality', 'postal_code'}:
    raise Exception('Unit only accepts values "municipality"/"postal_code"')

  price = price[[unit, 'price_per_area']]
  stats = price.groupby([unit]).agg(
    price_per_area=('price_per_area', 'mean'), 
    price_per_area_std=('price_per_area', 'std')
  )
  return stats

def choropleth_polys(unit: Literal['municipality', 'postal_code']):
  price = load_price_data()

  df = spatial_price_stats(price, unit)
  gdf = load_geo_data(unit)
  gdf = gdf.join(df, on=unit)
  #gdf = gdf[['geometry', 'postal_code', 'price_per_area', 'price_per_area_std']]

  extrema = {
    unit: [
      df['price_per_area'].min(),
      df['price_per_area'].max()
    ]
  }
  path = DB_DIR / 'colorbar_values.json'
  update_json(path, extrema)

  path = STATIC_DIR / f'realestate_choro_{unit}.json'
  gdf.to_file(path, driver='GeoJSON', encoding='utf-8')