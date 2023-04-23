import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

import requests
import json

import re
from pathlib import Path

'''
EPSG:4326 - WGS84 (geographic)
EPSG:3857 - WGS84/Pseudo-Mercator (Web-Mercator)
EPSG:32632 - WGS84 / UTM zone 32N
EPSG:25832 - ETRS89 / UTM zone 32N
EPSG:4258 - ETRS89 (geographic)
'''

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
  'Accept': 'application/json',
  'Accept-Language': 'en-US,en;q=0.5',
  'DNT': '1',
  'Connection': 'keep-alive',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-origin',
  'Sec-GPC': '1',
}

def get_municipalities(): 
  with requests.Session() as s:
    rs = s.get(
      'https://ws.geonorge.no/kommuneinfo/v1/kommuner', 
      headers=HEADERS
    )
    parse = json.loads(rs.text)
  
  rnm = {
    'kommunenavn': 'name',
    'kommunenavnNorsk': 'municipality',
    'kommunenummer': 'id'
  }
  df = pd.DataFrame.from_records(parse)
  df.rename(columns=rnm, inplace=True)
  return df

def search_municipality(query):
  params = {
    'knavn': query,
  }
  with requests.Session() as s:
    rs = s.get(
      'https://ws.geonorge.no/kommuneinfo/v1/sok', 
      headers=HEADERS, params=params
    )
    parse = json.loads(rs.text)

  return parse

def municipality_info(id: str) -> dict:
  with requests.Session() as s:
    rs = s.get(
      f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{id}', 
      headers=HEADERS
    )
    parse = json.loads(rs.text)

  return parse

def find_municipality(point: Point): 
  params = {
    'ost': point.x,
    'koordsys': '4258',
    'nord': point.y,
  }
  with requests.Session() as s:
    rs = s.get(
      'https://ws.geonorge.no/kommuneinfo/v1/punkt', 
      headers=HEADERS, params=params
    )
    parse = json.loads(rs.text)
    
  return parse

def municipality_poly(id: str) -> Polygon:
  url = f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{id}/omrade'
  with requests.Session() as s:
    rs = s.get(url, headers=HEADERS)
    parse = json.loads(rs.text)

  poly = Polygon(parse['omrade']['coordinates'][0][0])
  return poly

def municipality_polys(tolerance: float=0.) -> gpd.GeoDataFrame:
  df = get_municipalities()

  polys = [None] * len(df)
  for i, id in enumerate(df['id']):
    polys[i] = municipality_poly(id)

  gdf = gpd.GeoDataFrame(df, crs=4258, geometry=polys)
  if tolerance > 0:
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=tolerance)

  return gdf

'''
def municipality_polys(municipalities: list):
    
  df_mun = get_municipalities()
  pattern = r'(?<=\()[\w\s]+(?=\)$)'

  def get_feature(munipality):   
    temp = mun.split(' (')[0]
    
    mask = df_mun['kommunenavnNorsk'] == temp
    mun_id = df_mun.loc[mask]
    
    if len(mun_id) > 1:
      cnty = re.search(pattern, mun).group()
      
      hit = None
      for i in range(len(munId)):
        mun_info = municipality_info(mun_id.iloc[i]['kommunenummer'])
        if mun_info['fylkesnavn'] == cnty:
          hit = i
          break
              
      mun_id = mun_id.iloc[i]['kommunenummer']
    
    else:
      mun_id = mun_id['kommunenummer'].squeeze()
    
    url = f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{mun_id}/omrade'
    with requests.Session() as s:
      rs = s.get(url, headers=HEADERS)
      parse = json.loads(rs.text)

    feature = {'properties': {'id': munId, 'municipality': mun}, 'geometry': parse['omrade']}
    return feature

  path = Path.cwd() / 'data' / 'dgi' / 'nor_municipalities.json'

  if not path.exists():
    features = []
    for mun in municipalities:
      features.append(get_feature(mun))
    
    gdf = gpd.GeoDataFrame.from_features(features)
    gdf.set_crs(epsg=4258, inplace=True)
    gdf.to_file(path, driver='GeoJSON', encoding='utf-8')

  else:
    gdf = gpd.read_file(path)

    new_mun = set(municipalities).difference(set(gdf['municipality'].unique()))
    
    if new_mun:
      features = []
      for mun in new_mun:
        features.append(getFeature(mun))

      new_gdf = gpd.GeoDataFrame.from_features(features)
      new_gdf.set_crs(epsg=4258, inplace=True)
      new_gdf['municipality'] = new_mun

      gdf = gdf.append(new_gdf, ignore_index=True)
      gdf.to_file(path, driver='GeoJSON', encoding='utf-8')

  return gdf
'''