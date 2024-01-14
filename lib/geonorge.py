import aiometer
from functools import partial

import geopandas as gpd
import pandas as pd
import httpx
from shapely.geometry import Point, Polygon
import topojson as tj

from lib.const import HEADERS

"""
EPSG:4326 - WGS84 (geographic)
EPSG:3857 - WGS84/Pseudo-Mercator (Web-Mercator)
EPSG:32632 - WGS84 / UTM zone 32N
EPSG:25832 - ETRS89 / UTM zone 32N
EPSG:4258 - ETRS89 (geographic)
"""


def get_municipalities() -> pd.DataFrame:
  with httpx.Client() as client:
    rs = client.get('https://ws.geonorge.no/kommuneinfo/v1/kommuner', headers=HEADERS)
    parse = rs.json()

  rnm = {
    'kommunenavn': 'name',
    'kommunenavnNorsk': 'municipality',
    'kommunenummer': 'id',
  }
  df = pd.DataFrame.from_records(parse)
  df.rename(columns=rnm, inplace=True)
  return df


def search_municipality(query: str) -> dict:
  params = {
    'knavn': query,
  }
  with httpx.Client() as client:
    rs = client.get(
      'https://ws.geonorge.no/kommuneinfo/v1/sok', headers=HEADERS, params=params
    )
    parse = rs.json()

  return parse


def municipality_info(id: str) -> dict:
  with httpx.Client() as client:
    rs = client.get(
      f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{id}', headers=HEADERS
    )
    parse = rs.json()

  return parse


def find_municipality(point: Point):
  params = {
    'ost': point.x,
    'koordsys': '4258',
    'nord': point.y,
  }
  with httpx.Client() as client:
    rs = client.get(
      'https://ws.geonorge.no/kommuneinfo/v1/punkt', headers=HEADERS, params=params
    )
    parse = rs.json()

  return parse


async def municipality_poly(id: str) -> Polygon:
  url = f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{id}/omrade'
  limits = httpx.Limits(max_connections=10)
  async with httpx.AsyncClient(limits=limits) as client:
    rs = await client.get(url, headers=HEADERS)
    parse = rs.json()

  poly = Polygon(parse['omrade']['coordinates'][0][0])
  return poly


async def municipality_polys(tolerance: float = 0.0) -> gpd.GeoDataFrame:
  df = get_municipalities()

  # tasks = [asyncio.create_task(municipality_poly(id)) for id in df['id']]
  # polys = await asyncio.gather(*tasks)
  tasks = [partial(municipality_poly, id) for id in df['id']]
  polys = await aiometer.run_all(tasks, max_per_second=10)

  gdf = gpd.GeoDataFrame(df, crs=4258, geometry=polys)
  if tolerance > 0:
    topo = tj.Topology(gdf, prequantize=False)
    gdf = topo.toposimplify(tolerance).to_gdf()
    # gdf['geometry'] = gdf['geometry'].simplify(tolerance=tolerance)

  return gdf
