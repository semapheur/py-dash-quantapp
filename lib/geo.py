import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import httpx
import numpy as np
from shapely.geometry import Polygon, Point

'''EPSG
3857: Webmercator (projected)
4326: WGS84
'''

def rect_poly(p1: Point, p2: Point) -> Polygon:
  # p1 = (x_min, y_min), p2 = (x_max, y_max)

  p3 = Point(p2.x, p1.y)
  p4 = Point(p1.x, p2.y)

  return Polygon((p1, p3, p2, p4) )


def n_poly(sides: int, point: Point, circumradius: float) -> Polygon:
  return Polygon([
    (
      point.x + np.cos(angle) * circumradius, 
      point.y + np.sin(angle) * circumradius
    ) for angle in np.linspace(0, 2 * np.pi, sides + 1)
  ])

def hextiles(
  polygon: Polygon, 
  circumradius: float,
) -> list[Polygon]:

  # Bounding box
  x_min, y_min, x_max, y_max = polygon.bounds
  
  apothem = 0.5 * np.sqrt(3) * circumradius

  # Calculate the x,y coordinates of the centroids of the hexagons
  x = np.arange(x_min, x_max, 1.5 * circumradius)
  y = np.arange(y_min, y_max, 2 * apothem)

  xx, yy = np.meshgrid(x, y)
  yy[:,::2] -= apothem

  # Create hexagons as shapely polygons
  hexagons = []
  for i in range(len(y) - 1):
    for j in range(len(x) - 1):
      center = Point(xx[i, j], yy[i, j])
      if not center.intersects(polygon):
        continue
      
      hexagon = n_poly(6, center, circumradius)
      if hexagon.intersects(polygon):
        hexagons.append(hexagon)

  return hexagons

def country_poly(
  country: str, 
  save_path: Optional[str|Path] = None,
  crs: Optional[int] = None,
  mask: Optional[Polygon] = None
):
  url = (
    'https://raw.githubusercontent.com/georgique/world-geojson/develop/countries/'
    f'{country.lower()}.json'  
  )
  with httpx.Client() as client:
    rs = client.get(url)
    raw = rs.text
  
  gdf = gpd.read_file(raw, crs=3857)
  gdf = gdf[gdf.geometry.is_valid]

  if crs:
    gdf.to_crs(crs, inplace=True)

  if mask:
    gdf = gdf[gdf.geometry.intersects(mask)]

  if save_path:
    gdf.to_file(save_path, driver='GeoJSON')

  return gdf