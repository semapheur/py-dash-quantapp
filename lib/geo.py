import numpy as np
from shapely.geometry import Polygon, Point

def n_polygon(sides: int, point: Point, circumradius: float) -> Polygon:
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
      hexagon = n_polygon(6, center, circumradius)
      if hexagon.intersects(polygon):
        hexagons.append(hexagon)

  return hexagons