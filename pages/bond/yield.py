import asyncio

from dash import (
  callback,
  ctx,
  dcc,
  html,
  no_update,
  register_page,
  Output,
  Input,
  State,
)
from plotly import graph_objects as go
import pandas as pd

from lib.fred.data_series import DataSeries


async def yield_curve():
  treasury_series = (
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30",
  )

  tasks = [
    asyncio.create_task(DataSeries(sid).time_series()) for sid in treasury_series
  ]
  dfs = await asyncio.gather(*tasks)

  return pd.concat(dfs, axis=1)


register_page(__name__, path_template="/bond/yield", title="Yield curve")

df = asyncio.run(yield_curve())
years = df.index.max().year - df.index.min().year

figure = go.Figure(data=go.Surface(x=df.columns, y=df.index, z=df.to_numpy()))
figure.update_layout(
  scene={
    "xaxis": {"autorange": "reversed"},
    # "yaxis": {"nticks": years // 5},
    # "camera_eye": {"x": 0, "y": -1, "z": 0.5},
    "aspectratio": {"x": 1, "y": 3, "z": 1},
  }
)

layout = html.Main(
  className="h-full",
  children=[dcc.Graph(id="graph:yield_curve", className="h-full", figure=figure)],
)
