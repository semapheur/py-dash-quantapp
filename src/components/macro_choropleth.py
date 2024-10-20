from dash import dcc
import numpy as np
import plotly.graph_objects as go

from lib.world_bank import load_wdi


def MacroChoropleth(className: str) -> dcc.Graph:
  df = load_wdi(years=1)
  gdpcg = df["gdpcg"].iloc[0]

  fig = go.Figure(
    data=go.Choropleth(
      locations=gdpcg.index,
      z=gdpcg,
      customdata=np.stack(
        (
          df["pop"].iloc[0],
          df["gdp_cd"].iloc[0],
          df["gdpcg"].iloc[0],
          df["gdpc_ppp_cd"].iloc[0],
          df["cpi"].iloc[0],
        ),
        axis=-1,
      ),
      colorscale="RdBu",
      marker_line_color="black",
      colorbar=dict(len=1, orientation="h", thickness=10, xpad=150, y=1.05),
      hovertemplate=(
        "<b>%{location}</b><br>"
        "<b>Population:</b> %{customdata[0]:.2E}<br>"
        "<b>GDP:</b> %{customdata[1]:$.2E}<br>"
        "<b>GDP/C growth:</b> %{customdata[2]:.1f}%<br>"
        "<b>GDP/C PPP:</b> %{customdata[3]:$,.0f}<br>"
        "<b>CPI:</b> %{customdata[4]:.1f}%<br>"
      ),
    )
  )
  fig.update_layout(
    geo=dict(projection_type="equirectangular"),
    margin={"r": 10, "t": 10, "l": 10, "b": 10},
    updatemenus=[
      dict(
        buttons=list(
          [
            dict(
              args=[{"z": [df["gdpcg"].iloc[0]]}], label="GPD/C growth", method="update"
            ),
            dict(args=[{"z": [df["cpi"].iloc[0]]}], label="CPI", method="update"),
          ]
        ),
        direction="down",
        showactive=True,
        x=0,
        xanchor="left",
        y=1.2,
        yanchor="top",
      )
    ],
  )

  return dcc.Graph(figure=fig, className=className)
