import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf

def autocorrelation_traces(series: pd.Series, plot_pacf=False) -> list:

  corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)

  lower_y = corr_array[1][:,0] - corr_array[0]
  upper_y = corr_array[1][:,1] - corr_array[0]

  traces = []

  traces.extend([
    go.Scatter(
      x=(x,x), 
      y=(0,corr_array[0][x]), 
      mode='lines',
      line_color='#3f3f3f',
    ) for x in range(len(corr_array[0]))
  ])
  traces.append(
    go.Scatter(
      x=np.arange(len(corr_array[0])), 
      y=corr_array[0], mode='markers', 
      marker_color='#1f77b4',
      marker_size=12,
    )
  )
  traces.append(
    go.Scatter(
      x=np.arange(len(corr_array[0])),
      y=upper_y, mode='lines', 
      line_color='rgba(255,255,255,0)',
    )
  )
  traces.append(
    go.Scatter(
      x=np.arange(len(corr_array[0])), 
      y=lower_y, 
      mode='lines',
      fillcolor='rgba(32, 146, 230,0.3)',
      fill='tonexty', 
      line_color='rgba(255,255,255,0)',
    )
  )
  return traces