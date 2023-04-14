import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import probplot
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import acf, pacf
from datetime import datetime as dt

def acf_trace(data: np.ndarray, plot_pacf=False) -> list:
  corr_array = pacf(data, alpha=0.05) if plot_pacf else acf(data, alpha=0.05)

  lower_y = corr_array[1][:,0] - corr_array[0]
  upper_y = corr_array[1][:,1] - corr_array[0]

  trace = [
    go.Scatter(
      x=(x,x), 
      y=(0,corr_array[0][x]), 
      mode='lines',
      line_color='#3f3f3f',
    ) for x in range(len(corr_array[0]))
  ] + [
    go.Scatter(
      x=np.arange(len(corr_array[0])), 
      y=corr_array[0], mode='markers', 
      marker_color='#1f77b4',
      marker_size=12,
    ),
    go.Scatter(
      x=np.arange(len(corr_array[0])),
      y=upper_y, mode='lines', 
      line_color='rgba(255,255,255,0)',
    ),
    go.Scatter(
      x=np.arange(len(corr_array[0])), 
      y=lower_y, 
      mode='lines',
      fillcolor='rgba(32, 146, 230,0.3)',
      fill='tonexty', 
      line_color='rgba(255,255,255,0)',
    )
  ]
  return trace

def qqplot_trace(data: np.ndarray, dist: str, params: tuple = ()) -> list:
  qq = probplot(data, dist=dist, sparams=params)
  x = np.array([qq[0][0][0], qq[0][0][-1]])
  y = qq[1][1] + qq[1][0] * x

  trace = [
    go.Scatter(
      x=qq[0][0],
      y=qq[0][1],
      #name='Sample quantiles',
      mode='markers',
      marker_color='#19d3f3'
    ),
    go.Scatter(
      x=x,
      y=y,
      #name='Theoretical quantiles',
      mode='lines',
      line_color='#636efa'
    )
  ]
  return trace

def msdr_trace(data: pd.Series, regimes: int) -> list:
  # Markov switching dynamic regression
  msdr = MarkovRegression(data.values, k_regimes=regimes, trend='c')
  result = msdr.fit()

  state = np.argmax(result.smoothed_marginal_probabilities, axis=1)

  trace = [
    go.Scatter(
      x=data.index,
      y=result.smoothed_marginal_probabilities,
      mode='lines'
    )
  ]
  for r in range(regimes):
    temp = data.loc[state == r]

    trace.append(
      go.Scatter(
        x=temp.index,
        y=temp,
        mode='lines'
      )
    )
  
  return trace