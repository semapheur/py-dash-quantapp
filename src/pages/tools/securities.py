from dash import callback, dcc, html, no_update, register_page, Input, Output
import numpy as np
import pandas as pd
from plotly import graph_objects as go

from components.input import InputAIO

register_page(
  __name__, path_template="/tools/securities-calculator", title="Securities calculator"
)

layout = html.Main(
  className="size-full",
  children=[
    html.Form(
      className="flex gap-2 pt-2",
      method="",
      children=[
        InputAIO(
          "input:securities:principal",
          input_props={
            "placeholder": "Principal amount",
            "type": "number",
            "value": 1000,
          },
        ),
        InputAIO(
          "input:securities:contribution",
          input_props={
            "placeholder": "Contribution amount",
            "type": "number",
            "value": 0,
          },
        ),
        InputAIO(
          "input:securities:contribution-frequency",
          input_props={
            "placeholder": "Annual contribution frequency",
            "type": "number",
            "value": 1,
          },
        ),
        InputAIO(
          "input:securities:returns",
          input_props={
            "placeholder": "Annual returns (%)",
            "type": "number",
            "value": 10,
          },
        ),
        InputAIO(
          "input:securities:fees",
          input_props={
            "placeholder": "Annual fees (%)",
            "type": "number",
            "value": 0.5,
          },
        ),
        InputAIO(
          "input:securities:tax",
          input_props={
            "placeholder": "Capital gains tax (%)",
            "type": "number",
            "value": 37.84,
          },
        ),
        InputAIO(
          "input:securities:deduction",
          input_props={
            "placeholder": "Annual risk-free deduction (%)",
            "type": "number",
            "value": 2,
          },
        ),
        InputAIO(
          "input:securities:period",
          input_props={
            "placeholder": "Investment period (years)",
            "type": "number",
            "min": 0,
            "value": 10,
          },
        ),
      ],
    ),
    dcc.Graph(id="graph:securities"),
  ],
)


@callback(
  Output("graph:securities", "figure"),
  Input(InputAIO.aio_id("input:securities:principal"), "value"),
  Input(InputAIO.aio_id("input:securities:contribution"), "value"),
  Input(InputAIO.aio_id("input:securities:contribution-frequency"), "value"),
  Input(InputAIO.aio_id("input:securities:returns"), "value"),
  Input(InputAIO.aio_id("input:securities:fees"), "value"),
  Input(InputAIO.aio_id("input:securities:tax"), "value"),
  Input(InputAIO.aio_id("input:securities:deduction"), "value"),
  Input(InputAIO.aio_id("input:securities:period"), "value"),
)
def update_graph(
  principal: float | None,
  contribution: float | None,
  frequency: float | None,
  returns: float | None,
  fees: float | None,
  tax: float | None,
  deduction: float | None,
  years: float | None,
):
  if (
    principal is None
    or contribution is None
    or frequency is None
    or returns is None
    or fees is None
    or tax is None
    or deduction is None
    or years is None
  ):
    return no_update

  years = int(years)
  frequency = int(frequency)
  steps = years * frequency + 1
  t = np.linspace(0, years, steps)

  principal_array = np.full(steps, principal)
  balance = np.zeros(steps)
  gain = np.zeros(steps)
  real_gain = np.zeros(steps)

  balance[0] = principal

  annual_return = (returns - fees) / 100
  effective_periodic_return = (1 + annual_return) ** (1 / frequency) - 1
  effective_tax = (tax - deduction) / 100

  for i in range(1, len(t)):
    balance[i] = balance[i - 1] * (1 + effective_periodic_return)
    if i % frequency == 0:
      balance[i] += contribution

    if i % frequency == 0:
      effective_tax = (tax - (deduction * (i // frequency))) / 100

    gain[i] = balance[i] - principal
    real_gain[i] = gain[i] * (1 - effective_tax)

  fig = go.Figure()
  fig.add_scatter(
    x=t,
    y=principal_array,
    mode="lines",
    name="Principal",
    line_shape="hv",
    stackgroup="balance",
  )
  fig.add_scatter(
    x=t,
    y=real_gain,
    mode="lines",
    name="Real gain",
    line_shape="hv",
    stackgroup="balance",
  )
  fig.add_scatter(
    x=t,
    y=balance,
    mode="lines",
    name="Balance",
    line_shape="hv",
  )
  return fig
