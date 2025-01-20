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
  tax_rate: float | None,
  deduction_rate: float | None,
  years: float | None,
):
  if (
    principal is None
    or contribution is None
    or frequency is None
    or returns is None
    or fees is None
    or tax_rate is None
    or deduction_rate is None
    or years is None
  ):
    return no_update

  years = int(years)
  frequency = int(frequency)
  steps = years * frequency + 1
  t = np.linspace(0, years, steps)

  principals = np.full(steps, principal)
  balance = np.zeros(steps)
  tax = np.zeros(steps)
  gain = np.zeros(steps)
  real_gain = np.zeros(steps)

  balance[0] = principal

  annual_return = (returns - fees) / 100
  effective_periodic_return = (1 + annual_return) ** (1 / frequency) - 1
  tax_rate = tax_rate / 100
  deduction_rate = deduction_rate / 100

  deduction_basis = principal * (1 + deduction_rate)
  deduction = deduction_basis * deduction_rate

  contribution_made_this_year = False

  for i in range(1, len(t)):
    balance[i] = balance[i - 1] * (1 + effective_periodic_return) + contribution
    principals[i] = principals[i - 1] + contribution

    if i % frequency == 0:
      if not contribution_made_this_year:
        deduction_basis = (principals[i] + deduction) * (1 + deduction_rate)
        deduction += deduction_basis * deduction_rate
        contribution_made_this_year = True

    else:
      if i % frequency == 0:
        contribution_made_this_year = False

    gain[i] = balance[i] - principals[i]
    tax[i] = max(0, gain[i] - deduction) * tax_rate
    real_gain[i] = gain[i] - tax[i]

  fig = go.Figure()
  fig.add_scatter(
    x=t,
    y=principals,
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
    y=tax,
    mode="lines",
    name="Tax",
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

  fig.update_layout(
    title="Accumulated returns",
    xaxis_title="Years",
    yaxis_title="Value",
    hovermode="x unified",
  )
  return fig
