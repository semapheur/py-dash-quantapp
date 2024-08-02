from math import lcm

from dash import callback, dcc, html, no_update, register_page, Input, Output
from numba import jit
import numpy as np
from plotly import express as px

from components.input import InputAIO

register_page(__name__, path_template="/interest", title="Interest rate calculator")

layout = html.Main(
  className="h-full",
  children=[
    html.Form(
      className="flex gap-2 pt-2",
      method="",
      children=[
        InputAIO(
          "input:interest:principal",
          input_props={
            "placeholder": "Principal amount",
            "type": "number",
            "value": 1000,
          },
        ),
        InputAIO(
          "input:interest:rate",
          input_props={
            "placeholder": "Nominal interest rate (%)",
            "type": "number",
            "value": 5,
          },
        ),
        InputAIO(
          "input:interest:compound-frequency",
          input_props={
            "placeholder": "Compound frequency",
            "type": "number",
            "value": 1,
          },
        ),
        InputAIO(
          "input:interest:contribution",
          input_props={
            "placeholder": "Contribution amount",
            "type": "number",
            "value": 0,
          },
        ),
        InputAIO(
          "input:interest:contribution-frequency",
          input_props={
            "placeholder": "Contribution frequency",
            "type": "number",
            "value": 1,
          },
        ),
        InputAIO(
          "input:interest:period",
          input_props={
            "placeholder": "Period (years)",
            "type": "number",
            "min": 0,
            "value": 10,
          },
        ),
      ],
    ),
    dcc.Graph(id="graph:interest"),
  ],
)


@jit(nopython=True)
def calculate_growth(
  principal: float,
  rate: float,
  contribution: float,
  contribution_period: float,
  compound_period: float,
  steps: int,
):
  balance = np.zeros(steps)
  contributions = np.zeros(steps)
  interest = np.zeros(steps)

  balance[0] = principal

  for event in range(1, steps):
    # Add contribution if it's a contribution event
    if abs(event % contribution_period) < 1e-10:
      balance[event] = balance[event - 1] + contribution
      contributions[event] = contributions[event - 1] + contribution
    else:
      balance[event] = balance[event - 1]
      contributions[event] = contributions[event - 1]

    # Apply compound interest if it's a compounding event
    if abs(event % compound_period) < 1e-10:
      interest_amount = balance[event] * rate / (steps / compound_period)
      balance[event] += interest_amount
      interest[event] = interest[event - 1] + interest_amount
    else:
      interest[event] = interest[event - 1]

    return balance, contributions, interest


@jit(nopython=True)
def calculate_growth(
  principal: float,
  rate: float,
  contribution: float,
  contribution_period: float,
  compound_period: float,
  compound_frequency: float,
  steps: int,
):
  balance = np.zeros(steps)
  contributions = np.zeros(steps)
  interest = np.zeros(steps)

  balance[0] = principal

  for step in range(1, steps):
    if abs(step % contribution_period) < 1e-10:
      balance[step] = balance[step - 1] + contribution
      contributions[step] = contributions[step - 1] + contribution
    else:
      balance[step] = balance[step - 1]
      contributions[step] = contributions[step - 1]

    if abs(step % compound_period) < 1e-10:
      interest_amount = balance[step] * rate / compound_frequency
      balance[step] += interest_amount
      interest[step] = interest[step - 1] + interest_amount
    else:
      interest[step] = interest[step - 1]

    return balance, contributions, interest


@callback(
  Output("graph:interest", "figure"),
  Input(InputAIO.aio_id("input:interest:principal"), "value"),
  Input(InputAIO.aio_id("input:interest:rate"), "value"),
  Input(InputAIO.aio_id("input:interest:compound-frequency"), "value"),
  Input(InputAIO.aio_id("input:interest:contribution"), "value"),
  Input(InputAIO.aio_id("input:interest:contribution-frequency"), "value"),
  Input(InputAIO.aio_id("input:interest:period"), "value"),
)
def update_graph(
  principal: float | None,
  rate: float | None,
  compound_frequency: float | None,
  contribution: float | None,
  contribution_frequency: float | None,
  years: float | None,
):
  if (
    principal is None
    or rate is None
    or compound_frequency is None
    or contribution is None
    or contribution_frequency is None
    or years is None
  ):
    return no_update

  contrib_num, contrib_denom = contribution_frequency.as_integer_ratio()
  compound_num, compound_denom = compound_frequency.as_integer_ratio()

  lcm_denom = lcm(contrib_denom, compound_denom)

  steps_per_year = lcm(
    contrib_num * (lcm_denom // contrib_denom),
    compound_num * (lcm_denom // compound_denom),
  )
  steps = int(years * steps_per_year) + 1

  t = np.linspace(0, years, steps)

  balance = np.zeros(steps)
  contributions = np.zeros(steps)
  interest = np.zeros(steps)

  balance[0] = principal

  contribution_period = steps_per_year / contribution_frequency
  compound_period = steps_per_year / compound_frequency

  for step in range(1, steps):
    if abs(step % contribution_period) < 1e-10:
      balance[step] = balance[step - 1] + contribution
      contributions[step] = contributions[step - 1] + contribution
    else:
      balance[step] = balance[step - 1]
      contributions[step] = contributions[step - 1]

    if abs(step % compound_period) < 1e-10:
      interest_amount = balance[step] * (rate / 100) / compound_frequency
      balance[step] += interest_amount
      interest[step] = interest[step - 1] + interest_amount
    else:
      interest[step] = interest[step - 1]

  fig = px.line(x=t, y=balance, title="Compound interest", line_shape="hv")

  fig.update_layout(xaxis_title="Years", yaxis_title="Value")

  return fig
