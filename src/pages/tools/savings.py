from math import lcm

from dash import callback, dcc, html, no_update, register_page, Input, Output
import numpy as np
import pandas as pd
from plotly import graph_objects as go

from components.input import InputAIO

register_page(
  __name__, path_template="/tools/savings-calculator", title="Savings calculator"
)

layout = html.Main(
  className="size-full",
  children=[
    html.Form(
      className="flex gap-2 pt-2",
      method="",
      children=[
        InputAIO(
          "input:savings:principal",
          input_props={
            "placeholder": "Principal amount",
            "type": "number",
            "value": 1000,
          },
        ),
        InputAIO(
          "input:savings:rate",
          input_props={
            "placeholder": "Annual interest rate (%)",
            "type": "number",
            "value": 5,
          },
        ),
        InputAIO(
          "input:savings:compound-frequency",
          input_props={
            "placeholder": "Annual compound frequency",
            "type": "number",
            "value": 1,
          },
        ),
        InputAIO(
          "input:savings:contribution",
          input_props={
            "placeholder": "Contribution amount",
            "type": "number",
            "value": 0,
          },
        ),
        InputAIO(
          "input:savings:contribution-frequency",
          input_props={
            "placeholder": "Annual contribution frequency",
            "type": "number",
            "value": 1,
          },
        ),
        InputAIO(
          "input:savings:period",
          input_props={
            "placeholder": "Savings period (years)",
            "type": "number",
            "min": 0,
            "value": 10,
          },
        ),
        InputAIO(
          "input:savings:inflation",
          input_props={
            "placeholder": "Annual inflation (%)",
            "type": "number",
            "min": 0,
            "value": 0,
          },
        ),
      ],
    ),
    dcc.Graph(id="graph:savings"),
  ],
)


@callback(
  Output("graph:savings", "figure"),
  Input(InputAIO.aio_id("input:savings:principal"), "value"),
  Input(InputAIO.aio_id("input:savings:rate"), "value"),
  Input(InputAIO.aio_id("input:savings:compound-frequency"), "value"),
  Input(InputAIO.aio_id("input:savings:contribution"), "value"),
  Input(InputAIO.aio_id("input:savings:contribution-frequency"), "value"),
  Input(InputAIO.aio_id("input:savings:inflation"), "value"),
  Input(InputAIO.aio_id("input:savings:period"), "value"),
)
def update_graph(
  principal: float | None,
  rate: float | None,
  compound_frequency: float | None,
  contribution: float | None,
  contribution_frequency: float | None,
  inflation: float | None,
  years: float | None,
):
  if (
    principal is None
    or rate is None
    or compound_frequency is None
    or contribution is None
    or contribution_frequency is None
    or inflation is None
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

  inflation_per_period = (
    0 if inflation == 0 else (1 + (inflation / 100)) ** (1 / steps_per_year) - 1
  )

  t = np.linspace(0, years, steps)

  principal_array = np.full(steps, principal)
  balance = np.zeros(steps)
  contributions = np.zeros(steps)
  interest = np.zeros(steps)
  real_balance = np.zeros(steps)

  balance[0] = principal
  real_balance[0] = principal

  contribution_period = steps_per_year / contribution_frequency
  compound_period = steps_per_year / compound_frequency

  pending_interest = 0

  for step in range(1, steps):
    if abs(step % contribution_period) < 1e-10:
      contributions[step] = contributions[step - 1] + contribution
    else:
      contributions[step] = contributions[step - 1]

    step_interest = balance[step - 1] * (rate / 100) / steps_per_year
    pending_interest += step_interest

    if abs(step % compound_period) < 1e-10:
      interest[step] = interest[step - 1] + pending_interest
      pending_interest = 0
    else:
      interest[step] = interest[step - 1]

    balance[step] = principal + contributions[step] + interest[step]
    real_balance[step] = balance[step] / (1 + inflation_per_period) ** step

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
    y=contributions,
    mode="lines",
    name="Contributions",
    line_shape="hv",
    stackgroup="balance",
  )
  fig.add_scatter(
    x=t,
    y=interest,
    mode="lines",
    name="Interest",
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
  fig.add_scatter(
    x=t,
    y=real_balance,
    mode="lines",
    name="Real balance",
    line_shape="hv",
  )

  fig.update_layout(
    title="Accumulated savings",
    xaxis_title="Years",
    yaxis_title="Value",
    hovermode="x unified",
  )

  return fig
