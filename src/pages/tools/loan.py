from dash import callback, dcc, html, no_update, register_page, Input, Output
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from components.input import InputAIO

register_page(__name__, path_template="/tools/loan-calculator", title="Loan calculator")

layout = html.Main(
  className="size-full",
  children=[
    html.Form(
      className="flex gap-2 pt-2",
      method="",
      children=[
        InputAIO(
          "input:loan:amount",
          input_props={
            "placeholder": "Loan amount",
            "type": "number",
            "value": 1000000,
          },
        ),
        InputAIO(
          "input:loan:rate",
          input_props={
            "placeholder": "Interest rate (%)",
            "type": "number",
            "value": 5,
          },
        ),
        InputAIO(
          "input:loan:compound-frequency",
          input_props={
            "placeholder": "Interest frequency (per year)",
            "type": "number",
            "step": 1,
            "value": 12,
          },
        ),
        InputAIO(
          "input:loan:term",
          input_props={
            "placeholder": "Loan term (years)",
            "type": "number",
            "step": 1,
            "min": 0,
            "value": 10,
          },
        ),
        InputAIO(
          "input:loan:payment-frequency",
          input_props={
            "placeholder": "Payment frequency (per year)",
            "type": "number",
            "step": 1,
            "value": 12,
          },
        ),
        InputAIO(
          "input:loan:inflation",
          input_props={
            "placeholder": "Annual inflation (%)",
            "type": "number",
            "value": 0,
          },
        ),
      ],
    ),
    dcc.Graph(id="graph:loan", className="h-full"),
  ],
)


@callback(
  Output("graph:loan", "figure"),
  Input(InputAIO.aio_id("input:loan:amount"), "value"),
  Input(InputAIO.aio_id("input:loan:rate"), "value"),
  Input(InputAIO.aio_id("input:loan:compound-frequency"), "value"),
  Input(InputAIO.aio_id("input:loan:term"), "value"),
  Input(InputAIO.aio_id("input:loan:payment-frequency"), "value"),
  Input(InputAIO.aio_id("input:loan:inflation"), "value"),
)
def update_graph(
  loan_amount: float | None,
  rate: float | None,
  compound_frequency: int | None,
  loan_term: int | None,
  payment_frequency: int | None,
  inflation: float | None,
):
  if (
    loan_amount is None
    or rate is None
    or compound_frequency is None
    or loan_term is None
    or payment_frequency is None
    or inflation is None
  ):
    return no_update

  inflation_per_period = (
    0 if inflation == 0 else (1 + (inflation / 100)) ** (1 / payment_frequency) - 1
  )

  rate_per_period = (1 + (rate / 100) / compound_frequency) ** (
    compound_frequency / payment_frequency
  ) - 1
  num_payments = loan_term * payment_frequency
  payment = (
    loan_amount
    * (rate_per_period * (1 + rate_per_period) ** num_payments)
    / ((1 + rate_per_period) ** num_payments - 1)
  )

  balance = np.zeros(num_payments + 1)
  interest = np.zeros(num_payments + 1)
  principal = np.zeros(num_payments + 1)
  real_payment = np.zeros(num_payments + 1)

  balance[0] = loan_amount

  for period in range(1, num_payments + 1):
    interest[period] = balance[period - 1] * rate_per_period
    principal[period] = payment - interest[period]
    balance[period] = balance[period - 1] - principal[period]

    real_payment[period] = payment / (1 + inflation_per_period) ** period

  balance[-1] = 0
  total_cost = principal.cumsum() + interest.cumsum()
  real_cost = real_payment.cumsum()

  t = np.linspace(0, loan_term, num_payments + 1)

  fig = make_subplots(rows=2, cols=1, subplot_titles=("Loan cost", "Loan payments"))
  fig.append_trace(
    go.Scatter(
      x=t,
      y=principal.cumsum(),
      mode="lines",
      line_shape="hv",
      name="Principal",
      stackgroup="cost",
    ),
    row=1,
    col=1,
  )
  fig.append_trace(
    go.Scatter(
      x=t,
      y=interest.cumsum(),
      mode="lines",
      line_shape="hv",
      name="Interest",
      stackgroup="cost",
    ),
    row=1,
    col=1,
  )
  fig.append_trace(
    go.Scatter(x=t, y=total_cost, mode="lines", line_shape="hv", name="Nominal cost"),
    row=1,
    col=1,
  )
  fig.append_trace(
    go.Scatter(x=t, y=real_cost, mode="lines", line_shape="hv", name="Real cost"),
    row=1,
    col=1,
  )
  fig.append_trace(
    go.Scatter(x=t, y=balance, mode="lines", line_shape="hv", name="Balance"),
    row=1,
    col=1,
  )
  fig.append_trace(
    go.Bar(
      x=t,
      y=principal,
      name="Principal",
    ),
    row=2,
    col=1,
  )
  fig.append_trace(
    go.Bar(
      x=t,
      y=interest,
      name="Interest",
    ),
    row=2,
    col=1,
  )
  fig.update_layout(barmode="stack", hovermode="x unified")

  return fig
