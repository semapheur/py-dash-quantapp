import json

from dash import (
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
  MATCH,
)

from components.input import InputAIO

from lib.db.lite import read_sqlite
from lib.ticker.fetch import stored_companies

register_page(__name__, path="/edit")

companies = stored_companies().to_dict("records")

form = html.Form(
  className="flex gap-2 p-1",
  children=[
    dcc.Dropdown(
      id="dropdown:edit:company",
      className="w-100",
      options=companies,
      value="",
      placeholder="Company",
    ),
    dcc.Dropdown(
      id="dropdown:edit:statement",
      className="w-50",
      options=[],
      value="",
      placeholder="Statement",
    ),
  ],
)

layout = html.Main(
  className="size-full flex flex-col",
  children=[
    form,
    html.Div(
      className="size-full grid grid-cols-2 overflow-hidden",
      children=[
        dcc.RadioItems(
          className="max-h-full overflow-y-scroll",
          id="radio:edit:items",
          options=[],
        ),
        html.Div(
          id="div:edit:data",
          className="h-full flex flex-col gap-2 overflow-y-scroll",
        ),
      ],
    ),
    dcc.Store(id="store:edit:data"),
  ],
)


@callback(
  Output("dropdown:edit:statement", "options"),
  Input("dropdown:edit:company", "value"),
)
def update_dropdown(value: str):
  if not value:
    return no_update

  query = f"""
    SELECT
      date || "(" || fiscal_period || ")" AS label,
      date || "_" || fiscal_period AS value
    FROM '{value}'
  """
  df = read_sqlite("statements.db", query)

  return df.to_dict("records")


@callback(
  Output("radio:edit:items", "options"),
  Input("dropdown:edit:statement", "value"),
  State("dropdown:edit:company", "value"),
)
def update_items(date_period: str, company: str):
  if not (date_period and company):
    return no_update

  date, period = date_period.split("_")

  query = f"""
    SELECT key FROM '{company}', json_each(data)
    WHERE date = :date AND fiscal_period = :period
  """
  df = read_sqlite("statements.db", query, {"date": date, "period": period})

  return df["key"].tolist()


@callback(
  Output("store:edit:data", "data"),
  Input("radio:edit:items", "value"),
  State("dropdown:edit:company", "value"),
  State("dropdown:edit:statement", "value"),
)
def update_data(item: str, company: str, date_period: str):
  if not (item and company and date_period):
    return no_update
  date, period = date_period.split("_")

  query = f"""
    SELECT json_extract(data, '$.{item}') AS data FROM '{company}' 
    WHERE date = :date AND fiscal_period = :period
  """
  df = read_sqlite("statements.db", query, {"date": date, "period": period})
  data = json.loads(df["data"].iloc[0])

  return data


def period_input(period: dict, index: int):
  if "instant" in period:
    return [
      InputAIO(
        id=f"edit:period:instant:{index}",
        input_props={
          "type": "text",
          "value": period["instant"],
          "placeholder": "Instant",
        },
      )
    ]

  return [
    InputAIO(
      id=f"edit:period:startdate:{index}",
      input_props={
        "type": "text",
        "value": period["start_date"],
        "placeholder": "Start date",
      },
    ),
    InputAIO(
      id=f"edit:period:enddate:{index}",
      input_props={
        "type": "text",
        "value": period["end_date"],
        "placeholder": "End date",
      },
    ),
  ]


@callback(Output("div:edit:data", "children"), Input("store:edit:data", "data"))
def update_edit(data: list[dict]):
  if not data:
    return no_update

  forms = []
  for i, record in enumerate(data):
    period = period_input(record["period"], i)

    value = InputAIO(
      id=f"edit:value:{i}",
      input_props={
        "type": "number",
        "value": record["value"],
        "placeholder": "Value",
      },
    )

    unit = InputAIO(
      id=f"edit:unit:{i}",
      input_props={
        "type": "text",
        "value": record["unit"],
        "placeholder": "Unit",
      },
    )

    forms.append(
      html.Form(
        id={"type": "form:edit:data", "index": i},
        className="flex flex-col gap-2 p-2 border border-text/10 rounded-sm",
        children=[*period, value, unit],
      )
    )

  return forms
