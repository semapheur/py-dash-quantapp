import base64
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
from pydantic import ValidationError

from components.input import InputAIO

from lib.db.lite import read_sqlite
from lib.fin.models import FinStatement
from lib.fin.statement import df_to_statements, upsert_statements
from lib.styles import BUTTON_STYLE, UPLOAD_STYLE
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
    html.Button(
      "Export JSON", id="button:edit:export", className=BUTTON_STYLE, type="button"
    ),
    dcc.Upload(
      id="upload:edit:json",
      className=UPLOAD_STYLE,
      accept=".json",
      children=[html.Div(["Upload JSON"])],
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
          className="h-full flex flex-col gap-2 p-2 overflow-y-scroll",
        ),
      ],
    ),
    dcc.Store(id="store:edit:data"),
    dcc.Download(id="download:edit:json"),
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
  start_date = period.get("start_date", "")
  end_date = period.get("end_date", period.get("instant", ""))

  return [
    InputAIO(
      id=f"edit:period:startdate:{index}",
      input_props={
        "type": "text",
        "value": start_date,
        "placeholder": "Start date",
      },
    ),
    InputAIO(
      id=f"edit:period:enddate:{index}",
      input_props={
        "type": "text",
        "value": end_date,
        "placeholder": "End date",
      },
    ),
  ]


def member_form(member: dict[str, dict], index: int):
  children = []

  for i, m in enumerate(member):
    member_input = InputAIO(
      id=f"edit:member:{i}",
      input_props={
        "type": "text",
        "value": m,
        "placeholder": "Member",
      },
    )

    value_input = InputAIO(
      id=f"edit:member:value:{i}",
      input_props={
        "type": "number",
        "value": member[m]["value"],
        "placeholder": "Value",
      },
    )

    dim_input = InputAIO(
      id=f"edit:member:unit:{i}",
      input_props={
        "type": "text",
        "value": member[m]["value"],
        "placeholder": "Dim",
      },
    )

    children.append(
      html.Form(
        id={"type": "form:edit:member", "index": i},
        className="flex flex-col gap-2 p-2 border border-text/10 rounded-sm",
        children=[member_input, value_input, dim_input],
      )
    )

  return html.Form(
    id={"type": "form:edit:members", "index": index},
    children=children,
  )


@callback(
  Output("div:edit:data", "children"),
  Input("store:edit:data", "data"),
  State("radio:edit:items", "value"),
)
def update_edit(
  data: list[dict],
  item: str,
):
  if not data:
    return no_update

  forms = [
    InputAIO(
      id="edit:item", input_props={"type": "text", "value": item, "placeholder": "Item"}
    )
  ]
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

    children = [*period, value, unit]
    if "members" in record:
      children.append(member_form(record["members"], i))

    forms.append(
      html.Form(
        id={"type": "form:edit:data", "index": i},
        className="flex flex-col gap-2 p-2 border border-text/10 rounded-sm",
        children=[*period, value, unit],
      )
    )

  return forms


@callback(
  Output("download:edit:json", "data"),
  Input("button:edit:export", "n_clicks"),
  State("dropdown:edit:company", "value"),
  State("dropdown:edit:statement", "value"),
  prevent_initial_call=True,
)
def export_json(
  n: int,
  company: str,
  date_period: str,
):
  if not (n and company and date_period):
    return no_update

  date, period = date_period.split("_")

  query = f"""
    SELECT * FROM '{company}' 
    WHERE date = :date AND fiscal_period = :period
  """
  df = read_sqlite("statements.db", query, {"date": date, "period": period})
  statement = df_to_statements(df)[0]

  return {
    "content": statement.model_dump_json(exclude_unset=True, indent=2),
    "filename": f"{company}_{date}_{period}.json",
  }


@callback(
  Output("notification:edit", "message"),
  Output("notification:edit", "displayed"),
  Input("upload:edit:json", "contents"),
  State("upload:edit:json", "filename"),
  State("dropdown:edit:company", "value"),
  background=True,
)
def upload_json(contents: str, filename: str, company: str):
  if not contents:
    return no_update

  if not filename.endswith(".json"):
    message = f"Only JSON files supported. Invalid file type: {filename}"

    return (
      message,
      True,
    )

  _, content_string = contents.split(",")
  decoded = base64.b64decode(content_string)

  data = json.loads(decoded.decode("utf-8"))
  required_keys = {"url", "date", "scope", "fiscal_period", "fiscal_end", "currency"}
  if set(data.keys()) != required_keys:
    message = (
      f"Invalid JSON structure. Expected keys: {required_keys}, got: {set(data.keys())}"
    )
    return message, True

  try:
    record = FinStatement(
      url=data["url"],
      date=data["date"],
      scope=data["scope"],
      fiscal_period=data["fiscal_period"],
      fiscal_end=data["fiscal_end"],
      currency=set(data["currency"]),
      data=data["data"],
    )
  except ValidationError as e:
    message = f"Invalid JSON structure: {e.errors()}"
    return message, True

  upsert_statements("statements.db", id, [record])

  return None, False
