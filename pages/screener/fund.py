import asyncio
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

from dash import (
  callback,
  ctx,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
)
import dash_ag_grid as dag
import pandas as pd
from pandera.typing import DataFrame

from lib.db.lite import fetch_sqlite, insert_sqlite, read_sqlite
from lib.morningstar.fetch import fund_data

register_page(__name__, path_template="/screener/stock")


async def load_fund_data(
  where: str, params: dict[str, str], delta=1
) -> DataFrame | None:
  async def update_data():
    data = await fund_data()
    status = pd.DataFrame([{"last_updated": dt.now().date}])
    insert_sqlite(status, "fund.db", "status", "replace")
    insert_sqlite(data, "fund.db", "data", "replace")

  query = "SELECT last_updated FROM status"
  last_update_fetch = fetch_sqlite("fund.db", query)

  if last_update_fetch is None:
    await update_data()

  else:
    last_update = dt.strptime(last_update_fetch[0][0], "%Y-%m-%d")
    if relativedelta(dt.now(), last_update).days <= delta:
      await update_data()
  query = f"""
    SELECT
      f.legal_name,
      c.category,
      pb.primary_benchmark,
      d.*,
    FROM data d
    JOIN fund f ON d.security_id = f.security_id
    JOIN category c ON f.category_id = c.category_id
    JOIN pb ON f.primary_benchmark_id = pb.primary_benchmark_id
    {where}
  """
  stored_data = read_sqlite("fund.db", query, params)

  return stored_data


provider_query = (
  "SELECT branding_company AS label, branding_company_id AS value FROM branding_company"
)
provider = read_sqlite("fund.db", provider_query)

category_query = "SELECT category AS label, category_id AS value FROM category"
category = read_sqlite("fund.db", category_query)

layout = html.Main(
  className="h-full grid grid-cols-[1fr_4fr]",
  children=[
    html.Div(
      className="h-full flex flex-col",
      children=[
        dcc.Dropdown(
          id="dropdown:screener-fund:provider",
          options=[] if provider is None else provider.to_dict("records"),
          value="",
        ),
        dcc.Dropdown(
          id="dropdown:screener-fund:category",
          options=[] if category is None else category.to_dict("records"),
          value="",
        ),
      ],
    ),
    dag.AgGrid(
      id="table:screener-fund",
      getRowId="params.data.company",
      columnSize="autoSize",
      dashGridOptions={"tooltipInteraction": True},
      style={"height": "100%"},
    ),
    dcc.Store(id="store:screener-fund:query", data={}),
  ],
)


@callback(
  Output("store:screener-fund:query", "data"),
  Input("dropdown:screener-fund:provider", "value"),
  Input("dropdown:screener-fund:category", "value"),
  State("store:screener-fund:query", "data"),
)
def update_query(provider: str, category: str, query: dict):
  if not (provider and category):
    return no_update

  dropdown_id = ctx.triggered_id
  if dropdown_id == "dropdown:screener-fund:provider":
    query["provider"] = {"column": "f.branding_company_id", "value": provider}

  if dropdown_id == "dropdown:screener-fund:category":
    query["category"] = {"column": "f.cateogory_id", "value": category}

  return query


@callback(
  Output("table:screener-fund", "columnDefs"),
  Output("table:screener-fund", "rowData"),
  Input("store:screener-fund:query", "data"),
  background=True,
)
def update_table(query_data: dict[str, dict[str, str]]):
  if not query_data:
    return no_update

  where_items = []
  params = {}
  for k, v in query_data.items():
    where_items.append(f"{v["column"]} = :{k}")
    params[k] = v["value"]

  where = f"WHERE {" AND ".join(where_items)}"

  data = asyncio.run(load_fund_data(where, params, 1))

  if data is None:
    return no_update

  column_defs = [{"field": c} for c in data.columns]

  return column_defs, data.to_dict("records")
