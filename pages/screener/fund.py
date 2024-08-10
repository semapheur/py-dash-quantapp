from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

from dash import callback, dcc, html, no_update, register_page, Input, Output
import dash_ag_grid as dag
import pandas as pd
from pandera.typing import DataFrame

from lib.db.lite import fetch_sqlite, insert_sqlite, read_sqlite
from lib.morningstar.fetch import fund_data

register_page(__name__, path_template="/screener/stock")


async def load_fund_data(where: str, delta: int) -> DataFrame | None:
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
  stored_data = read_sqlite("fund.db", query)

  return stored_data


query = """SELECT branding_company AS label, branding_company_id AS value FROM branding_company"""
provider = read_sqlite("fund.db", query)

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
        )
      ],
    ),
    dag.AgGrid(
      id="table:screener-fund",
      getRowId="params.data.company",
      columnSize="autoSize",
      dashGridOptions={"tooltipInteraction": True},
      style={"height": "100%"},
    ),
  ],
)


@callback(
  Output("table:screener-fund", "colDefs"),
  Output("table:screener-fund", "rowData"),
  Input("dropdown:screener-fund:provider", "value"),
)
def update_table(provider: str):
  return no_update
