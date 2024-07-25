from dash import callback, html, dcc, no_update, Input, Output, State, MATCH


def CompanyHeader(_id: str):
  return html.Nav(
    className="flex gap-4",
    children=[
      dcc.Link(
        "Overview",
        href=f"/company/{_id}/overview",
        id={"type": "link:company", "index": "overview"},
      ),
      dcc.Link(
        "Financials",
        href=f"/company/{_id}/financials",
        id={"type": "link:company", "index": "financials"},
      ),
      dcc.Link(
        "Fundamentals",
        href=f"/company/{_id}/fundamentals",
        id={"type": "link:company", "index": "fundamentals"},
      ),
      dcc.Link(
        "Valuation",
        href=f"/company/{_id}/valuation",
        id={"type": "link:company", "index": "valuation"},
      ),
    ],
  )


@callback(
  Output({"type": "link:company", "id": MATCH}, "className"),
  Input("location:app", "pathname"),
  State({"type": "link:company", "id": MATCH}, "href"),
)
def update_link(path: str, href: str):
  if path == href:
    return "text-secondary"
  else:
    return no_update
