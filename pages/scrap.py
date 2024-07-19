import base64
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import io
import re
from pathlib import Path
import sqlite3
from typing import Literal

from dash import (
  ALL,
  callback,
  dcc,
  html,
  no_update,
  register_page,
  Input,
  Output,
  State,
  Patch,
)
import dash_ag_grid as dag
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
import plotly.graph_objects as go
import httpx
from img2table.document import PDF
from img2table.ocr import TesseractOCR
import numpy as np
import pandas as pd
import pdfplumber

from components.ticker_select import TickerSelectAIO
from components.input import InputAIO
from components.modal import OpenCloseModalAIO

from lib.const import HEADERS
from lib.db.lite import fetch_sqlite
from lib.fin.models import (
  FinStatement,
  Item,
  Instant,
  Interval,
  Scope,
  FiscalPeriod,
)
from lib.fin.statement import upsert_statements
from lib.morningstar.ticker import Stock
from lib.utils import download_file, split_multiline

register_page(__name__, path="/scrap")

main_style = (
  "relative h-full bg-primary"  # grid grid-cols-[minmax(min-content,20vw)_1fr_1fr]
)
input_style = "p-1 rounded-l border-l border-t border-b border-text/10"
button_style = "px-2 rounded bg-secondary/50 text-text"
group_button_style = "px-2 rounded-r bg-secondary/50 text-text"
radio_style = (
  "relative flex gap-4 px-2 py-1 border border-text/50 rounded "
  "before:absolute before:left-1 before:top-0 before:-translate-y-1/2 "
  "before:content-['Extract_method'] before:px-1 before:bg-primary before:text-xs"
)

text_options_style = (
  "relative grid grid-cols-2 gap-x-1 border border-text/50 rounded px-1 pt-4 pb-1 "
  "before:absolute before:left-1 before:top-0 before:-translate-y-1/2 "
  "before:bg-primary before:px-1 before:text-text/50 before:text-xs"
)

resize_handle_style = "h-full w-0.5 bg-text/50 hover:bg-secondary hover:w-1"

scrap_options_sidebar = html.Aside(
  className="relative flex flex-col grow gap-2 p-2",
  children=[
    TickerSelectAIO(id="scrap"),
    dcc.Dropdown(id="dropdown:scrap:document", placeholder="Document"),
    html.Form(
      className="flex",
      action="",
      children=[
        dcc.Input(
          id="input:scrap:pages",
          className=input_style,
          placeholder="Pages",
          type="text",
        ),
        html.Button(
          "Extract",
          id="button:scrap:extract",
          className=group_button_style,
          type="button",
          n_clicks=0,
        ),
      ],
    ),
    html.Form(
      className="flex gap-1",
      children=[
        dcc.RadioItems(
          id="radioitems:scrap:extract-method",
          className=radio_style,
          labelClassName="gap-1 text-text",
          labelStyle={"display": "flex"},
          options=[
            {"label": "Text", "value": "text"},
            {"label": "Image", "value": "image"},
          ],
          value="text",
        ),
        html.Button(
          "Options",
          id="button:scrap:text-options",
          className=button_style,
          type="button",
          n_clicks=0,
        ),
      ],
    ),
    html.Button(
      "Extract words",
      id=OpenCloseModalAIO.open_id("scrap:words"),
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Annotate page",
      id="button:scrap:download-image",
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Delete rows",
      id="button:scrap:delete",
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Rename headers",
      id=OpenCloseModalAIO.open_id("scrap:headers"),
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Export to JSON",
      id="button:scrap:export",
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Form(
      className="grid grid-cols-2 gap-x-1 gap-y-2",
      children=[
        InputAIO(
          "scrap:id",
          "100%",
          {"className": "col-span-2"},
          {"type": "text", "placeholder": "Company ID"},
        ),
        InputAIO(
          "scrap:date", "100%", input_props={"type": "text", "placeholder": "Date"}
        ),
        InputAIO(
          "scrap:fiscal-end",
          "100%",
          input_props={
            "type": "text",
            "placeholder": "Fiscal end",
            "value": "12-31",
          },
        ),
        dcc.Dropdown(
          id="dropdown:scrap:scope",
          className="outline-none",
          placeholder="Scope",
          options=[
            {"label": "Annual", "value": "annual"},
            {"label": "Quarterly", "value": "quarterly"},
          ],
        ),
        dcc.Dropdown(
          id="dropdown:scrap:period",
          placeholder="Period",
          options=["FY", "Q1", "Q2", "Q3", "Q4"],
        ),
        InputAIO(
          "scrap:factor",
          "100%",
          input_props={"value": 1e6, "placeholder": "Factor", "type": "number"},
        ),
        InputAIO(
          "scrap:currency",
          "100%",
          input_props={"value": "NOK", "placeholder": "Currency", "type": "text"},
        ),
      ],
    ),
    dcc.Upload(
      id="upload:scrap:image",
      className="py-1 border border-dashed border-text/50 rounded hover:border-secondary text-center cursor-pointer",
      accept="image/*",
      children=[html.Div(["Upload image"])],
    ),
  ],
)

table_options_form = html.Div(
  id="div:scrap:text-options",
  className="absolute top-0 left-full w-1/5 h-full flex flex-col gap-2 py-2 bg-primary z-[999]",
  children=[
    html.H3("Table extraction options", className="text-text mb-2"),
    html.Form(
      className=text_options_style + " before:content-['Strategy']",
      children=[
        dcc.Dropdown(
          id="dropdown:scrap:text-options:vertical-strategy",
          placeholder="Vertical",
          options=["lines", "lines_strict", "text"],
          value="lines",
        ),
        dcc.Dropdown(
          id="dropdown:scrap:text-options:horizontal-strategy",
          placeholder="Horizontal",
          options=["lines", "lines_strict", "text"],
          value="lines",
        ),
      ],
    ),
    html.Form(
      className=text_options_style + " before:content-['Minimum_words']",
      children=[
        InputAIO(
          "scrap:text-options:min-words-vertical",
          "100%",
          input_props={
            "placeholder": "Vertical",
            "type": "number",
            "min": 0,
            "step": 1,
            "value": 3,
          },
        ),
        InputAIO(
          "scrap:text-options:min-words-horizontal",
          "100%",
          input_props={
            "placeholder": "Horizontal",
            "type": "number",
            "min": 0,
            "step": 1,
            "value": 1,
          },
        ),
      ],
    ),
    html.Form(
      className=text_options_style + " before:content-['Snap_tolerance']",
      children=[
        InputAIO(
          "scrap:text-options:snap-x-tolerance",
          "100%",
          input_props={"placeholder": "x", "type": "number", "min": 0, "value": 3},
        ),
        InputAIO(
          "scrap:text-options:snap-y-tolerance",
          "100%",
          input_props={"placeholder": "y", "type": "number", "min": 0, "value": 3},
        ),
      ],
    ),
    html.Form(
      className=text_options_style + " before:content-['Join_tolerance']",
      children=[
        InputAIO(
          "scrap:text-options:join-x-tolerance",
          "100%",
          input_props={"placeholder": "x", "type": "number", "min": 0, "value": 3},
        ),
        InputAIO(
          "scrap:text-options:join-y-tolerance",
          "100%",
          input_props={"placeholder": "y", "type": "number", "min": 0, "value": 3},
        ),
      ],
    ),
    html.Form(
      className=text_options_style + " before:content-['Intersection_tolerance']",
      children=[
        InputAIO(
          "scrap:text-options:intersection-x-tolerance",
          "100%",
          input_props={"placeholder": "x", "type": "number", "min": 0, "value": 3},
        ),
        InputAIO(
          "scrap:text-options:intersection-y-tolerance",
          "100%",
          input_props={"placeholder": "y", "type": "number", "min": 0, "value": 3},
        ),
      ],
    ),
    html.Form(
      className=text_options_style + " before:content-['Text_tolerance']",
      children=[
        InputAIO(
          "scrap:text-options:text-x-tolerance",
          "100%",
          input_props={"placeholder": "x", "type": "number", "min": 0, "value": 3},
        ),
        InputAIO(
          "scrap:text-options:text-y-tolerance",
          "100%",
          input_props={"placeholder": "y", "type": "number", "min": 0, "value": 3},
        ),
      ],
    ),
  ],
)

layout = html.Main(
  className=main_style,
  children=[
    PanelGroup(
      id="panelgroup:scrap",
      direction="horizontal",
      className="size-full",
      children=[
        Panel(
          id="panel:scrap:controls",
          defaultSizePercentage=20,
          minSizePixels=5,
          children=[scrap_options_sidebar],
        ),
        PanelResizeHandle(html.Div(className=resize_handle_style)),
        Panel(
          id="panel:scrap:pdf",
          defaultSizePercentage=40,
          minSizePixels=5,
          children=[
            dcc.Loading(
              parent_className="size-full",
              children=[
                html.ObjectEl(
                  id="object:scrap:pdf",
                  type="application/pdf",
                  width="100%",
                  height="100%",
                )
              ],
              target_components={"object:scrap:pdf": "data"},
            ),
          ],
        ),
        PanelResizeHandle(html.Div(className=resize_handle_style)),
        Panel(
          id="panel:scrap:table",
          defaultSizePercentage=40,
          minSizePixels=5,
          children=[
            dag.AgGrid(
              id="table:scrap",
              getRowId="params.data.company",
              columnSize="autoSize",
              defaultColDef={"editable": True},
              dashGridOptions={
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 10,
              },
              style={"height": "100%"},
            )
          ],
        ),
      ],
    ),
    table_options_form,
    OpenCloseModalAIO(
      "scrap:headers",
      "Rename headers",
      children=[
        html.Div(
          className="flex flex-col",
          children=[
            html.Form(id="form:scrap:headers", className="flex flex-col gap-1"),
            html.Button(
              "Update", id="button:scrap:headers:update", className=button_style
            ),
          ],
        )
      ],
    ),
    OpenCloseModalAIO(
      "scrap:words",
      "PDF words",
      children=[
        html.Div(
          className="size-full grid grid-cols-2",
          children=[
            dcc.Loading(
              parent_className="size-full",
              children=[
                dcc.Graph(
                  id="graph:scrap:words",
                  className="size-full",
                  config={"scrollZoom": True},
                ),
              ],
              target_components={"graph:scrap:words": "figure"},
            ),
            dag.AgGrid(
              id="table:scrap:words",
              getRowId="params.data.company",
              columnSize="autoSize",
              defaultColDef={"editable": True},
              dashGridOptions={
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 10,
              },
              style={"height": "100%"},
            ),
          ],
        )
      ],
      dialog_props={
        "style": {
          "height": "100%",
          "width": "100%",
        },
      },
    ),
    dcc.Download(id="download:scrap:image"),
    dcc.ConfirmDialog(
      id="notification:scrap:table-error",
      message="Unable to parse table",
    ),
    dcc.Store(id="store:scrap:pixel-scale", data={}),
  ],
)


def doc_url(doc_id: str) -> str:
  return (
    f"https://doc.morningstar.com/document/{doc_id}.msdoc/"
    "?clientid=euretailsite&key=9ab7c1c01e51bcec"
  )


@callback(
  Output("dropdown:scrap:document", "options"),
  Input(TickerSelectAIO.aio_id("scrap"), "value"),
  background=True,
)
def update_dropdown(ticker: str):
  if not ticker:
    return no_update

  docs = Stock(*ticker.split("|")).documents()
  docs.rename(columns={"doc_id": "value"}, inplace=True)
  docs["label"] = (
    docs["date"] + " - " + docs["doc_type"] + " (" + docs["language"] + ")"
  )

  return docs[["label", "value"]].to_dict("records")


@callback(
  Output("object:scrap:pdf", "data"),
  Input("dropdown:scrap:document", "value"),
  prevent_initial_call=True,
  background=True,
)
def update_object(doc_id: str):
  if not doc_id:
    return no_update

  pdf_path = Path(f"assets/docs/{doc_id}.pdf")
  if not pdf_path.exists():
    url = doc_url(doc_id)
    download_file(url, pdf_path)

  return str(pdf_path)


@callback(
  Output("div:scrap:text-options", "className"),
  Input("button:scrap:text-options", "n_clicks"),
  State("div:scrap:text-options", "className"),
  prevent_initial_call=True,
)
def open_table_settings(n_clicks: int, className: str):
  if not n_clicks:
    return no_update

  if n_clicks % 2 == 0:
    return className + " -translate-x-full"

  return className.replace("-translate-x-full", "").strip()


@callback(
  Output("graph:scrap:words", "figure", allow_duplicate=True),
  Output("table:scrap:words", "columnDefs"),
  Output("table:scrap:words", "rowData"),
  Output("store:scrap:pixel-scale", "data"),
  Input(OpenCloseModalAIO.open_id("scrap:words"), "n_clicks"),
  State("dropdown:scrap:document", "value"),
  State("input:scrap:pages", "value"),
  prevent_initial_call=True,
  background=True,
)
def extract_words(n_clicks: int, doc_id: str, pages_text: str):
  if not (n_clicks and doc_id and pages_text):
    return no_update

  def create_hover_template(pdf_width, pdf_height, img_width, img_height):
    # Create arrays for x and y coordinates
    x = np.linspace(0, pdf_width, img_width)
    y = np.linspace(pdf_height, 0, img_height)  # Reverse y-axis for PDF coordinates

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Create hover template
    hovertemplate = (
      "<b>x:</b> %{customdata[0]:.2f}<br><b>y:</b> %{customdata[1]:.2f}<br>"
    )

    return X, Y, hovertemplate

  pdf_path = Path(f"assets/docs/{doc_id}.pdf")
  pages = [int(p) - 1 for p in pages_text.split(",")]
  with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[pages[0]]
    words = page.extract_words()
    img = pdf.pages[104].to_image(resolution=300, antialias=True)

  df = pd.DataFrame.from_records(words)

  prefix = "data:image/png;base64,"
  with io.BytesIO() as stream:
    img.original.save(stream, format="png")
    base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")

  X, Y, hovertemplate = create_hover_template(
    page.width, page.height, img.original.width, img.original.height
  )
  fig = go.Figure(
    go.Image(
      source=base64_string, customdata=np.dstack((X, Y)), hovertemplate=hovertemplate
    )
  )
  pixel_scale = img.original.width / page.width

  for x0, x1, top, bottom in zip(df["x0"], df["x1"], df["top"], df["bottom"]):
    fig.add_shape(
      type="rect",
      x0=x0 * pixel_scale,
      x1=x1 * pixel_scale,
      y0=top * pixel_scale,
      y1=bottom * pixel_scale,
      xref="x",
      yref="y",
    )

  fig.update_layout(margin=dict(b=10, t=10))
  fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

  columnDefs = [
    {
      "field": c,
      "cellDataType": "text" if df.dtypes.loc[c] in ("object", "bool") else "number",
      "filter": True,
    }
    for c in df.columns
  ]

  return fig, columnDefs, df.to_dict("records"), {"pixel_scale": pixel_scale}


@callback(
  Output("graph:scrap:words", "figure"),
  Input("table:scrap:words", "virtualRowData"),
  State("store:scrap:pixel-scale", "data"),
  prevent_initial_call=True,
)
def update_word_image(row_data: list[dict], pixel_data: dict[str, str]):
  if not (row_data and pixel_data):
    return no_update

  fig: go.Figure = Patch()
  pixel_scale = pixel_data["pixel_scale"]

  df = pd.DataFrame.from_records(row_data)

  shapes = []
  for x0, x1, top, bottom in zip(df["x0"], df["x1"], df["top"], df["bottom"]):
    shapes.append(
      go.layout.Shape(
        type="rect",
        x0=x0 * pixel_scale,
        x1=x1 * pixel_scale,
        y0=top * pixel_scale,
        y1=bottom * pixel_scale,
        xref="x",
        yref="y",
      )
    )

  fig.layout.shapes = shapes

  return fig


@callback(
  Output("download:scrap:image", "data"),
  Input("button:scrap:download-image", "n_clicks"),
  State("dropdown:scrap:document", "value"),
  State("input:scrap:pages", "value"),
  prevent_initial_call=True,
  background=True,
)
def annotate_image(
  n_clicks: int,
  doc_id: str,
  pages_text: str,
):
  if not (n_clicks and doc_id and pages_text):
    return no_update

  pdf_path = Path(f"assets/docs/{doc_id}.pdf")

  pages = [int(p) - 1 for p in pages_text.split(",")]
  with pdfplumber.open(pdf_path) as pdf:
    img = pdf.pages[pages[0]].to_image(resolution=600)

  with io.BytesIO() as stream:
    img.original.save(stream, format="png")
    return dcc.send_bytes(stream.getvalue(), "page.png")


@callback(
  Output("table:scrap", "columnDefs"),
  Output("table:scrap", "rowData"),
  Output("notification:scrap:table-error", "displayed"),
  Input("button:scrap:extract", "n_clicks"),
  State("dropdown:scrap:document", "value"),
  State("input:scrap:pages", "value"),
  State("radioitems:scrap:extract-method", "value"),
  State(InputAIO.aio_id("scrap:factor"), "value"),
  State(InputAIO.aio_id("scrap:currency"), "value"),
  State("dropdown:scrap:text-options:vertical-strategy", "value"),
  State("dropdown:scrap:text-options:horizontal-strategy", "value"),
  State(InputAIO.aio_id("scrap:text-options:min-words-vertical"), "value"),
  State(InputAIO.aio_id("scrap:text-options:min-words-horizontal"), "value"),
  State(InputAIO.aio_id("scrap:text-options:snap-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:snap-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:join-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:join-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:intersection-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:intersection-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:text-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:text-options:text-y-tolerance"), "value"),
  prevent_initial_call=True,
  background=True,
)
def update_table(
  n_clicks: int,
  doc_id: str,
  pages_text: str,
  method: Literal["text", "image"],
  factor: int,
  currency: str,
  vertical_strategy: Literal["lines", "lines_strict", "text"],
  horizontal_strategy: Literal["lines", "lines_strict", "text"],
  min_words_vertical: float,
  min_words_horizontal: float,
  snap_x_tolerance: float,
  snap_y_tolerance: float,
  join_x_tolerance: float,
  join_y_tolerance: float,
  intersection_x_tolerance: float,
  intersection_y_tolerance: float,
  text_x_tolerance: float,
  text_y_tolerance: float,
):
  if not (n_clicks and doc_id and pages_text):
    return no_update

  pdf_path = Path(f"assets/docs/{doc_id}.pdf")
  pdf_src = io.BytesIO()
  if pdf_path.exists():
    with open(pdf_path, "rb") as pdf_file:
      pdf_src.write(pdf_file.read())
  else:
    url = doc_url(doc_id)
    response = httpx.get(url=url, headers=HEADERS)
    pdf_src.write(response.content)

  pages = [int(p) - 1 for p in pages_text.split(",")]

  if method == "text":
    settings = {
      "vertical_strategy": vertical_strategy or "lines",
      "horizontal_strategy": horizontal_strategy or "lines",
      "snap_x_tolerance": snap_x_tolerance or 3,
      "snap_y_tolerance": snap_y_tolerance or 3,
      "join_x_tolerance": join_x_tolerance or 3,
      "join_y_tolerance": join_y_tolerance or 3,
      "min_words_vertical": min_words_vertical or 3,
      "min_words_horizontal": min_words_horizontal or 1,
      "intersection_x_tolerance": intersection_x_tolerance or 3,
      "intersection_y_tolerance": intersection_y_tolerance or 3,
      "text_x_tolerance": text_x_tolerance or 3,
      "text_y_tolerance": text_y_tolerance or 3,
    }

    with pdfplumber.open(pdf_path) as pdf_file:
      table = pdf_file.pages[104].extract_table(table_settings=settings)

    df = pd.DataFrame(table)
    result = df.apply(split_multiline, axis=1)
    df = pd.concat(result.tolist(), ignore_index=True)
    df.dropna(how="all", inplace=True)

  elif method == "image":
    pdf = PDF(src=pdf_src, pages=pages)

    ocr = TesseractOCR(lang="eng")

    tables = pdf.extract_tables(
      ocr=ocr,
      # borderless_tables=True if "borderless" in options else False,
      # implicit_rows=True if "implicit" in options else False,
    )

    # TODO: merge tables
    if len(tables[pages[0]]) == 0:
      return None, None, True

    df = tables[pages[0]][0].df

  df["period"] = "instant"
  df["factor"] = factor
  df["unit"] = currency
  diff = ["period", "factor", "unit"]
  df = df[diff + list(df.columns.difference(diff))]

  columnDefs: list[dict[str, str | bool | dict]] = [
    {"field": str(c)} for c in df.columns
  ]
  columnDefs[0].update(
    {
      "checkboxSelection": True,
      "cellEditor": "agSelectCellEditor",
      "cellEditorParams": {"values": ["instant", "duration"]},
    }
  )
  columnDefs[1].update({"type": "numericColumn"})

  return columnDefs, df.to_dict("records"), False


@callback(
  Output("table:scrap", "deleteSelectedRows"),
  Input("button:scrap:delete", "n_clicks"),
  prevent_initial_call=True,
)
def selected(_: int):
  return True


@callback(
  Output("form:scrap:headers", "children"),
  Input("table:scrap", "columnDefs"),
  prevent_initial_call=True,
)
def update_form(cols: list[dict]):
  if not cols:
    return no_update

  return [
    dcc.Input(
      id={"type": "input:scrap:headers", "index": i},
      className="px-1 rounded border border-text/10 hover:border-text/50 focus:border-secondary",
      placeholder=f"Field {i}",
      value=col["field"],
      type="text",
    )
    for (i, col) in enumerate(cols[3:])
  ]


@callback(
  Output("table:scrap", "columnDefs", allow_duplicate=True),
  Output("table:scrap", "rowData", allow_duplicate=True),
  Input("button:scrap:headers:update", "n_clicks"),
  State({"type": "input:scrap:headers", "index": ALL}, "value"),
  State("table:scrap", "columnDefs"),
  State("table:scrap", "rowData"),
  prevent_initial_call=True,
)
def toggle_cols(n: int, new_names: list[str], cols: list[dict], rows: list[dict]):
  if not n:
    return no_update

  df = pd.DataFrame.from_records(rows)
  df = df[[col["field"] for col in cols]]

  col_map = {col: name for (col, name) in zip(df.columns[3:], new_names)}
  df.rename(columns=col_map, inplace=True)

  for i, name in enumerate(new_names):
    cols[i + 3]["field"] = name

  return cols, df.to_dict("records")


@callback(
  Output(InputAIO.aio_id("scrap:id"), "value"),
  Input(TickerSelectAIO.aio_id("scrap"), "value"),
)
def update_input(ticker: str):
  if not ticker:
    return no_update

  security_id = ticker.split("|")[0]
  query = "SELECT company_id FROM stock WHERE security_id = :security_id"
  company_id = fetch_sqlite("ticker.db", query, {"security_id": security_id})[0][0]

  if company_id is None:
    return no_update

  return company_id


@callback(
  Output(InputAIO.aio_id("scrap:date"), "value"),
  Input("dropdown:scrap:document", "value"),
  State("dropdown:scrap:document", "options"),
)
def update_document_dropdown(doc: str, options: list[dict[str, str]]):
  if not doc:
    return no_update

  label = [x["label"] for x in options if x["value"] == doc][0]

  pattern = r"\d{4}-\d{2}-\d{2}"
  match = re.search(pattern, label)

  if not match:
    return ""

  return match.group()


@callback(
  Output("dropdown:scrap:scope", "value"),
  Output("dropdown:scrap:period", "value"),
  Input("dropdown:scrap:document", "value"),
  State("dropdown:scrap:document", "options"),
)
def update_scope_dropdown(doc: str, options: list[dict[str, str]]):
  if not doc:
    return no_update

  label = [x["label"] for x in options if x["value"] == doc][0]

  pattern = r"(annual|quarterly)"
  match = re.search(pattern, label, flags=re.I)

  if not match:
    return "", ""

  scope = match.group().lower()
  period = "FY" if scope == "annual" else ""

  return (scope, period)


@callback(
  Output("button:scrap:export", "disabled"),
  Input(InputAIO.aio_id("scrap:fiscal-end"), "value"),
)
def validate_fiscal_end(fiscal_end: str) -> bool:
  pattern = re.compile(
    r"^(0[1-9]|1[0-2])-(0[1-9]|1\d|2[0-8])|(0[13-9]|"
    r"1[0-2])-29|0[13-9]|1[0-2]-(29|30)|(0[13578]|1[02])-31$"
  )

  if pattern.match(fiscal_end):
    return False
  else:
    return True


@callback(
  Output("button:scrap:export", "id"),
  Input("button:scrap:export", "n_clicks"),
  State("table:scrap", "rowData"),
  State("dropdown:scrap:document", "value"),
  State(InputAIO.aio_id("scrap:id"), "value"),
  State(InputAIO.aio_id("scrap:date"), "value"),
  State("dropdown:scrap:scope", "value"),
  State("dropdown:scrap:period", "value"),
  State(InputAIO.aio_id("scrap:fiscal-end"), "value"),
  prevent_initial_call=True,
)
def export(
  n: int,
  rows: list[dict],
  url: str,
  id: str,
  date: str,
  scope: Scope,
  period: FiscalPeriod,
  fiscal_end: str,
):
  def parse_period(scope: Scope, date_text: str, row: pd.Series) -> Instant | Interval:
    date = dt.strptime(date_text, "%Y-%m-%d").date()

    if row["period"] == "instant":
      return Instant(instant=date)

    start_date = dt.strptime(period, "%Y-%m-%d").date()

    if scope == "annual":
      start_date -= relativedelta(years=1)
      months = 12

    elif scope == "quarterly":
      start_date -= relativedelta(months=3)
      months = 3

    return Interval(start_date=start_date, end_date=date, months=months)

  if not n:
    return no_update

  data: dict[str, list[Item]] = {}

  df = pd.DataFrame.from_records(rows)
  if "item" not in set(df.columns):
    return "button:scrap:export"

  dates = list(df.columns.difference(["period", "factor", "unit", "item"]))
  df.loc[:, dates] = df[dates].replace(r"[^\d.]", "", regex=True).astype(float)

  currencies = set()

  for _, r in df.iterrows():
    if r["unit"] != "shares":
      currencies.add(r["unit"])

    data[r["item"]] = [
      Item(
        value=r[d] * float(r["factor"]),
        unit=r["unit"],
        period=parse_period(scope, d, r),
      )
      for d in dates
    ]

  records = [
    FinStatement(
      url=url,
      date=dt.strptime(date, "%Y-%m-%d").date(),
      scope=scope,
      period=period,
      fiscal_end=fiscal_end,
      currency=currencies,
      data=data,
    )
  ]

  upsert_statements("statements.db", id, records)

  return "button:scrap:export"
