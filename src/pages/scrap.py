import asyncio
import base64
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import io
import re
from pathlib import Path
from typing import Literal, TypedDict

from dash import (
  ALL,
  callback,
  ctx,
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
import numpy as np
import pandas as pd
import pdfplumber

from components.company_select import CompanySelectAIO
from components.input import InputAIO
from components.input_button import InputButtonAIO
from components.modal import OpenCloseModalAIO

from lib.fin.models import (
  FinStatement,
  Item,
  Instant,
  Interval,
  Scope,
  FiscalPeriod,
)
from lib.fin.statement import upsert_statements
from lib.fin.taxonomy import search_taxonomy
from lib.scrap import download_file_memory
from lib.utils import split_multiline, pascal_case

register_page(__name__, path="/scrap")


class ItemTable(TypedDict):
  action: Literal["create", "update", "none"]
  gaap: str
  item: str
  label_long: str
  label_short: str
  type: Literal[
    "monetary",
    "fundamental",
    "percent",
    "per_day",
    "per_share",
    "personnel",
    "ratio",
    "shares",
  ]
  balance: Literal["debit", "credit"]
  aggregate: Literal["average", "recalc", "sum", "tail"]


main_style = "h-full bg-primary"
input_style = "p-1 rounded-l border-l border-t border-b border-text/10"
button_style = "px-2 rounded bg-secondary/50 text-text"
group_button_style = "px-2 rounded-r bg-secondary/50 text-text"
radio_style = (
  "relative flex gap-4 px-2 py-1 border border-text/50 rounded "
  "before:absolute before:left-1 before:top-0 before:-translate-y-1/2 "
  "before:content-['Extract_method'] before:px-1 before:bg-primary before:text-xs"
)

scrap_options_style = (
  "relative grid grid-cols-2 gap-x-1 gap-y-2 border border-text/50 rounded px-1 pt-4 pb-1 "
  "before:absolute before:left-1 before:top-0 before:-translate-y-1/2 "
  "before:bg-primary before:px-1 before:text-text/50 before:text-xs"
)

resize_handle_style = "h-full w-0.5 bg-text/50 hover:bg-secondary hover:w-1"

scrap_controls_sidebar = html.Aside(
  className="relative flex flex-col grow gap-2 p-2",
  children=[
    dcc.Upload(
      id="upload:scrap:pdf",
      className="py-1 border border-dashed border-text/50 rounded hover:border-secondary text-center cursor-pointer",
      accept="pdf",
      children=[html.Span(["Upload PDF"])],
    ),
    InputButtonAIO(
      "scrap:url",
      input_props={"placeholder": "Document URL"},
      button_props={"children": "Fetch"},
    ),
    InputButtonAIO(
      "scrap:pages",
      input_props={"placeholder": "Pages"},
      button_props={"children": "Extract"},
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
          id=OpenCloseModalAIO.open_id("scrap:options"),
          className=button_style,
          type="button",
          n_clicks=0,
        ),
      ],
    ),
    html.Form(
      className="grid grid-cols-2 gap-1",
      children=[
        html.Button(
          "Add row",
          id="button:scrap:add-rows",
          className=button_style,
          type="button",
          n_clicks=0,
        ),
        html.Button(
          "Delete rows",
          id="button:scrap:delete-rows",
          className=button_style,
          type="button",
          n_clicks=0,
        ),
        html.Button(
          "Delete columns",
          id=OpenCloseModalAIO.open_id("scrap:delete-columns"),
          className=button_style,
          type="button",
          n_clicks=0,
        ),
        html.Button(
          "Rename columns",
          id=OpenCloseModalAIO.open_id("scrap:rename-columns"),
          className=button_style,
          type="button",
          n_clicks=0,
        ),
      ],
    ),
    html.Button(
      "Record items",
      id=OpenCloseModalAIO.open_id("scrap:record-items"),
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Export to CSV",
      id="button:scrap:export-csv",
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    html.Button(
      "Export to JSON",
      id="button:scrap:export-json",
      className=button_style,
      type="button",
      n_clicks=0,
    ),
    CompanySelectAIO(
      id="scrap:company-id",
    ),
    html.Form(
      className="grid grid-cols-2 gap-x-1 gap-y-2",
      children=[
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
    dcc.Download(id="download:scrap:json"),
  ],
)

scrap_options_sidebar = html.Div(
  id="div:scrap:options",
  className="pr-2 flex flex-col gap-2 p-2 bg-primary",
  children=[
    html.Form(
      className=scrap_options_style + " before:content-['Bounding_box']",
      children=[
        InputAIO(
          "scrap:options:bounding-box:x0",
          "100%",
          input_props={
            "placeholder": "Left (x0)",
            "type": "number",
            "min": 0,
          },
        ),
        InputAIO(
          "scrap:options:bounding-box:y0",
          "100%",
          input_props={
            "placeholder": "Top (y0)",
            "type": "number",
            "min": 0,
          },
        ),
        InputAIO(
          "scrap:options:bounding-box:x1",
          "100%",
          input_props={
            "placeholder": "Right (x1)",
            "type": "number",
            "min": 0,
          },
        ),
        InputAIO(
          "scrap:options:bounding-box:y1",
          "100%",
          input_props={
            "placeholder": "Bottom (y1)",
            "type": "number",
            "min": 0,
          },
        ),
      ],
    ),
    html.Form(
      className=scrap_options_style + " before:content-['Strategy']",
      title=(
        '"lines": Use the page\'s graphical lines — including the sides of rectangle objects — as the borders of potential table-cells.\n'
        '"lines_strict": Use the page\'s graphical lines — but not the sides of rectangle objects — as the borders of potential table-cells.\n'
        '"text": For vertical_strategy: Deduce the (imaginary) lines that connect the left, right, or center of words on the page, and use those lines as the borders of potential table-cells. For horizontal_strategy, the same but using the tops of words.\n'
        '"explicit": Only use the lines explicitly defined in explicit_vertical_lines / explicit_horizontal_lines.'
      ),
      children=[
        dcc.Dropdown(
          id="dropdown:scrap:options:vertical-strategy",
          placeholder="Vertical",
          options=["lines", "lines_strict", "text"],
          value="lines",
        ),
        dcc.Dropdown(
          id="dropdown:scrap:options:horizontal-strategy",
          placeholder="Horizontal",
          options=["lines", "lines_strict", "text"],
          value="lines",
        ),
      ],
    ),
    html.Form(
      className=scrap_options_style + " before:content-['Minimum_words']",
      children=[
        InputAIO(
          "scrap:options:min-words-vertical",
          "100%",
          form_props={
            "title": 'When using "vertical_strategy": "text", at least min_words_vertical words must share the same alignment.'
          },
          input_props={
            "placeholder": "Vertical",
            "type": "number",
            "min": 0,
            "step": 1,
            "value": 3,
          },
        ),
        InputAIO(
          "scrap:options:min-words-horizontal",
          "100%",
          form_props={
            "title": 'When using "horizontal_strategy": "text", at least min_words_horizontal words must share the same alignment.'
          },
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
      className=scrap_options_style + " before:content-['Snap_tolerance']",
      title='Parallel lines within snap_tolerance points will be "snapped" to the same horizontal/vertical position',
      children=[
        InputAIO(
          "scrap:options:snap-x-tolerance",
          "100%",
          input_props={
            "placeholder": "x",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
        InputAIO(
          "scrap:options:snap-y-tolerance",
          "100%",
          input_props={
            "placeholder": "y",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
      ],
    ),
    html.Form(
      className=scrap_options_style + " before:content-['Join_tolerance']",
      title="Line segments on the same infinite line, and whose ends are within join_tolerance of one another, will be joined into a single line segment.",
      children=[
        InputAIO(
          "scrap:options:join-x-tolerance",
          "100%",
          input_props={
            "placeholder": "x",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
        InputAIO(
          "scrap:options:join-y-tolerance",
          "100%",
          input_props={
            "placeholder": "y",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
      ],
    ),
    html.Form(
      className=scrap_options_style + " before:content-['Intersection_tolerance']",
      title="When combining edges into cells, orthogonal edges must be within intersection_tolerance points to be considered intersecting",
      children=[
        InputAIO(
          "scrap:options:intersection-x-tolerance",
          "100%",
          input_props={
            "placeholder": "x",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
        InputAIO(
          "scrap:options:intersection-y-tolerance",
          "100%",
          input_props={
            "placeholder": "y",
            "type": "number",
            "min": 0,
            "value": 3,
          },
        ),
      ],
    ),
    html.Form(
      className=scrap_options_style + " before:content-['Text_tolerance']",
      title="These text_-prefixed settings also apply to the table-identification algorithm when the text strategy is used. I.e., when that algorithm searches for words, it will expect the individual letters in each word to be no more than text_x_tolerance/text_y_tolerance points apart.",
      children=[
        InputAIO(
          "scrap:options:text-x-tolerance",
          "100%",
          input_props={"placeholder": "x", "type": "number", "min": 0, "value": 3},
        ),
        InputAIO(
          "scrap:options:text-y-tolerance",
          "100%",
          input_props={"placeholder": "y", "type": "number", "min": 0, "value": 3},
        ),
      ],
    ),
    html.Button(
      "Preview",
      id="button:scrap:preview",
      className=button_style,
    ),
    html.Button(
      "Download image",
      id="button:scrap:download-image",
      className=button_style,
      type="button",
      n_clicks=0,
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
          children=[scrap_controls_sidebar],
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
              columnSize="autoSize",
              defaultColDef={"editable": True},
              dashGridOptions={
                "rowSelection": "multiple",
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 10,
              },
              style={"height": "100%"},
            )
          ],
        ),
      ],
    ),
    OpenCloseModalAIO(
      "scrap:delete-columns",
      "Delete columns",
      children=[
        html.Div(
          className="flex flex-col",
          children=[
            html.Form(id="form:scrap:columns", className="flex flex-col gap-1"),
            html.Button(
              "Delete", id="button:scrap:delete-columns", className=button_style
            ),
          ],
        )
      ],
    ),
    OpenCloseModalAIO(
      "scrap:rename-columns",
      "Rename columns",
      children=[
        html.Div(
          className="flex flex-col",
          children=[
            html.Form(id="form:scrap:rename-columns", className="flex flex-col gap-1"),
            html.Button(
              "Update", id="button:scrap:rename-columns", className=button_style
            ),
          ],
        )
      ],
    ),
    OpenCloseModalAIO(
      "scrap:record-items",
      "Record items",
      dialog_props={
        "style": {
          "height": "75%",
          "width": "75%",
        }
      },
      children=[
        html.Div(
          className="size-full grid grid-rows-[auto_1fr] gap-1",
          children=[
            html.Form(
              className="flex gap-1",
              children=[
                html.Button(
                  "Search",
                  id="button:scrap:search-items",
                  className=button_style,
                  type="button",
                  n_clicks=0,
                ),
                html.Button(
                  "Record",
                  id="button:scrap:record-items",
                  className=button_style,
                  type="button",
                  n_clicks=0,
                ),
              ],
            ),
            dag.AgGrid(
              id="table:scrap:items",
              columnDefs=[
                {
                  "field": "action",
                  "cellDataType": "text",
                  "cellEditor": "agSelectCellEditor",
                  "cellEditorParams": {"values": ["create", "update", "none"]},
                },
                {"field": "taxonomy", "cellDataType": "text"},
                {"field": "item", "cellDataType": "text", "editable": False},
                {"field": "long", "header": "Label (long)", "cellDataType": "text"},
                {"field": "short", "header": "Label (short)", "cellDataType": "text"},
                {
                  "field": "type",
                  "cellDataType": "text",
                  "cellEditor": "agSelectCellEditor",
                  "cellEditorParams": {
                    "values": [
                      "monetary",
                      "fundamental",
                      "percent",
                      "per_day",
                      "per_share",
                      "personnel",
                      "ratio",
                      "shares",
                    ]
                  },
                },
                {
                  "field": "balance",
                  "cellDataType": "text",
                  "cellEditor": "agSelectCellEditor",
                  "cellEditorParams": {"values": ["debit", "credit"]},
                },
                {
                  "field": "aggregate",
                  "cellDataType": "text",
                  "cellEditor": "agSelectCellEditor",
                  "cellEditorParams": {"values": ["average", "recalc", "sum", "tail"]},
                },
              ],
              columnSize="autoSize",
              defaultColDef={"editable": True},
              dashGridOptions={
                "rowSelection": "multiple",
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 10,
              },
              style={"height": "100%", "width": "100%", "min-height": "100px"},
            ),
          ],
        )
      ],
    ),
    OpenCloseModalAIO(
      "scrap:options",
      "Table extraction options",
      children=[
        html.Div(
          id="panelgroup:scrap:options",
          className="size-full grid grid-cols-[1fr_2fr_2fr]",
          children=[
            scrap_options_sidebar,
            dcc.Loading(
              parent_className="size-full",
              children=[
                dcc.Graph(
                  id="graph:scrap:preview",
                  className="size-full",
                  config={"scrollZoom": True},
                ),
              ],
              target_components={"graph:scrap:preview": "figure"},
            ),
            dag.AgGrid(
              id="table:scrap:preview",
              columnSize="autoSize",
              defaultColDef={"editable": True},
              dashGridOptions={
                "rowSelection": "multiple",
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 10,
              },
              style={"height": "100%"},
            ),
          ],
        ),
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
    dcc.Store(id="store:scrap:image-data", data={}),
  ],
)


@callback(
  Output("object:scrap:pdf", "data"),
  Input(InputButtonAIO.button_id("scrap:url"), "n_clicks"),
  State(InputButtonAIO.input_id("scrap:url"), "value"),
  prevent_initial_call=True,
  background=True,
)
def update_object(n_clicks: int, url: str):
  if not (url and n_clicks):
    return no_update

  return url


@callback(
  Output("graph:scrap:preview", "figure", allow_duplicate=True),
  Output("table:scrap:preview", "columnDefs"),
  Output("table:scrap:preview", "rowData"),
  Output("store:scrap:image-data", "data"),
  Input("button:scrap:preview", "n_clicks"),
  State("object:scrap:pdf", "data"),
  State(InputButtonAIO.input_id("scrap:pages"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:x0"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:y0"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:x1"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:y1"), "value"),
  State("dropdown:scrap:options:vertical-strategy", "value"),
  State("dropdown:scrap:options:horizontal-strategy", "value"),
  State(InputAIO.aio_id("scrap:options:min-words-vertical"), "value"),
  State(InputAIO.aio_id("scrap:options:min-words-horizontal"), "value"),
  State(InputAIO.aio_id("scrap:options:snap-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:snap-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:join-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:join-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:intersection-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:intersection-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:text-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:text-y-tolerance"), "value"),
  prevent_initial_call=True,
  background=True,
)
def preview_extraction(
  n_clicks: int,
  pdf_url: str,
  pages_text: str,
  x0: float,
  y0: float,
  x1: float,
  y1: float,
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
  if not (n_clicks and pdf_url and pages_text):
    return no_update

  def create_hover_template(
    offset_x: float,
    offset_y: float,
    pdf_width: float,
    pdf_height: float,
    img_width: float,
    img_height: float,
  ):
    x = np.linspace(offset_x, pdf_width + offset_x, img_width)
    y = np.linspace(offset_y, pdf_height + offset_y, img_height)

    X, Y = np.meshgrid(x, y)

    hovertemplate = (
      "<b>x:</b> %{customdata[0]:.2f}<br><b>y:</b> %{customdata[1]:.2f}<br>"
    )

    return X, Y, hovertemplate

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

  pages = [int(p) - 1 for p in pages_text.split(",")]

  pdf_stream = asyncio.run(download_file_memory(pdf_url))
  with pdfplumber.open(pdf_stream) as pdf:
    page = pdf.pages[pages[0]]

    if x0 and y0 and x1 and y1:
      page = page.crop((x0, y0, x1, y1), strict=True)

    debug = page.debug_tablefinder(settings)

    df = pd.DataFrame.from_records(debug.edges)
    img = page.to_image(resolution=300, antialias=True)

  prefix = "data:image/png;base64,"
  with io.BytesIO() as stream:
    img.original.save(stream, format="png")
    base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")

  offset_x = x0 or 0
  offset_y = y0 or 0
  X, Y, hovertemplate = create_hover_template(
    offset_x, offset_y, page.width, page.height, img.original.width, img.original.height
  )
  fig = go.Figure(
    go.Image(
      source=base64_string, customdata=np.dstack((X, Y)), hovertemplate=hovertemplate
    )
  )
  pixel_scale = img.original.width / page.width

  df["line"] = df["orientation"] + (df.groupby("orientation").cumcount() + 1).astype(
    str
  )
  df = df[["line", "x0", "top", "x1", "bottom"]]

  for line, x0, y0, x1, y1 in zip(
    df["line"], df["x0"], df["top"], df["x1"], df["bottom"]
  ):
    fig.add_shape(
      name=line,
      type="line",
      editable=True,
      x0=(x0 - offset_x) * pixel_scale,
      y0=(y0 - offset_y) * pixel_scale,
      x1=(x1 - offset_x) * pixel_scale,
      y1=(y1 - offset_y) * pixel_scale,
      xref="x",
      yref="y",
      line_color="red",
    )

  fig.update_layout(margin=dict(b=10, t=10))
  fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

  column_defs = [
    {
      "field": c,
      "cellDataType": "text" if df.dtypes.loc[c] in ("object", "bool") else "number",
      "valueFormatter": {
        "function": "d3.format(',.2f')(params.value)"
        if df.dtypes.loc[c] == "float"
        else None
      },
      "filter": True,
    }
    for c in df.columns
  ]
  column_defs[0].update(
    {
      "checkboxSelection": True,
    }
  )

  image_data = {
    "pixel_scale": pixel_scale,
    "offset_x": offset_x,
    "offset_y": offset_y,
  }

  return fig, column_defs, df.to_dict("records"), image_data


@callback(
  Output("graph:scrap:preview", "figure"),
  Input("table:scrap:preview", "cellValueChanged"),
  Input("table:scrap:preview", "selectedRows"),
  State("table:scrap:preview", "rowData"),
  State("store:scrap:image-data", "data"),
  prevent_initial_call=True,
)
def update_preview(
  _: dict, sel_row: list[dict], row_data: list[dict], image_data: dict
):
  if not (sel_row and row_data and image_data):
    return no_update

  df_rows = pd.DataFrame.from_records(row_data)

  sel_lines = {r["line"] for r in sel_row}

  pixel_scale = image_data["pixel_scale"]
  offset_x = image_data["offset_x"]
  offset_y = image_data["offset_y"]

  shapes = []
  for line, x0, y0, x1, y1 in zip(
    df_rows["line"], df_rows["x0"], df_rows["top"], df_rows["x1"], df_rows["bottom"]
  ):
    shapes.append(
      go.layout.Shape(
        name=line,
        type="line",
        editable=True,
        x0=(x0 - offset_x) * pixel_scale,
        y0=(y0 - offset_y) * pixel_scale,
        x1=(x1 - offset_x) * pixel_scale,
        y1=(y1 - offset_y) * pixel_scale,
        xref="x",
        yref="y",
        line_color="green" if line in sel_lines else "red",
      )
    )

  fig = Patch()
  fig["layout"]["shapes"] = shapes

  return fig


@callback(
  Output("download:scrap:image", "data"),
  Input("button:scrap:download-image", "n_clicks"),
  State(InputButtonAIO.input_id("scrap:url"), "value"),
  State(InputButtonAIO.input_id("scrap:pages"), "value"),
  prevent_initial_call=True,
  background=True,
)
def annotate_image(
  n_clicks: int,
  scrap_url: str,
  pages_text: str,
):
  if not (n_clicks and scrap_url and pages_text):
    return no_update

  pdf_path = Path(f"assets/docs/{scrap_url}.pdf")

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
  Input(InputButtonAIO.button_id("scrap:pages"), "n_clicks"),
  State("object:scrap:pdf", "data"),
  State(InputButtonAIO.input_id("scrap:pages"), "value"),
  State("radioitems:scrap:extract-method", "value"),
  State(InputAIO.aio_id("scrap:factor"), "value"),
  State(InputAIO.aio_id("scrap:currency"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:x0"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:y0"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:x1"), "value"),
  State(InputAIO.aio_id("scrap:options:bounding-box:y1"), "value"),
  State("dropdown:scrap:options:vertical-strategy", "value"),
  State("dropdown:scrap:options:horizontal-strategy", "value"),
  State(InputAIO.aio_id("scrap:options:min-words-vertical"), "value"),
  State(InputAIO.aio_id("scrap:options:min-words-horizontal"), "value"),
  State(InputAIO.aio_id("scrap:options:snap-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:snap-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:join-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:join-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:intersection-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:intersection-y-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:text-x-tolerance"), "value"),
  State(InputAIO.aio_id("scrap:options:text-y-tolerance"), "value"),
  prevent_initial_call=True,
  background=True,
)
def update_table(
  n_clicks: int,
  pdf_url: str,
  pages_text: str,
  method: Literal["text", "image"],
  factor: int,
  currency: str,
  x0: float,
  y0: float,
  x1: float,
  y1: float,
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
  if not (n_clicks and pdf_url and pages_text):
    return no_update

  pages = [int(p) - 1 for p in pages_text.split(",")]

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

  pdf_stream = asyncio.run(download_file_memory(pdf_url))
  with pdfplumber.open(pdf_stream) as pdf:
    page = pdf.pages[pages[0]]

    if x0 and y0 and x1 and y1:
      page = page.crop((x0, y0, x1, y1))

    table = page.extract_table(table_settings=settings)

  df = pd.DataFrame(table)

  df = df.loc[:, (df.notna() & (df != "")).any()]
  df = df.loc[(df.notna() & (df != "")).any(axis=1), :]
  result = df.apply(split_multiline, axis=1)
  df = pd.concat(result.tolist(), ignore_index=True)

  df["period"] = "instant"
  df["factor"] = factor
  df["unit"] = currency
  diff = ["period", "factor", "unit"]
  df = df[diff + list(df.columns.difference(diff))]

  column_defs: list[dict[str, str | bool | dict]] = [
    {"field": str(c)} for c in df.columns
  ]
  column_defs[0].update(
    {
      "checkboxSelection": True,
      "cellEditor": "agSelectCellEditor",
      "cellEditorParams": {"values": ["instant", "duration"]},
    }
  )
  column_defs[1].update({"type": "numericColumn"})

  return column_defs, df.to_dict("records"), False


@callback(
  Output("table:scrap", "deleteSelectedRows"),
  Input("button:scrap:delete-rows", "n_clicks"),
  prevent_initial_call=True,
)
def delete_rows(_: int):
  return True


@callback(
  Output("table:scrap", "rowTransaction"),
  Input("button:scrap:add-rows", "n_clicks"),
  State("table:scrap", "columnDefs"),
  prevent_initial_call=True,
)
def add_rows(_: int, cols: list[dict]):
  if not cols:
    return no_update

  return {"addIndex": 0, "add": [{col["field"]: None for col in cols}]}


@callback(
  Output("form:scrap:columns", "children"),
  Input("table:scrap", "columnDefs"),
  prevent_initial_call=True,
)
def delete_columns_form(cols: list[dict]):
  if not cols:
    return no_update

  return dcc.Checklist(
    id="checklist:scrap:columns",
    options=[col["field"] for col in cols[3:]],
    value=[],
  )


@callback(
  Output("form:scrap:rename-columns", "children"),
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
  Input("button:scrap:delete-columns", "n_clicks"),
  Input("button:scrap:rename-columns", "n_clicks"),
  State("checklist:scrap:columns", "value"),
  State({"type": "input:scrap:headers", "index": ALL}, "value"),
  State("table:scrap", "columnDefs"),
  State("table:scrap", "rowData"),
  prevent_initial_call=True,
)
def update_columns(
  n_delete: int,
  n_update: int,
  del_cols: list[str],
  new_names: list[str],
  cols: list[dict],
  rows: list[dict],
):
  if not (cols and rows):
    return no_update

  df = pd.DataFrame.from_records(rows)
  df = df[[col["field"] for col in cols]]

  if ctx.triggered_id == "button:scrap:delete-columns":
    new_cols = [col for col in cols if col["field"] not in del_cols]
    df.drop(columns=del_cols, inplace=True)

    return new_cols, df.to_dict("records")

  if ctx.triggered_id == "button:scrap:rename-columns":
    col_map = {col: name for (col, name) in zip(df.columns[3:], new_names)}
    df.rename(columns=col_map, inplace=True)

    for i, name in enumerate(new_names):
      cols[i + 3]["field"] = name

    return cols, df.to_dict("records")

  return no_update


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
  Output("table:scrap", "exportDataAsCsv"),
  Input("button:scrap:export-csv", "n_clicks"),
)
def export_csv(n_clicks: int):
  if n_clicks:
    return True
  return False


@callback(
  Output("table:scrap:items", "rowData"),
  Input(OpenCloseModalAIO.open_id("scrap:record-items"), "n_clicks"),
  State("table:scrap", "rowData"),
  prevent_initial_call=True,
)
def item_modal(n_clicks: int, rows: list[dict]):
  if not (n_clicks and rows):
    return no_update

  df = pd.DataFrame.from_records(rows)
  if "item" not in set(df.columns):
    return no_update

  df.loc[:, "item"] = df["item"].apply(lambda x: pascal_case(x))

  empty_columns = pd.DataFrame(
    "",
    index=df.index,
    columns=["taxonomy", "label_long", "label_short", "type", "balance", "aggregate"],
  )
  row_data = pd.concat([df[["item"]], empty_columns], axis=1)

  return row_data.to_dict("records")


@callback(
  Output("table:scrap:items", "rowData", allow_duplicate=True),
  Input("button:scrap:search-items", "n_clicks"),
  State("table:scrap:items", "rowData"),
  prevent_initial_call=True,
  background=True,
)
def search_items(n_clicks: int, rows: list[dict]):
  if not (n_clicks and rows):
    return no_update

  for i in range(len(rows)):
    find = search_taxonomy(rows[i]["item"])
    rows[i]["action"] = "create"
    if find is not None:
      rows[i]["action"] = "none"
      rows[i]["taxonomy"] = find["item"]

  return rows


@callback(
  Output("download:scrap:json", "data"),
  Input("button:scrap:export-json", "n_clicks"),
  State("table:scrap", "rowData"),
  State(InputButtonAIO.input_id("scrap:url"), "value"),
  State(CompanySelectAIO.aio_id("scrap:company-id"), "value"),
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
    return no_update

  df.loc[:, "item"] = df["item"].apply(lambda x: pascal_case(x))

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

  record = FinStatement(
    url=url,
    date=dt.strptime(date, "%Y-%m-%d").date(),
    scope=scope,
    fiscal_period=period,
    fiscal_end=fiscal_end,
    currency=currencies,
    data=data,
  )

  # upsert_statements("statements.db", id, records)

  return {
    "content": record.model_dump_json(exclude_unset=True, indent=2),
    "filename": f"{id}_{date}_{period}.json",
  }
