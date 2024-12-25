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

register_page(__name__, path="/scrap/htm", title="Scrap HTM")

main_style = "size-full bg-primary"

resize_handle_style = "h-full w-0.5 bg-text/50 hover:bg-secondary hover:w-1"

html.Main(
  className=main_style,
  children=[
    PanelGroup(
      id="panelgroup:scrap-htm",
      direction="horizontal",
      className="size-full",
      children=[
        Panel(
          id="panel:scrap-htm:controls",
          defaultSizePercentage=20,
          minSizePixels=5,
          children=[scrap_controls_sidebar],
        ),
        PanelResizeHandle(html.Div(className=resize_handle_style)),
        Panel(
          id="panel:scrap-htm:pdf",
          defaultSizePercentage=40,
          minSizePixels=5,
          children=[
            dcc.Loading(
              parent_className="size-full",
              children=[
                html.Iframe(
                  id="iframe:scrap-htm:htm",
                  width="100%",
                  height="100%",
                )
              ],
              target_components={"object:scrap-htm:pdf": "data"},
            ),
          ],
        ),
        PanelResizeHandle(html.Div(className=resize_handle_style)),
        Panel(
          id="panel:scrap-htm:table",
          defaultSizePercentage=40,
          minSizePixels=5,
          children=[
            dag.AgGrid(
              id="table:scrap-htm",
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
    )
  ],
)
