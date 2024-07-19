from typing import Literal

from dash import (
  clientside_callback,
  ClientsideFunction,
  html,
  Input,
  Output,
  State,
  MATCH,
)


class CloseModalAIO(html.Dialog):
  modal_type: Literal["close", "open-close"] = "close"

  @classmethod
  def dialog_id(cls, aio_id: str):
    return {"component": f"{cls.modal_type}-modal-aio", "aio_id": aio_id}

  @classmethod
  def close_id(cls, aio_id: str):
    return {
      "component": f"{cls.modal_type}-modal-aio",
      "subcomponent": "button:close",
      "aio_id": aio_id,
    }

  def __init__(
    self,
    aio_id: str,
    title: str | None = None,
    children: list = [],
    dialog_props: dict | None = None,
  ):
    dialog_class = "m-auto rounded-md shadow-md dark:shadow-black/50"
    dialog_style = {
      "maxHeight": "75%",
      "maxWidth": "75%",
    }

    dialog_props = dialog_props.copy() if dialog_props else {}
    dialog_props.setdefault("className", dialog_class)
    dialog_props.setdefault("style", dialog_style)

    super().__init__(
      id=self.__class__.dialog_id(aio_id),
      **dialog_props,
      children=[
        html.Div(
          className="flex flex-col h-full px-2 pb-2 bg-primary",
          children=[
            html.Header(
              className="grid grid-cols-[1fr_auto] gap-4",
              children=[
                html.H1(
                  title if title is not None else "", className="m-auto text-text"
                ),
                html.Button(
                  "X",
                  id=self.__class__.close_id(aio_id),
                  className="self-end text-text hover:text-red-600",
                ),
              ],
            ),
            *children,
          ],
        )
      ],
    )


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="close_modal"),
  Output(CloseModalAIO.dialog_id(MATCH), "id"),
  Input(CloseModalAIO.close_id(MATCH), "n_clicks"),
  State(CloseModalAIO.dialog_id(MATCH), "id"),
  prevent_initial_call=True,
)


class OpenCloseModalAIO(CloseModalAIO):
  modal_type = "open-close"

  @classmethod
  def open_id(cls, aio_id: str):
    return {
      "component": f"{cls.modal_type}-modal-aio",
      "subcomponent": "button:open",
      "aio_id": aio_id,
    }

  def __init__(
    self,
    aio_id: str,
    title: str | None = None,
    children: list = [],
    dialog_props: dict | None = None,
  ):
    super().__init__(
      aio_id=aio_id, title=title, children=children, dialog_props=dialog_props
    )


clientside_callback(
  ClientsideFunction(namespace="clientside", function_name="handle_modal"),
  Output(OpenCloseModalAIO.dialog_id(MATCH), "id"),
  Input(OpenCloseModalAIO.open_id(MATCH), "n_clicks_timestamp"),
  Input(OpenCloseModalAIO.close_id(MATCH), "n_clicks_timestamp"),
  State(OpenCloseModalAIO.dialog_id(MATCH), "id"),
  prevent_initial_call=True,
)
