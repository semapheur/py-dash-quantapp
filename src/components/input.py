from dash import dcc, html

input_style = (
  "peer h-full w-full p-1 bg-primary text-text "
  "rounded border border-text/10 hover:border-text/50 focus:border-secondary "
  "placeholder-transparent"
)
label_style = (
  "absolute left-1 top-0 -translate-y-1/2 px-1 bg-primary text-text/50 text-xs "
  "peer-placeholder-shown:text-base peer-placeholder-shown:text-text/50 "
  "peer-placeholder-shown:top-1/2 peer-focus:top-0 peer-focus:-translate-y-1/2 "
  "peer-focus:text-secondary peer-focus:text-xs transition-all"
)


class InputAIO(html.Form):
  @staticmethod
  def aio_id(id: str):
    return {"component": "input-aio", "aio_id": id}

  def __init__(
    self,
    id: str,
    width: str | None = None,
    form_props: dict | None = None,
    input_props: dict | None = None,
  ):
    form_props = form_props.copy() if form_props is not None else {}
    form_props.setdefault("className", "")

    input_props = input_props.copy() if input_props is not None else {}
    input_props.setdefault("className", input_style)
    input_props.setdefault("placeholder", "")
    input_props.setdefault("type", "text")

    form_style = {"width": width or "auto"}

    super().__init__(
      className=f"peer relative {form_props["className"]}".strip(),
      style=form_style,
      children=[
        dcc.Input(id=self.__class__.aio_id(id), **input_props),
        html.Label(
          htmlFor=str(self.__class__.aio_id(id)),
          className=label_style,
          children=[input_props["placeholder"]],
        ),
      ],
    )
