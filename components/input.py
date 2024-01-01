from dash import dcc, html
import uuid

input_style = (
  'peer h-full p-1 bg-primary text-text '
  'rounded border border-text/10 hover:border-text/50 focus:border-secondary '
  'placeholder-transparent'
)
label_style = (
  'absolute left-1 -top-2 px-1 bg-primary text-text/500 text-xs '
  'peer-placeholder-shown:text-base peer-placeholder-shown:text-text/50 '
  'peer-placeholder-shown:top-1 peer-focus:-top-2 peer-focus:text-secondary '
  'peer-focus:text-xs transition-all'
)

class InputAIO(html.Form):
  @staticmethod
  def _id(aio_id: str):
    return {
      'component': 'input-aio',
      'aio_id': aio_id
    }
  
  def __init__(self, aio_id:str=None, input_props:dict=None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    input_props = input_props.copy() if input_props else {}

    input_props.setdefault('className', input_style)
    input_props.setdefault('placeholder', '')
    input_props.setdefault('type', 'text')

    super().__init__(children=[
      dcc.Input(
        id=self.__class__._id(aio_id), 
        **input_props
      ),
      html.Label(
        htmlFor=str(self.__class__._id(aio_id)), 
        className=label_style, 
        children=[input_props['placeholder']]
      ),
    ])