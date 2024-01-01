from dash import clientside_callback, ClientsideFunction, html, Input, Output, State, MATCH
import uuid

dialog_style = 'm-auto max-h-[75%] max-w-[75%] rounded-md shadow-md dark:shadow-black/50'

class ModalAIO(html.Dialog):
  @staticmethod
  def dialog_id(aio_id:str):
    return {
      'component': 'modal-aio:dialog',
      'aio_id': aio_id
    }
  
  @staticmethod
  def close_id(aio_id:str):
    return {
      'component': 'modal-aio:button:close',
      'aio_id': aio_id
    }
  
  @staticmethod
  def open_id(aio_id:str):
    return {
      'component': 'modal-aio:button:open',
      'aio_id': aio_id
    }
  
  def __init__(self, aio_id:str=None, title:str=None, children:list=[], dialog_props:dict=None):
    if aio_id is None:
      aio_id = str(uuid.uuid4())

    dialog_props = dialog_props.copy() if dialog_props else {}
    dialog_props.setdefault('className', dialog_style)

    super().__init__(id=self.__class__.dialog_id(aio_id), **dialog_props, children=[
      html.Div(className='flex flex-col h-full px-2 pb-2', children=[
        html.Header(className='grid grid-cols-[1fr_auto] gap-4', children=[
          html.H1(title if title is not None else 'Modal',  className='m-auto text-text'),
          html.Button('X', 
            id=self.__class__.close_id(aio_id),
            className='self-end text-text hover:text-red-600'
          )
        ]),
        *children
      ])
    ])

  clientside_callback(
  ClientsideFunction(
    namespace='clientside',
    function_name='handle_modal'
  ),
    Output(dialog_id(MATCH), 'id'),
    Input(open_id(MATCH), 'n_clicks_timestamp'),
    Input(close_id(MATCH), 'n_clicks_timestamp'),
    State(dialog_id(MATCH), 'id')
  )