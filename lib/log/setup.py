import atexit
import datetime as dt
import json
import logging
from logging.config import dictConfig, valid_ident, ConvertingList, ConvertingDict
import logging.handlers
from pathlib import Path
from queue import Queue
from typing import cast
from typing_extensions import override

from lib.utils import get_constructor_args

LOG_RECORD_BUILTIN_ATTRS = {
  'args',
  'asctime',
  'created',
  'exc_info',
  'exc_text',
  'filename',
  'funcName',
  'levelname',
  'levelno',
  'lineno',
  'module',
  'msecs',
  'message',
  'msg',
  'name',
  'pathname',
  'process',
  'processName',
  'relativeCreated',
  'stack_info',
  'thread',
  'threadName',
  'taskName',
}


def setup_queue_handler():
  queue = Queue(-1)
  queue_handler = logging.handlers.QueueHandler(queue)
  listener = logging.handlers.QueueListener(
    queue,
    logging.StreamHandler(),
    logging.handlers.RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3),
  )
  listener.start()
  atexit.register(listener.stop)

  return queue_handler


def setup_logging():
  config_file = Path('lib/log/logging_config.json')

  with open(config_file, 'r') as f:
    config = json.load(f)
  dictConfig(config)
  # queue_handler = logging.getHandlerByName('queue_handler')
  # if queue_handler is not None:
  #  queue_handler.listener.start()
  #  atexit.register(queue_handler.listener.stop)


def resolve_object(config: ConvertingDict):
  if (resolved := config.get('__resolved_value')) is not None:
    return resolved

  constructor = config.configurator.resolve(config.pop('class', config.pop('()')))
  kwargs_set = get_constructor_args(constructor.__init__)
  kwargs = {k: config[k] for k in config if valid_ident(k) and k in kwargs_set}
  obj = constructor(**kwargs)

  props = config.get('.')
  if props is not None:
    for name, value in props.items():
      setattr(obj, name, value)

  return obj


def resolve_objects(configs: ConvertingList) -> list[logging.Handler]:
  return [
    resolve_object(configs[i]) if isinstance(configs[i], ConvertingDict) else configs[i]
    for i in range(len(configs))
  ]


class QueueListenerHandler(logging.handlers.QueueHandler):
  def __init__(
    self,
    handlers: ConvertingList | list[logging.Handler],
    queue: ConvertingDict | Queue = Queue(-1),
    respect_handler_level=False,
    auto_run=True,
  ):
    if isinstance(queue, ConvertingDict):
      queue = cast(Queue, resolve_object(queue))
    super().__init__(queue)

    if isinstance(handlers, ConvertingList):
      handlers = cast(list[logging.Handler], resolve_objects(handlers))

    self.listener = logging.handlers.QueueListener(
      self.queue, *handlers, respect_handler_level=respect_handler_level
    )
    if auto_run:
      self.start()
      atexit.register(self.stop)

  def start(self):
    self.listener.start()

  def stop(self):
    self.listener.stop()

  def emit(self, record: logging.LogRecord):
    return super().emit(record)


class LogJSONFormatter(logging.Formatter):
  def __init__(self, fmt_keys: dict[str, str] | None = None):
    super().__init__()
    self.fmt_keys = fmt_keys if fmt_keys is not None else {}

  @override
  def format(self, record: logging.LogRecord) -> str:
    message = self._prepare_log_dict(record)
    return json.dumps(message, default=str, indent=2)

  def _prepare_log_dict(self, record: logging.LogRecord):
    always_fields = {
      'message': record.getMessage(),
      'timestamp': dt.datetime.fromtimestamp(
        record.created, tz=dt.timezone.utc
      ).isoformat(),
    }

    if record.exc_info is not None:
      always_fields['exc_info'] = self.formatException(record.exc_info)

    if record.stack_info is not None:
      always_fields['stack_info'] = self.formatStack(record.stack_info)

    message = {
      key: msg_val
      if (msg_val := always_fields.pop(val, None)) is not None
      else getattr(record, val)
      for key, val in self.fmt_keys.items()
    }
    message.update(always_fields)

    for key, val in record.__dict__.items():
      if key not in LOG_RECORD_BUILTIN_ATTRS:
        message[key] = val

    return message
