import atexit
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
from queue import Queue
from typing_extensions import override

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
  logging.config.dictConfig(config)
  # queue_handler = logging.getHandlerByName('queue_handler')
  # if queue_handler is not None:
  #  queue_handler.listener.start()
  #  atexit.register(queue_handler.listener.stop)


def resolve_handlers(h):
  if not isinstance(h, logging.config.ConvertingList):
    return h

  return [h[i] for i in range(len(h))]


def resolve_queue(q):
  if not isinstance(q, logging.config.ConvertingDict):
    return q

  if '__resolved_value__' in q:
    return q['__resolved_value__']

  class_name = q.pop('class')
  class_ = q.configurator.resolve(class_name)
  props = q.pop('.', None)
  kwargs = {k: q[k] for k in q if logging.config.valid_ident(k)}
  result = class_(**kwargs)

  if props:
    for name, value in props.items():
      setattr(result, name, value)

  q['__resolved_value__'] = result
  return result


class QueueListenerHandler(logging.handlers.QueueHandler):
  def __init__(
    self,
    handlers,
    respect_handler_level=False,
    auto_run=True,
    queue=Queue(-1),
  ):
    super().__init__(queue)
    handlers = resolve_handlers(handlers)

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

  def emit(self, record: logging.handlers.LogRecord):
    return super().emit(record)


class LogJSONFormatter(logging.Formatter):
  def __init__(self, fmt_keys: dict[str, str] | None = None):
    super().__init__()
    self.fmt_keys = fmt_keys if fmt_keys is not None else {}

  @override
  def format(self, record: logging.LogRecord) -> str:
    message = self._prepare_log_dict(record)
    return json.dumps(message, default=str)

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
