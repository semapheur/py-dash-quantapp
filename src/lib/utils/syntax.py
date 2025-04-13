def get_constructor_args(class_obj) -> set[str]:
  import inspect

  signature = inspect.signature(class_obj.__init__)
  return set(signature.parameters.keys())
