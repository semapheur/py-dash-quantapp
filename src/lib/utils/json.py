import json
import re
from pathlib import Path


def load_json(path: str | Path) -> dict:
  with open(path, "r") as f:
    return json.load(f)


def update_json(path: str | Path, data: dict):
  if isinstance(path, str):
    path = Path(path)

  if path.suffix != ".json":
    path = path.with_suffix(".json")

  try:
    with open(path, "r") as f:
      file_data = json.load(f)

  except (FileNotFoundError, json.JSONDecodeError):
    file_data = {}

  file_data.update(data)

  with open(path, "w") as f:
    json.dump(file_data, f)


def minify_json(path: str | Path, new_name: str | None = None):
  if isinstance(path, str):
    path = Path(path)

  with open(path, "r") as f:
    data = json.load(f)

  if not new_name:
    new_path = path.with_name(f"{path.stem}_mini.json")
  else:
    new_path = path.with_name(new_name).with_suffix(".json")

  with open(new_path, "w") as f:
    json.dump(data, f, separators=(",", ":"))
