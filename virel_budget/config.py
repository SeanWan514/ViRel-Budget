from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    config["_root_dir"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_root_dir"]) / path
