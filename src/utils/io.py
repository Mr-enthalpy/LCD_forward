from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_config(path: str | Path, config: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

