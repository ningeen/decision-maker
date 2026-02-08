from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from models import Project

DATA_DIR = Path("data")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    return cleaned.lower() or "project"


def project_path(name: str) -> Path:
    ensure_data_dir()
    return DATA_DIR / f"{slugify(name)}.json"


def list_projects() -> List[str]:
    ensure_data_dir()
    return sorted(path.stem for path in DATA_DIR.glob("*.json"))


def save_project(project: Project) -> Path:
    path = project_path(project.name)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(project.to_dict(), handle, indent=2)
    return path


def load_project(name: str) -> Project | None:
    path = project_path(name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Project.from_dict(data)
