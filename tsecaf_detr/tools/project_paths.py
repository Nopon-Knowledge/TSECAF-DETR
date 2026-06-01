from __future__ import annotations

import os
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DATASETS_ROOT = Path(os.environ.get("WHEAT_DATASETS_ROOT", str(PROJECT_ROOT / "datasets"))).expanduser()


def resolve_first_existing(*candidates: str | Path) -> str | None:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None
