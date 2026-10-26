"""One place for the classified source tree's project and runtime paths.

Configuration, logs and generated files belong beside the root launch scripts,
not inside app/. Flat disposable test projects keep their historical layout.
"""
from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR


def project_path(value: str | Path) -> Path:
    """Resolve relative runtime data under the project; preserve absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def bot_script_path(project: str | Path) -> Path:
    """Select one exact script location, never a substring or process-name match."""
    base = Path(project).resolve()
    # An app directory denotes the new layout even if its bot is missing: the
    # preflight must reject that broken install rather than pick another script.
    return base / "app" / "bot.py" if (base / "app").is_dir() else base / "bot.py"
