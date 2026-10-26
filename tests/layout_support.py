"""Resolve the real app/tests layout without flattening or reading private config."""
from pathlib import Path
import sys


def project_directory(value=None):
    path = Path(value).resolve() if value is not None else Path(__file__).resolve().parents[1]
    if path.name in {"app", "tests", "tools"} and (path.parent / "app" / "bot.py").is_file():
        return path.parent
    if (path / "app" / "bot.py").is_file():
        return path
    raise ValueError("Expected the actual project directory containing app/bot.py")


def app_directory(value=None):
    return project_directory(value) / "app"


def configure_test_paths(value=None):
    project = project_directory(value)
    for path in (project / "app", project / "tests"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return project
