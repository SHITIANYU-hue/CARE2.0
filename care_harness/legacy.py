"""One import bridge to the existing research implementations."""

from importlib import import_module as _import_module
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = REPO_ROOT / "experiments" / "care_replay"
SCRIPTS = REPLAY_ROOT / "scripts"


def import_module(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return _import_module(name)
