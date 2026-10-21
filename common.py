"""Shared helpers: where files live, settings, small utilities."""
import os
import sys
from datetime import datetime
from pathlib import Path

APP_NAME = "TFT Lab Coach"
DEFAULT_LIVE_MODEL = "gemini-3.8-live"
DEFAULT_REVIEW_MODEL = "gemini-2.5-flash"
DEFAULT_TALK_EVERY = 8


def app_dir() -> Path:
    """Folder next to the exe (frozen) or next to this file (source).

    Never the PyInstaller temp folder (sys._MEIPASS), which is deleted on exit.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def env_path() -> Path:
    return app_dir() / ".env"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def live_model() -> str:
    return os.environ.get("TFT_LIVE_MODEL") or DEFAULT_LIVE_MODEL


def review_model() -> str:
    return os.environ.get("TFT_REVIEW_MODEL") or DEFAULT_REVIEW_MODEL


def self_command(*args: str) -> list:
    """Command line that re-launches this app with extra arguments."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, str(Path(__file__).resolve().parent / "app.py"), *args]


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
