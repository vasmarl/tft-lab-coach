"""Videos only, never live matches: detect a running League game and hide
League/Riot windows from the window picker."""
import psutil

# A running match: the game client itself.
GAME_PROCESSES = {"league of legends.exe"}

# Never offered in the window picker.
HIDDEN_PROCESSES = {
    "league of legends.exe",
    "leagueclient.exe",
    "leagueclientux.exe",
    "leagueclientuxrender.exe",
    "riotclientservices.exe",
    "riot client.exe",
    "riotclientux.exe",
}
HIDDEN_TITLE_WORDS = ("league of legends", "riot client", "teamfight tactics")

BLOCK_MESSAGE = ("A League of Legends match is running. TFT Lab Coach only works on "
                 "recorded videos, never on live games.")


def running_process_names() -> list:
    names = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.info["name"]:
                names.append(p.info["name"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return names


def game_running(process_names=None) -> bool:
    """True if a League match (League of Legends.exe) is running.

    The League client alone (LeagueClientUx.exe etc.) is fine: it's open
    all the time while people watch videos.
    """
    if process_names is None:
        process_names = running_process_names()
    return any(n.lower() in GAME_PROCESSES for n in process_names)


def is_hidden_window(title: str, process: str) -> bool:
    if (process or "").lower() in HIDDEN_PROCESSES:
        return True
    t = (title or "").lower()
    return any(w in t for w in HIDDEN_TITLE_WORDS)


def filter_windows(windows, own_pid=None):
    """windows: iterable of (hwnd, title, process, pid). Keeps allowed ones."""
    kept = []
    for hwnd, title, process, pid in windows:
        if own_pid is not None and pid == own_pid:
            continue
        if not title.strip() or is_hidden_window(title, process):
            continue
        kept.append((hwnd, title, process, pid))
    return kept


def label(window) -> str:
    _, title, process, _ = window
    return f"{title} [{process}]"
