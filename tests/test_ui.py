"""Build the tkinter screens (runs on the Windows CI runner; skipped without Tk)."""
import sys
import time

import pytest

tk = pytest.importorskip("tkinter")
if sys.platform != "win32":
    pytest.skip("Tk UI smoke test runs on Windows", allow_module_level=True)

import capture  # noqa: E402
import keystore  # noqa: E402
import launcher  # noqa: E402
import screen_mode  # noqa: E402


@pytest.fixture(scope="module")
def tk_root():
    # one Tk per process: a second Tk() fails on the hosted CI Python
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def root(tk_root):
    win = tk.Toplevel(tk_root)
    yield win
    try:
        win.destroy()
    except tk.TclError:
        pass


def pump(root, until=lambda: False, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end and not until():
        root.update()


def test_key_screen_then_main(root, monkeypatch):
    monkeypatch.setattr(keystore, "load_key", lambda: "")
    saved = {}
    monkeypatch.setattr(keystore, "check_key", lambda k: (True, keystore.MSG_OK))
    monkeypatch.setattr(keystore, "save_key", lambda k: saved.setdefault("key", k))
    app = launcher.Launcher(root)
    app.key_entry.insert(0, "AIzaTEST")
    app.on_connect()
    pump(root, until=lambda: "key" in saved)
    assert saved["key"] == "AIzaTEST"
    texts = [w.cget("text") for w in app.frame.winfo_children() if isinstance(w, tk.Label)]
    assert "TFT Lab Coach" in texts


def test_bad_key_message(root, monkeypatch):
    monkeypatch.setattr(keystore, "load_key", lambda: "")
    monkeypatch.setattr(keystore, "check_key", lambda k: (False, keystore.MSG_INVALID))
    app = launcher.Launcher(root)
    app.key_entry.insert(0, "nope")
    app.on_connect()
    pump(root, until=lambda: app.key_msg.cget("text") == keystore.MSG_INVALID)
    assert app.key_msg.cget("text") == keystore.MSG_INVALID


def test_window_picker_lists_allowed_windows(root, monkeypatch):
    monkeypatch.setattr(capture, "list_windows", lambda: [
        (1, "TFT guide - YouTube - Google Chrome", "chrome.exe", 10),
        (2, "League of Legends", "LeagueClientUx.exe", 11),
    ])
    sc = screen_mode.ScreenCoach(root, "k", 8)
    pump(root, seconds=0.2)
    assert sc.listbox.get(0, "end") == ("TFT guide - YouTube - Google Chrome [chrome.exe]",)
