"""Windows window listing and client-area capture (pywin32 + mss)."""
import ctypes
import os
import sys

import numpy as np
import psutil

try:
    import win32gui
    import win32process
except ImportError:  # not Windows (tests mock these)
    win32gui = None
    win32process = None


def set_dpi_aware():
    """Per-monitor DPI awareness so client rects match real pixels on scaled displays."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"


def list_windows() -> list:
    """All visible top-level windows with a title: (hwnd, title, process, pid)."""
    found = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        found.append((hwnd, title, _process_name(pid), pid))
        return True

    win32gui.EnumWindows(cb, None)
    return found


def own_pids() -> set:
    pids = {os.getpid()}
    try:
        pids.add(os.getppid())   # the launcher that started us
    except Exception:
        pass
    return pids


def window_visible(hwnd) -> bool:
    return bool(win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)
                and not win32gui.IsIconic(hwnd))


def grab_window(hwnd, sct):
    """BGR numpy frame of the window's client area, or None if not visible."""
    if not window_visible(hwnd):
        return None
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width, height = right - left, bottom - top
    if width < 16 or height < 16:
        return None
    x, y = win32gui.ClientToScreen(hwnd, (0, 0))
    shot = sct.grab({"left": x, "top": y, "width": width, "height": height})
    bgra = np.asarray(shot)
    return np.ascontiguousarray(bgra[:, :, :3])
