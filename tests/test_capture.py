from types import SimpleNamespace as NS

import numpy as np

import capture


class FakeSct:
    def __init__(self):
        self.asked = None

    def grab(self, mon):
        self.asked = mon
        # mss returns BGRA pixels
        img = np.zeros((mon["height"], mon["width"], 4), np.uint8)
        img[..., 0] = 10   # B
        img[..., 1] = 20   # G
        img[..., 2] = 30   # R
        img[..., 3] = 255  # A
        return img


def fake_win32(visible=True, iconic=False, size=(1280, 720), origin=(100, 50)):
    return NS(
        IsWindow=lambda h: True,
        IsWindowVisible=lambda h: visible,
        IsIconic=lambda h: iconic,
        GetClientRect=lambda h: (0, 0, size[0], size[1]),
        ClientToScreen=lambda h, pt: (origin[0] + pt[0], origin[1] + pt[1]),
    )


def test_grab_returns_bgr_frame_of_client_size(monkeypatch):
    monkeypatch.setattr(capture, "win32gui", fake_win32())
    sct = FakeSct()
    frame = capture.grab_window(1234, sct)
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
    assert tuple(frame[0, 0]) == (10, 20, 30)
    assert sct.asked == {"left": 100, "top": 50, "width": 1280, "height": 720}


def test_grab_minimized_window_returns_none(monkeypatch):
    monkeypatch.setattr(capture, "win32gui", fake_win32(iconic=True))
    assert capture.grab_window(1, FakeSct()) is None


def test_list_windows_uses_enumwindows(monkeypatch):
    titles = {1: "TFT guide - YouTube - Google Chrome", 2: "", 3: "Hidden"}
    win = NS(
        EnumWindows=lambda cb, extra: [cb(h, extra) for h in titles],
        IsWindowVisible=lambda h: h != 3,
        GetWindowText=lambda h: titles[h],
    )
    monkeypatch.setattr(capture, "win32gui", win)
    monkeypatch.setattr(capture, "win32process", NS(GetWindowThreadProcessId=lambda h: (0, 4242)))
    monkeypatch.setattr(capture, "_process_name", lambda pid: "chrome.exe")
    assert capture.list_windows() == [(1, "TFT guide - YouTube - Google Chrome", "chrome.exe", 4242)]
