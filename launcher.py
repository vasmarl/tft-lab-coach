"""Launcher window: key setup screen + main screen with the two modes."""
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox

import keystore
from common import APP_NAME, DEFAULT_TALK_EVERY, self_command

KEY_URL = "https://aistudio.google.com/apikey"

BG = "#14161c"
CARD = "#1e2129"
FG = "#f2f3f5"
MUTED = "#9aa0ab"
ACCENT = "#e8b04a"
GOOD = "#5fd18a"
BAD = "#ff7a7a"
VIDEO_TYPES = [("Video files", "*.mp4 *.mkv *.mov *.avi"), ("All files", "*.*")]


def launch_mode(*args: str):
    """Start a mode as its own process so the launcher stays responsive."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(self_command(*args), creationflags=flags, close_fds=True)


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_NAME)
        root.configure(bg=BG)
        root.geometry("560x520")
        root.minsize(520, 480)
        self.frame = None
        self.talk_every = tk.IntVar(value=DEFAULT_TALK_EVERY)
        if keystore.load_key():
            self.show_main()
        else:
            self.show_key_screen()

    # ---------- helpers ----------
    def _clear(self):
        if self.frame is not None:
            self.frame.destroy()
        self.frame = tk.Frame(self.root, bg=BG, padx=28, pady=22)
        self.frame.pack(fill="both", expand=True)
        return self.frame

    def _label(self, parent, text, size=11, color=FG, bold=False, **pack):
        lbl = tk.Label(parent, text=text, bg=parent["bg"], fg=color, justify="left",
                       font=("Segoe UI", size, "bold" if bold else "normal"),
                       wraplength=480, anchor="w")
        lbl.pack(fill="x", **pack)
        return lbl

    def _button(self, parent, text, command, big=False, color=ACCENT):
        btn = tk.Button(parent, text=text, command=command, bg=color, fg="#111111",
                        activebackground=FG, relief="flat", cursor="hand2",
                        font=("Segoe UI", 13 if big else 10, "bold"),
                        padx=14, pady=12 if big else 6)
        return btn

    # ---------- key screen ----------
    def show_key_screen(self, can_go_back=False):
        f = self._clear()
        self._label(f, "Connect your Google AI key", 18, bold=True, pady=(0, 6))
        self._label(f, "The coach runs on Google Gemini. It's free to start.", 11, MUTED, pady=(0, 14))
        steps = tk.Frame(f, bg=CARD, padx=16, pady=12)
        steps.pack(fill="x")
        for s in ("1.  Click \"Get a free key\" and sign in with Google.",
                  "2.  Click \"Create API key\" and copy it.",
                  "3.  Paste it below and click Connect."):
            self._label(steps, s, 11, pady=2)
        self._button(f, "Get a free key  ↗", lambda: webbrowser.open(KEY_URL)).pack(anchor="w", pady=14)

        self._label(f, "Paste your key:", 10, MUTED)
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x", pady=(4, 6))
        self.key_entry = tk.Entry(row, show="•", font=("Segoe UI", 12), bg=CARD, fg=FG,
                                  insertbackground=FG, relief="flat")
        self.key_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.connect_btn = self._button(row, "Connect", self.on_connect)
        self.connect_btn.pack(side="left", padx=(8, 0))
        self.key_entry.bind("<Return>", lambda e: self.on_connect())
        self.key_entry.focus_set()
        self.key_msg = self._label(f, "", 10, MUTED, pady=(4, 0))
        if can_go_back:
            back = tk.Label(f, text="← Back", fg=MUTED, bg=BG, cursor="hand2", font=("Segoe UI", 10, "underline"))
            back.pack(anchor="w", pady=(16, 0))
            back.bind("<Button-1>", lambda e: self.show_main())

    def on_connect(self):
        key = self.key_entry.get().strip()
        if not key:
            self.key_msg.config(text="Paste your key first.", fg=BAD)
            return
        self.connect_btn.config(state="disabled")
        self.key_msg.config(text="Checking the key...", fg=MUTED)

        result = queue.Queue()
        threading.Thread(target=lambda: result.put(keystore.check_key(key)), daemon=True).start()
        self._wait_key_check(key, result)

    def _wait_key_check(self, key, result):
        """Poll the worker's result on the UI thread (Tk isn't thread-safe)."""
        try:
            ok, msg = result.get_nowait()
        except queue.Empty:
            self.root.after(100, self._wait_key_check, key, result)
            return
        self._key_checked(key, ok, msg)

    def _key_checked(self, key, ok, msg):
        self.connect_btn.config(state="normal")
        if not ok:
            self.key_msg.config(text=msg, fg=BAD)
            return
        try:
            keystore.save_key(key)
        except OSError as e:
            self.key_msg.config(text=f"Key works, but couldn't save it next to the app: {e}", fg=BAD)
            return
        self.show_main()

    # ---------- main screen ----------
    def show_main(self):
        f = self._clear()
        self._label(f, "TFT Lab Coach", 22, bold=True)
        self._label(f, "Live AI coaching on recorded TFT games", 11, MUTED, pady=(0, 18))

        self._button(f, "1   Coach a video file", self.coach_file, big=True).pack(fill="x", pady=5)
        self._button(f, "2   Coach a video on my screen", self.coach_screen, big=True).pack(fill="x", pady=5)

        sl = tk.Frame(f, bg=BG)
        sl.pack(fill="x", pady=(16, 4))
        self._label(sl, "Talk every (sec)", 10, MUTED)
        tk.Scale(sl, from_=4, to=20, orient="horizontal", variable=self.talk_every,
                 bg=BG, fg=FG, troughcolor=CARD, highlightthickness=0,
                 activebackground=ACCENT).pack(fill="x")

        keyrow = tk.Frame(f, bg=BG)
        keyrow.pack(fill="x", pady=(14, 0))
        tk.Label(keyrow, text="Google AI key connected ✓", fg=GOOD, bg=BG,
                 font=("Segoe UI", 10)).pack(side="left")
        change = tk.Label(keyrow, text="Change key", fg=MUTED, bg=BG, cursor="hand2",
                          font=("Segoe UI", 10, "underline"))
        change.pack(side="left", padx=12)
        change.bind("<Button-1>", lambda e: self.show_key_screen(can_go_back=True))

        self._label(f, "Use headphones · press M (or Ctrl+Shift+M) to ask a question",
                    9, MUTED, side="bottom")

    def coach_file(self):
        path = filedialog.askopenfilename(title="Choose a TFT video", filetypes=VIDEO_TYPES)
        if path:
            self._start("--file", path)

    def coach_screen(self):
        self._start("--screen")

    def _start(self, *args):
        try:
            launch_mode(*args, "--every", str(self.talk_every.get()))
        except OSError as e:
            messagebox.showerror(APP_NAME, f"Couldn't start the coach: {e}")


def run() -> int:
    root = tk.Tk()
    Launcher(root)
    root.mainloop()
    return 0
