"""Screen mode: pick a window that plays a TFT video, capture it at 1 fps and
show the coach in a small always-on-top caption panel."""
import queue
import tkinter as tk
from tkinter import messagebox

import capture
import safeguard
from common import APP_NAME
from engine import LiveCoach

BG = "#111318"
FG = "#f2f3f5"
MUTED = "#9aa0ab"
ACCENT = "#e8b04a"
PANEL_W, PANEL_H = 460, 170
GUARD_EVERY_TICKS = 5


class ScreenCoach:
    def __init__(self, root: tk.Tk, api_key: str, talk_every: float):
        self.root = root
        self.api_key = api_key
        self.talk_every = talk_every
        self.windows = []
        self.coach = None
        self.hwnd = None
        self.sct = None
        self.ticks = 0
        self.hotkeys = queue.Queue()
        self.frame = None
        self.visible = True
        root.title(f"{APP_NAME} - pick a window")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.stop)
        self.show_picker()

    # ---------- window picker ----------
    def show_picker(self):
        self.root.geometry("640x420")
        f = self.frame = tk.Frame(self.root, bg=BG, padx=16, pady=14)
        f.pack(fill="both", expand=True)
        tk.Label(f, text="Pick the window that plays the TFT video", bg=BG, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(f, text="Browser, YouTube, media player... (League/Riot windows are hidden: videos only)",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 8))
        self.listbox = tk.Listbox(f, bg="#1e2129", fg=FG, selectbackground=ACCENT,
                                  selectforeground="#111", font=("Segoe UI", 10),
                                  relief="flat", activestyle="none")
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self.start())
        self.listbox.bind("<Return>", lambda e: self.start())
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x", pady=(10, 0))
        tk.Button(row, text="Refresh", command=self.refresh, relief="flat",
                  font=("Segoe UI", 10)).pack(side="left")
        tk.Button(row, text="Start coaching", command=self.start, bg=ACCENT, relief="flat",
                  font=("Segoe UI", 10, "bold")).pack(side="right")
        self.refresh()

    def refresh(self):
        pids = capture.own_pids()
        wins = [w for w in capture.list_windows() if w[3] not in pids
                and not w[1].startswith(APP_NAME)]
        self.windows = safeguard.filter_windows(wins)
        self.listbox.delete(0, "end")
        for w in self.windows:
            self.listbox.insert("end", safeguard.label(w))

    def start(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo(APP_NAME, "Pick a window from the list first.")
            return
        if safeguard.game_running():
            messagebox.showwarning(APP_NAME, safeguard.BLOCK_MESSAGE)
            return
        hwnd, title, process, _ = self.windows[sel[0]]
        self.hwnd = hwnd
        self.frame.destroy()
        self.show_panel()
        import mss
        self.sct = mss.mss()
        self.coach = LiveCoach(self.api_key, self.talk_every, kind="screen",
                               source=f"{title} [{process}]")
        self.coach.start()
        self._register_hotkeys()
        self.tick()
        self.refresh_panel()

    # ---------- caption panel ----------
    def show_panel(self):
        r = self.root
        r.title(f"{APP_NAME} LIVE")
        r.attributes("-topmost", True)
        try:
            r.attributes("-alpha", 0.9)
        except tk.TclError:
            pass
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{PANEL_W}x{PANEL_H}+{sw - PANEL_W - 20}+{sh - PANEL_H - 70}")
        f = self.frame = tk.Frame(r, bg=BG, padx=10, pady=8)
        f.pack(fill="both", expand=True)
        self.status_lbl = tk.Label(f, text="Connecting...", bg=BG, fg=ACCENT, anchor="w",
                                   font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(fill="x")
        self.caption_lbl = tk.Label(f, text="", bg=BG, fg=FG, anchor="nw", justify="left",
                                    wraplength=PANEL_W - 24, font=("Segoe UI", 10))
        self.caption_lbl.pack(fill="both", expand=True, pady=(4, 4))
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x")
        self.mic_btn = tk.Button(row, text="Mic", width=8, command=self.toggle_mic, relief="flat")
        self.mic_btn.pack(side="left")
        self.mic_bg = self.mic_btn.cget("bg")
        self.pause_btn = tk.Button(row, text="Pause", width=8, command=self.toggle_pause, relief="flat")
        self.pause_btn.pack(side="left", padx=6)
        tk.Button(row, text="Stop", width=8, command=self.stop, relief="flat").pack(side="left")
        tk.Label(row, text="Ctrl+Shift+M / P / Q", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="right")

    def _register_hotkeys(self):
        try:
            import keyboard
            keyboard.add_hotkey("ctrl+shift+m", lambda: self.hotkeys.put("mic"))
            keyboard.add_hotkey("ctrl+shift+p", lambda: self.hotkeys.put("pause"))
            keyboard.add_hotkey("ctrl+shift+q", lambda: self.hotkeys.put("stop"))
        except Exception as e:  # panel buttons always work
            print("Global hotkeys unavailable:", e)

    def toggle_mic(self):
        if self.coach:
            self.coach.toggle_mic()

    def toggle_pause(self):
        if self.coach:
            self.coach.set_paused(not self.coach.paused)
            self.pause_btn.config(text="Resume" if self.coach.paused else "Pause")

    def tick(self):
        """Once a second: safeguard + capture the chosen window."""
        if self.coach is None:
            return
        self.ticks += 1
        if self.ticks % GUARD_EVERY_TICKS == 0 and safeguard.game_running():
            self.coach.submit_frame(None)
            messagebox.showwarning(APP_NAME, safeguard.BLOCK_MESSAGE)
            self.stop()
            return
        self.visible = False
        if not self.coach.paused:
            try:
                frame = capture.grab_window(self.hwnd, self.sct)
            except Exception as e:
                print("Capture failed:", e)
                frame = None
            self.visible = frame is not None
            self.coach.submit_frame(frame)
        else:
            self.visible = True
        self.root.after(1000, self.tick)

    def refresh_panel(self):
        if self.coach is None:
            return
        while not self.hotkeys.empty():
            action = self.hotkeys.get_nowait()
            if action == "mic":
                self.toggle_mic()
            elif action == "pause":
                self.toggle_pause()
            elif action == "stop":
                self.stop()
                return
        status = self.coach.status_line()
        if not self.visible:
            status = "Window not visible | " + status
        self.status_lbl.config(text=status)
        self.caption_lbl.config(text=self.coach.caption or "Watching the game...")
        self.mic_btn.config(text="Mic ON" if self.coach.mic_on else "Mic",
                            bg=ACCENT if self.coach.mic_on else self.mic_bg)
        self.root.after(200, self.refresh_panel)

    def stop(self):
        coach, self.coach = self.coach, None
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        if coach:
            coach.stop()
            print("Transcript:", coach.transcript_path)
        self.root.destroy()


def run(api_key: str, talk_every: float = 8) -> int:
    capture.set_dpi_aware()
    root = tk.Tk()
    if safeguard.game_running():
        root.withdraw()
        messagebox.showwarning(APP_NAME, safeguard.BLOCK_MESSAGE)
        root.destroy()
        return 2
    ScreenCoach(root, api_key, talk_every)
    root.mainloop()
    return 0
