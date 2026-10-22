"""TFT Lab Coach - single entry point.

  TFTCoach.exe                   launcher
  TFTCoach.exe --file VIDEO      live coaching on a video file
  TFTCoach.exe --screen          live coaching on a window on screen
  TFTCoach.exe --review VIDEO    cheaper round-by-round review (gemini-3.8-flash)
  TFTCoach.exe --selftest        import everything, write selftest_ok.txt, exit 0
Options: --every SECONDS (talk interval, 4-20), --model NAME (live model)
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

from common import APP_NAME, DEFAULT_TALK_EVERY, app_dir

APP_MODULES = ["common", "keystore", "engine", "overlay", "file_mode", "launcher",
               "safeguard", "capture", "screen_mode", "review_mode", "strategist", "hudread", "setdata", "rollodds"]


def setup_output():
    """A windowed exe has no console: send print() to a log next to the exe."""
    if sys.stdout is None or sys.stderr is None or getattr(sys, "frozen", False):
        try:
            log = open(app_dir() / "tft_coach_output.log", "a", encoding="utf-8", buffering=1)
        except OSError:
            log = open(os.devnull, "w")
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv[1:])}\n")
        sys.stdout = log
        sys.stderr = log


def report_crash(exc: BaseException, show_box: bool = True):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        with open(app_dir() / "tft_coach_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv[1:])}\n{tb}")
    except OSError:
        pass
    print(tb)
    if show_box:
        last = tb.strip().splitlines()[-1] if tb.strip() else repr(exc)
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(APP_NAME, f"Something went wrong:\n\n{last}\n\n"
                                           f"Details: tft_coach_error.log next to the app.")
            root.destroy()
        except Exception:
            pass


def selftest() -> int:
    import importlib
    deps = ["cv2", "numpy", "sounddevice", "mss", "psutil", "dotenv", "google.genai", "onnxruntime",
            "rapidocr_onnxruntime",
            "google.genai.types", "tkinter", "keyboard"]
    if sys.platform == "win32":
        deps += ["win32gui", "win32process", "win32api"]
    loaded = []
    for name in deps + APP_MODULES:
        importlib.import_module(name)
        loaded.append(name)
    # exercise a few things that need native libraries
    import numpy as np
    import sounddevice as sd
    from engine import encode_frame
    encode_frame(np.zeros((90, 160, 3), np.uint8))
    portaudio = sd.get_portaudio_version()[1]
    # OCR models load and read digits; set data is bundled; roll math runs
    import cv2
    import hudread
    import rollodds
    import setdata
    img = np.full((120, 400, 3), 30, np.uint8)
    cv2.putText(img, "3-2  47", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    ocr_text = " ".join(t for t, *_ in hudread.read(img))
    if "47" not in ocr_text:
        raise RuntimeError(f"OCR self-check failed: {ocr_text!r}")
    data = setdata.load()
    if not data.get("champions"):
        raise RuntimeError("Set data missing from the bundle")
    rollodds.hit_chance(6, cost=2, need=1, gold=20, units_of_cost=13, trials=200)
    loaded.append(f"ocr={ocr_text!r} set={data['set']['number']} champions={len(data['champions'])}")
    from google.genai import types
    types.LiveConnectConfig(response_modalities=["AUDIO"],
                            context_window_compression=types.ContextWindowCompressionConfig(
                                sliding_window=types.SlidingWindow()))
    (app_dir() / "selftest_ok.txt").write_text(
        f"OK {datetime.now():%Y-%m-%d %H:%M:%S}\n{portaudio}\n" + "\n".join(loaded) + "\n",
        encoding="utf-8")
    print("selftest ok")
    return 0


def need_key() -> str:
    import keystore
    key = keystore.load_key()
    if not key:
        raise SystemExit("No Google AI key yet. Open TFTCoach.exe without arguments to connect one.")
    return key


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="TFTCoach", description=APP_NAME)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", metavar="VIDEO")
    g.add_argument("--screen", action="store_true")
    g.add_argument("--review", metavar="VIDEO")
    g.add_argument("--selftest", action="store_true")
    p.add_argument("--every", type=float, default=DEFAULT_TALK_EVERY)
    p.add_argument("--model")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()

    import keystore
    keystore.load_env()
    if args.model:
        os.environ["TFT_LIVE_MODEL"] = args.model

    if args.file:
        import file_mode
        return file_mode.run(args.file, need_key(), args.every)
    if args.screen:
        import screen_mode
        return screen_mode.run(need_key(), args.every)
    if args.review:
        import review_mode
        return review_mode.run(args.review, need_key())
    import launcher
    return launcher.run()


if __name__ == "__main__":
    setup_output()
    selftest_mode = "--selftest" in sys.argv
    try:
        code = main()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
            if not selftest_mode:
                report_crash(RuntimeError(e.code))
            code = 1
        else:
            code = e.code or 0
    except BaseException as e:
        report_crash(e, show_box=not selftest_mode)
        code = 1
    sys.exit(code)
