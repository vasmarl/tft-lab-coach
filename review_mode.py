"""Review mode (command line only): cheaper, non-live.

Plays the video, sends one frame every 5 s of video to gemini-2.5-flash for a
short JSON tip, shows + speaks it, then writes an end-of-game review.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2

from common import app_dir, fmt_time, review_model, stamp
from engine import encode_frame
from file_mode import VideoClock
from overlay import draw_bar, fit, window_closed

WINDOW = "TFT Lab Coach REVIEW"
SAMPLE_EVERY = 5.0

TIP_PROMPT = """You are a Challenger-level Teamfight Tactics coach reviewing a frame of a recorded TFT game.
Reply with JSON only:
{"skip": true|false, "stage": "e.g. 3-2 or unknown", "state": "very short board/economy summary", "tip": "one concrete coaching tip, max 25 words"}
Set "skip" to true if the frame is not TFT gameplay (menus, loading, webcam only, ads) or if there is nothing new worth saying.
Previous tips (do not repeat them):
"""

REVIEW_PROMPT = """You are a Challenger-level Teamfight Tactics coach. Below are timestamped notes you made while watching a recorded TFT game.
Write an end-of-game review in Markdown with exactly these sections:
## What went well
(3 bullets)
## Biggest mistakes
(3 bullets, each naming the stage, e.g. "Stage 3-2: ...")
## Focus for next game
(3 bullets)
Be concrete and refer to the notes. Notes:
"""


def parse_tip(text: str):
    """Parse the model's JSON tip; returns dict or None."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("skip"):
        return None
    tip = " ".join(str(data.get("tip", "")).split())
    if not tip:
        return None
    words = tip.split()
    if len(words) > 25:
        tip = " ".join(words[:25])
    return {"stage": str(data.get("stage", "?")), "state": str(data.get("state", "")), "tip": tip}


class Speaker:
    """Speaks tips one at a time in the background (Windows System.Speech)."""

    def __init__(self):
        self.q = queue.Queue()
        self.proc = None
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text: str):
        self.q.put(text)

    def _worker(self):
        engine = None
        while True:
            text = self.q.get()
            if text is None:
                return
            try:
                if sys.platform == "win32":
                    safe = text.replace("'", "''")
                    ps = ("Add-Type -AssemblyName System.Speech; "
                          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                          f"$s.Rate = 1; $s.Speak('{safe}')")
                    self.proc = subprocess.Popen(
                        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    self.proc.wait()
                elif sys.platform == "darwin":
                    self.proc = subprocess.Popen(["say", text])
                    self.proc.wait()
                else:
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print("Speech failed:", e)

    def stop(self):
        self.q.put(None)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def run(path: str, api_key: str, client=None) -> int:
    from google import genai
    from google.genai import types

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0

    client = client or genai.Client(api_key=api_key)
    model = review_model()
    notes = []           # (video_seconds, tip dict)
    caption = {"text": "Watching... first tip in a few seconds.", "status": "Review"}
    busy = threading.Event()
    speaker = Speaker()
    log_path = app_dir() / f"coach_log_{stamp()}.md"

    def ask(jpeg: bytes, at: float):
        try:
            previous = "\n".join(f"- {n['tip']}" for _, n in notes[-6:]) or "(none)"
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                          TIP_PROMPT + previous],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   temperature=0.4))
            tip = parse_tip(resp.text)
            if tip:
                notes.append((at, tip))
                caption["text"] = f"[{tip['stage']}] {tip['tip']}"
                speaker.say(tip["tip"])
            caption["status"] = "Review"
        except Exception as e:
            print("Tip request failed:", repr(e))
            caption["status"] = "Review (tip failed, retrying)"
        finally:
            busy.clear()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    clock = VideoClock()
    frame_idx, shown, paused, sized = 0, None, False, False
    next_sample = 1.0
    try:
        while True:
            if not paused:
                target = int(clock.elapsed() * fps)
                while frame_idx < target - 1:
                    if not cap.grab():
                        break
                    frame_idx += 1
                if frame_idx <= target:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    shown = fit(frame)
                    if not sized:
                        cv2.resizeWindow(WINDOW, shown.shape[1], shown.shape[0])
                        sized = True
                    at = frame_idx / fps
                    if at >= next_sample and not busy.is_set():
                        busy.set()
                        next_sample = at + SAMPLE_EVERY
                        jpeg = encode_frame(frame)
                        threading.Thread(target=ask, args=(jpeg, at), daemon=True).start()
            if shown is not None:
                status = f"{fmt_time(frame_idx / fps)} | {'PAUSED' if paused else caption['status']} | {len(notes)} tips | space pause  q quit"
                cv2.imshow(WINDOW, draw_bar(shown, status, caption["text"]))
            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or window_closed(WINDOW):
                break
            if key == ord(" "):
                paused = not paused
                clock.pause(paused)
    finally:
        cap.release()
        speaker.stop()

    # final review
    if shown is not None:
        caption["text"] = "Writing your end-of-game review..."
        cv2.imshow(WINDOW, draw_bar(shown, "Review", caption["text"]))
        cv2.waitKey(1)
    deadline = time.monotonic() + 20
    while busy.is_set() and time.monotonic() < deadline:
        time.sleep(0.1)
    review = write_review(client, model, notes, path, log_path)
    cv2.destroyAllWindows()
    print("Review saved:", log_path)
    return 0 if review else 1


def write_review(client, model, notes, video_path, log_path) -> str:
    note_lines = "\n".join(f"- [{fmt_time(at)}] stage {n['stage']}: {n['state']} -> {n['tip']}"
                           for at, n in notes)
    review = ""
    if notes:
        try:
            review = client.models.generate_content(model=model, contents=REVIEW_PROMPT + note_lines).text
        except Exception as e:
            review = f"_Couldn't generate the review: {e}_"
    else:
        review = "_No tips were collected (video too short, or not TFT gameplay?)._"
    text = (f"# TFT Lab Coach - Round-by-round review\n\n"
            f"- Date: {datetime.now():%Y-%m-%d %H:%M}\n- Video: {os.path.basename(video_path)}\n"
            f"- Model: {model}\n\n## Tips during the game\n\n{note_lines or '(none)'}\n\n"
            f"# End-of-game review\n\n{(review or '').strip()}\n")
    log_path.write_text(text, encoding="utf-8")
    return review
