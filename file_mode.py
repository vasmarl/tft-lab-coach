"""File mode: play a video in an OpenCV window and coach it live."""
import os
import time

import cv2

from common import fmt_time
from engine import LiveCoach
from overlay import draw_bar, fit, window_closed

WINDOW = "TFT Lab Coach LIVE"
END_GRACE = 1.5


class VideoClock:
    """Real-time position in the video that stops while paused."""

    def __init__(self):
        self.start = time.monotonic()
        self.paused_at = None
        self.paused_total = 0.0

    def pause(self, on: bool):
        now = time.monotonic()
        if on and self.paused_at is None:
            self.paused_at = now
        elif not on and self.paused_at is not None:
            self.paused_total += now - self.paused_at
            self.paused_at = None

    def elapsed(self) -> float:
        now = self.paused_at or time.monotonic()
        return now - self.start - self.paused_total


def run(path: str, api_key: str, talk_every: float = 8) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if total else 0

    coach = LiveCoach(api_key, talk_every, kind="live", source=os.path.basename(path))
    coach.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    clock = VideoClock()
    frame_idx = 0
    shown = None
    paused = False
    ended_at = None

    try:
        while True:
            if not paused and ended_at is None:
                target = int(clock.elapsed() * fps)
                frame = None
                # behind schedule: grab (decode-free) frames we won't draw
                while frame_idx < target - 1:
                    if not cap.grab():
                        break
                    frame_idx += 1
                if frame_idx <= target:
                    ok, frame = cap.read()
                    if not ok:
                        ended_at = time.monotonic()
                    else:
                        frame_idx += 1
                        shown = fit(frame)
                        coach.submit_frame(frame)

            if shown is not None:
                t = fmt_time(frame_idx / fps)
                if duration:
                    t += " / " + fmt_time(duration)
                state = "END" if ended_at else coach.status_line()
                status = f"{t} | {state} | space pause  m mic  q quit"
                cv2.imshow(WINDOW, draw_bar(shown, status, coach.caption))

            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or window_closed(WINDOW):
                break
            if key == ord(" "):
                paused = not paused
                clock.pause(paused)
                coach.set_paused(paused)
            elif key in (ord("m"), ord("M")):
                coach.toggle_mic()
            if ended_at:
                # let the last sentence finish
                if time.monotonic() - ended_at > END_GRACE and not coach.speaking:
                    break
                if time.monotonic() - ended_at > END_GRACE + 8:
                    break
    finally:
        cap.release()
        coach.stop()
        cv2.destroyAllWindows()
        print("Transcript:", coach.transcript_path)
    return 0
