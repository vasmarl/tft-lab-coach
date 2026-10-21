"""CI helper: real Gemini Live round trip (needs GOOGLE_API_KEY).

Streams frames of a video to the live model for ~40 s and checks that the
coach answered with transcribed speech. Exit 0 = ok, 1 = no answer.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import engine  # noqa: E402

key = os.environ.get("GOOGLE_API_KEY", "").strip()
if not key:
    print("No GOOGLE_API_KEY secret: skipping the live check.")
    sys.exit(0)

video = sys.argv[1]
out_dir = Path(sys.argv[2])
engine.app_dir = lambda: out_dir

coach = engine.LiveCoach(key, talk_every=5, kind="live", source=Path(video).name)
coach.start()
cap = cv2.VideoCapture(video)
start = time.monotonic()
while time.monotonic() - start < 40:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    coach.submit_frame(frame)
    time.sleep(1 / 30)
    if int(time.monotonic() - start) % 5 == 0:
        print(f"{time.monotonic() - start:5.1f}s  {coach.status_line()}  frames={coach.frames_sent}  "
              f"caption={coach.caption[-80:]!r}", flush=True)
coach.stop()
log = coach.transcript_path.read_text(encoding="utf-8")
print(log)
if "**Coach:**" not in log:
    print("FAIL: the live model never answered. Error:", coach.error)
    sys.exit(1)
print("Live check OK, model:", coach.model)
