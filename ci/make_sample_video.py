"""CI helper: write a short synthetic 'game' video (moving shapes + a fake HUD)."""
import sys

import cv2
import numpy as np

out = sys.argv[1]
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
w, h, fps = 1280, 720, 30
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(seconds * fps):
    t = i / fps
    f = np.full((h, w, 3), (40, 60, 30), np.uint8)
    for k in range(7):  # "units" on a hex board
        x = int(300 + k * 110 + 20 * np.sin(t + k))
        y = int(380 + 30 * np.cos(t * 1.3 + k))
        cv2.circle(f, (x, y), 34, (60 + 25 * k, 120, 220 - 20 * k), -1)
    cv2.putText(f, f"Stage 3-{1 + int(t) // 10}   Gold {30 + int(t)}   Level 6   HP 54",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)
    vw.write(f)
vw.release()
print("wrote", out)
