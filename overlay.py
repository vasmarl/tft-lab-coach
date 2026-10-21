"""OpenCV drawing helpers shared by file mode and review mode."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_W, MAX_H = 1280, 720


def fit(frame: np.ndarray) -> np.ndarray:
    """Scale a frame down so the window fits on a laptop screen."""
    h, w = frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def wrap(text: str, width_px: int, scale: float, thickness: int = 1) -> list:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_bar(frame: np.ndarray, status: str, caption: str, max_lines: int = 3) -> np.ndarray:
    """Dark semi-transparent bar at the bottom: status line + wrapped caption."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.45, w / 1800)
    line_h = int(32 * scale / 0.6)
    pad = 12
    lines = wrap(caption, w - 2 * pad, scale)[-max_lines:]
    bar_h = pad * 2 + line_h * (1 + max(1, len(lines)))
    y0 = h - bar_h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    y = y0 + pad + line_h - 6
    cv2.putText(out, status, (pad, y), FONT, scale * 0.85, (120, 220, 255), 1, cv2.LINE_AA)
    for ln in lines:
        y += line_h
        cv2.putText(out, ln, (pad, y), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def window_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
