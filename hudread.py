"""Exact numbers from the screen (OCR), so the analyst never misreads gold or HP.

RapidOCR (ONNX, offline, bundled in the exe) reads every piece of text on the
frame with its position. Stage, level and XP are parsed deterministically; the
full list of readings (with positions) goes to the analyst, which only has to
decide which number is which (gold above the shop, HP in the player list...).
"""
import re
import threading

import cv2
import numpy as np

OCR_WIDTH = 1920
MAX_READINGS = 70

_ocr = None
_ocr_lock = threading.Lock()


def _engine():
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        return _ocr


def read(frame, region=None) -> list:
    """[(text, confidence, x, y)] with x, y = text centre as a 0-1 fraction of the frame.

    region: optional (x0, y0, x1, y1) fractions to read only part of the frame.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region or (0, 0, 1, 1)
    crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if crop.size == 0:
        return []
    scale = 1.0
    if crop.shape[1] > OCR_WIDTH:
        scale = OCR_WIDTH / crop.shape[1]
        crop = cv2.resize(crop, (OCR_WIDTH, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    result, _ = _engine()(crop)
    out = []
    for box, text, conf in result or []:
        pts = np.asarray(box, dtype=float) / scale
        cx = x0 + pts[:, 0].mean() / w
        cy = y0 + pts[:, 1].mean() / h
        out.append((str(text).strip(), float(conf), round(cx, 3), round(cy, 3)))
    return out


STAGE_RE = re.compile(r"^([1-7])\s*[-–]\s*([1-7])$")
LEVEL_RE = re.compile(r"(?:lv|lvl|level)\.?\s*(\d{1,2})", re.I)
XP_RE = re.compile(r"^(\d{1,3})\s*/\s*(\d{1,3})$")


def parse(readings: list) -> dict:
    """Facts that can be parsed without guessing."""
    facts = {}
    for text, conf, x, y in readings:
        if conf < 0.6:
            continue
        t = text.replace(" ", "")
        m = STAGE_RE.match(t)
        if m and y < 0.12 and "stage" not in facts:
            facts["stage"] = f"{m.group(1)}-{m.group(2)}"
            continue
        m = LEVEL_RE.search(text)
        if m and y > 0.6 and "level" not in facts:
            lvl = int(m.group(1))
            if 1 <= lvl <= 10:
                facts["level"] = lvl
            continue
        m = XP_RE.match(t)
        if m and y > 0.6 and "xp" not in facts:
            facts["xp"] = f"{m.group(1)}/{m.group(2)}"
    return facts


def read_stage(frame):
    """Fast stage read from the top strip only (used to detect a new round)."""
    try:
        return parse(read(frame, (0.25, 0.0, 0.75, 0.1))).get("stage")
    except Exception:
        return None


def prompt_text(readings: list, facts: dict) -> str:
    """Readings for the analyst: exact text + where it is on screen."""
    if not readings:
        return ""
    keep = [r for r in readings if r[1] >= 0.5]
    # numbers first: they're what the analyst needs exactly
    keep.sort(key=lambda r: (not any(ch.isdigit() for ch in r[0]), r[3], r[2]))
    keep = keep[:MAX_READINGS]
    lines = ["SCREEN TEXT READ BY OCR (exact characters; x,y = position from top-left, 0-1). "
             "Use these digits for gold, HP, level, XP and stage instead of estimating; "
             "gold is the number just above the shop at the bottom centre, the player list with HP is on the right."]
    if facts:
        lines.append("Parsed: " + ", ".join(f"{k}={v}" for k, v in facts.items()))
    lines.append("; ".join(f"'{t}'@({x:.2f},{y:.2f})" for t, _, x, y in keep))
    return "\n".join(lines)
