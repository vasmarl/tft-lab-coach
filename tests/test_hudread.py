import cv2
import numpy as np
import pytest

import hudread

pytest.importorskip("rapidocr_onnxruntime")


def fake_hud():
    img = np.full((1080, 1920, 3), (40, 60, 30), np.uint8)
    put = lambda t, x, y, s=1.0: cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2)
    put("3-2", 900, 40)
    put("Lvl. 6", 300, 880, 0.9)
    put("12/36", 300, 930, 0.8)
    put("47", 950, 880, 1.2)
    put("61", 1800, 400, 0.8)
    return img


def test_reads_exact_digits_and_parses_hud():
    readings = hudread.read(fake_hud())
    texts = {t for t, *_ in readings}
    assert {"3-2", "47", "61"} <= texts
    facts = hudread.parse(readings)
    assert facts["stage"] == "3-2" and facts["level"] == 6 and facts["xp"] == "12/36"
    text = hudread.prompt_text(readings, facts)
    assert "'47'@(0.51," in text and "stage=3-2" in text


def test_read_stage_from_top_strip():
    assert hudread.read_stage(fake_hud()) == "3-2"
