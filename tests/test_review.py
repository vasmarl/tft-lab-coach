from types import SimpleNamespace as NS

import numpy as np

import overlay
import review_mode


def test_parse_tip():
    tip = review_mode.parse_tip('{"skip": false, "stage": "3-2", "state": "40 gold", "tip": "Level to 7 now."}')
    assert tip == {"stage": "3-2", "state": "40 gold", "tip": "Level to 7 now."}
    assert review_mode.parse_tip('{"skip": true, "tip": "x"}') is None
    assert review_mode.parse_tip("not json") is None
    long = " ".join(["word"] * 40)
    assert len(review_mode.parse_tip('{"tip": "%s"}' % long)["tip"].split()) == 25


def test_write_review(tmp_path):
    client = NS(models=NS(generate_content=lambda model, contents: NS(text="## What went well\n- a")))
    notes = [(65.0, {"stage": "2-1", "state": "s", "tip": "Slam bow."})]
    out = tmp_path / "coach_log.md"
    review_mode.write_review(client, "m", notes, "game.mp4", out)
    text = out.read_text()
    assert "[01:05] stage 2-1" in text and "## What went well" in text


def test_draw_bar_keeps_size():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = overlay.draw_bar(frame, "00:10 | Live | mic off", "word " * 80)
    assert out.shape == frame.shape
    assert out[-5:].mean() > 0   # bar drawn at the bottom
