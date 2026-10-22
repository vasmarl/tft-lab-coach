import json
import time
from types import SimpleNamespace as NS

import numpy as np

import engine
import strategist

ANALYSIS = {
    "gameplay": True, "phase": "planning", "stage": "3-2", "gold": 42, "level": 6, "hp": 61,
    "read": "Jinx 2-star with Guinsoo's, Vi pair on bench.",
    "comp": "Jinx reroll, Vi/Sevika frontline",
    "plan": "Roll to 10 gold now for 3-star Jinx, level at 4-1.",
    "items": "Slam Last Whisper on Jinx; Bramble on Vi.",
    "positioning": "Jinx back-left corner, Vi in front of her.",
    "now": ["Buy both Jinx and roll to 10 gold - you're at 61 HP and 2 copies off 3-star."],
    "insight": "Two players hold Vi, so cap at 2-star and spend on Jinx.",
}


class FakeModels:
    def __init__(self, fail_first=None):
        self.calls = []
        self.fail_first = fail_first

    def generate_content(self, model, contents, config):
        self.calls.append((model, config))
        if self.fail_first and len(self.calls) == 1:
            raise Exception(self.fail_first)
        if config.tools:
            return NS(text="Set 99, patch 99.1. Jinx reroll is S tier.")
        return NS(text=json.dumps(ANALYSIS))


def make(tmp_path, monkeypatch, fail_first=None):
    monkeypatch.setattr(strategist, "app_dir", lambda: tmp_path)
    models = FakeModels(fail_first)
    got = []
    frame = np.zeros((1080, 1920, 3), np.uint8)
    s = strategist.Strategist("k", lambda: (frame, 1), lambda: False, got.append,
                              client=NS(models=models), model="gemini-4-argon")
    return s, models, got


def test_parse_and_notes():
    a = strategist.parse_analysis("```json\n" + json.dumps(ANALYSIS) + "\n```")
    assert a["stage"] == "3-2" and len(a["now"]) == 1
    notes = strategist.notes_text(a)
    assert "Stage 3-2 | Gold 42 | Level 6 | HP 61" in notes and "Do now: Buy both Jinx" in notes
    assert strategist.parse_analysis('{"gameplay": false}') is None
    assert strategist.parse_analysis("nope") is None


def test_analysis_uses_meta_and_hi_res_frame(tmp_path, monkeypatch):
    s, models, got = make(tmp_path, monkeypatch)
    s.meta = s.research_meta()
    assert "Jinx reroll" in s.meta
    assert (tmp_path / f"tft_meta_{time.strftime('%Y-%m-%d')}.md").exists()
    a = s.analyse(np.zeros((1080, 1920, 3), np.uint8))
    assert a["comp"].startswith("Jinx")
    model, cfg = models.calls[-1]
    assert model == "gemini-4-argon" and cfg.response_mime_type == "application/json"


def test_falls_back_when_model_unavailable(tmp_path, monkeypatch):
    s, models, got = make(tmp_path, monkeypatch, fail_first="404 NOT_FOUND models/gemini-4-argon")
    a = s.analyse(np.zeros((720, 1280, 3), np.uint8))
    assert a and s.model == strategist.FALLBACK_MODEL
    assert models.calls[-1][0] == strategist.FALLBACK_MODEL


def test_live_prompt_carries_analyst_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "app_dir", lambda: tmp_path)
    coach = engine.LiveCoach("k")
    assert coach.advice_prompt() == engine.COMMENT_PROMPT
    coach._on_analysis(strategist.parse_analysis(json.dumps(ANALYSIS)))
    prompt = coach.advice_prompt()
    assert "ANALYST NOTES" in prompt and "Roll to 10 gold now" in prompt
