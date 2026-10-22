"""Analyst: the 'brain' behind the live voice coach.

The live audio model is fast but shallow. This thread runs a strong reasoning
model on a high-resolution frame at the moments that matter (new round, shop
or bench change, augment/carousel screen; at least every MAX_GAP s), keeps a
running game plan (comp, carry, items, econ/level/roll plan, positioning, next
actions) and hands it to the live coach, which turns it into spoken orders.

The model gets facts instead of guessing: exact digits read by OCR, the
current set's champions/traits/recipes, and roll odds computed in code.

At session start it also looks up the current set/patch meta with Google
Search, so the advice is about real comps of the current patch.
"""
import json
import threading
import time
from datetime import date

import cv2
import numpy as np

import hudread
import rollodds
import setdata
from common import app_dir, strategy_model

FALLBACK_MODEL = "gemini-3.8-flash"
MIN_GAP = 4.0              # never analyse more often than this
MAX_GAP = 12.0             # analyse at least this often even if nothing obvious changed
CHECK_EVERY = 1.0          # how often to look for a key moment
STAGE_EVERY = 2.0          # how often to OCR the stage (top strip)
FULL_CHANGE = 28.0         # mean pixel change of the whole screen = new screen (augment, carousel)
BOTTOM_CHANGE = 12.0       # change in the shop/bench strip = shop refresh, buy/sell
FRAME_WIDTH = 1600
JPEG_QUALITY = 90

META_PROMPT = """Use Google Search to find the CURRENT Teamfight Tactics set and patch (today is {today}) and write a compact Challenger cheat sheet for it, max 450 words, plain text:
1. Set name and patch number.
2. The 6-8 strongest comps right now: name, carry (+ its best 3 items), core units, key traits, when to roll (level/stage), and the typical opener.
3. Strongest augments / items / economy patterns this patch, and what is currently overrated.
Be specific (unit, trait and item names). No intro text.
4. Only if official patch notes for this set changed the shop odds or champion pool sizes, end with exactly one line:
ROLL_TABLE {{"shop_odds": {{"4": [55,30,15,0,0], "5": [...], ...}}, "pool_copies": {{"1": 30, "2": 25, ...}}}}
Otherwise leave that line out."""

ANALYST_PROMPT = """You are a top Challenger Teamfight Tactics player (1000+ LP) analysing one screenshot of a RECORDED TFT game to decide the player's next moves. The player is the one whose board is shown.

Read EVERYTHING visible: stage-round, gold, level and XP bar, the player's HP and every opponent's HP (right-side list), win/loss streak, the 5 shop units (cost, pairs with the bench/board), bench, board units with star level and items, item components, the traits panel on the left (active counts and what's 1 unit away), chosen augments, augment/carousel choices on screen.

Then think like a Challenger, not a beginner. Never give generic advice ("manage your economy", "position well", "save interest"); the player knows the basics. Give decisions specific to THIS board:
- the comp to play: the best final board from these units/items/augments and the current meta; whether to commit or pivot, and why
- item plan: exact completed items from these components, who holds them now and who is the final carry; slam now or hold, judged by HP and streak
- econ / level / roll plan for the next 2-3 rounds with concrete numbers (e.g. "level to 7 at 4-1, roll to 10 gold for 2-star Jinx, keep the Vi pair")
- the shop right now: exactly what to buy, what to sell, which pairs to keep
- positioning: concrete placements (carry corner, which side the frontline goes, counter-position against the strongest opponent you can see)
- risks: HP danger, contested units, a strong opponent to play around
On augment or carousel screens, name the exact pick and why.

TFT rules you must respect (never recite them): +1 gold interest per 10 banked up to +5 at 50; 4 gold buys 4 XP; a reroll costs 2 gold; board units can't exceed level; 3 copies = 2-star, 3 two-stars = 3-star; 2 components = 1 item; max 3 items per unit; bench holds 9.
Only suggest actions the player can afford with the gold you can see. If a number is unreadable, write "unclear"; never invent it.

{setdata}{ocr}{meta}{previous}{why}
If the player is rolling or about to roll for a specific unit, fill "roll_target" and the app will compute the exact hit chance; use the "Roll math" from your previous analysis when deciding roll vs level.
Reply with JSON only:
{{"gameplay": true|false,
 "phase": "planning|combat|augment|carousel|other",
 "stage": "e.g. 3-2", "gold": number or null, "level": number or null, "hp": number or null,
 "read": "what you see that matters, 1-2 sentences",
 "comp": "target final board + main carry",
 "plan": "econ/level/roll plan for the next 2-3 rounds with numbers",
 "items": "item plan",
 "positioning": "concrete placement",
 "now": ["1-3 orders for right now, most important first, each with a short reason"],
 "roll_target": null or {{"unit": "name", "cost": 1-5, "level": current level, "need": copies still needed for the next star level, "gold": gold the player would roll, "copies_gone": copies of that unit you can see on any board or bench}},
 "insight": "one non-obvious Challenger-level insight about this exact spot"}}"""


def encode_hi(bgr) -> bytes:
    h, w = bgr.shape[:2]
    if w > FRAME_WIDTH:
        bgr = cv2.resize(bgr, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def parse_analysis(text: str):
    text = (text or "").strip()
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("gameplay", True):
        return None
    now = data.get("now") or []
    if isinstance(now, str):
        now = [now]
    data["now"] = [str(x).strip() for x in now if str(x).strip()][:3]
    return data


def notes_text(a: dict) -> str:
    """Compact notes for the live voice model."""
    parts = []
    for key, label in (("stage", "Stage"), ("gold", "Gold"), ("level", "Level"), ("hp", "HP")):
        if a.get(key) not in (None, "", "unclear"):
            parts.append(f"{label} {a[key]}")
    lines = [" | ".join(parts)] if parts else []
    for key, label in (("read", "Read"), ("comp", "Comp"), ("plan", "Plan"), ("items", "Items"),
                       ("positioning", "Positioning"), ("insight", "Insight")):
        if a.get(key):
            lines.append(f"{label}: {a[key]}")
    if a.get("roll_math"):
        lines.append(f"Roll math (computed): {a['roll_math']}")
    if a.get("now"):
        lines.append("Do now: " + " / ".join(a["now"]))
    return "\n".join(lines)


def is_retryable_model_error(exc) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("429", "resource_exhausted", "quota", "404", "not found",
                                   "not supported", "permission", "403"))


class Strategist:
    def __init__(self, api_key: str, get_frame, is_paused, on_analysis, client=None, model=None):
        self.api_key = api_key
        self.get_frame = get_frame          # -> (frame, seq)
        self.is_paused = is_paused
        self.on_analysis = on_analysis      # callback(dict)
        self.model = model or strategy_model()
        self.meta = ""
        self.last = None
        self._client = client
        self._stop = threading.Event()
        self._thread = None
        self.calls = 0
        self.failures = 0
        self.set_data = setdata.load()
        self.set_text = setdata.prompt_text(self.set_data)
        self.roll_info = {"units_per_cost": self.set_data.get("units_per_cost") or {}}
        self.ocr_ok = True

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---------- model calls ----------
    def _generate(self, contents, config: dict, model=None):
        from google.genai import types
        try:
            return self._client.models.generate_content(
                model=model or self.model, contents=contents,
                config=types.GenerateContentConfig(**config))
        except Exception as e:
            text = str(e).lower()
            # an option the model doesn't accept: retry once without the extras
            if "invalid" in text and ("thinking" in text or "media_resolution" in text or "argument" in text):
                plain = {k: v for k, v in config.items() if k not in ("thinking_config", "media_resolution")}
                if plain != config:
                    return self._client.models.generate_content(
                        model=model or self.model, contents=contents,
                        config=types.GenerateContentConfig(**plain))
            raise

    def _call(self, contents, config: dict):
        try:
            return self._generate(contents, config)
        except Exception as e:
            if self.model != FALLBACK_MODEL and is_retryable_model_error(e):
                print(f"Analyst model {self.model} unavailable ({str(e)[:120]}); using {FALLBACK_MODEL}")
                self.model = FALLBACK_MODEL
                return self._generate(contents, config)
            raise

    def research_meta(self) -> str:
        cache = app_dir() / f"tft_meta_{date.today():%Y-%m-%d}.md"
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        from google.genai import types
        resp = self._call(META_PROMPT.format(today=f"{date.today():%Y-%m-%d}"),
                          {"tools": [types.Tool(google_search=types.GoogleSearch())]})
        meta = (resp.text or "").strip()
        self._apply_roll_table(meta)
        if meta:
            try:
                cache.write_text(meta, encoding="utf-8")
            except OSError:
                pass
        return meta

    def _apply_roll_table(self, meta: str):
        for line in meta.splitlines():
            if line.strip().startswith("ROLL_TABLE"):
                try:
                    table = json.loads(line.strip()[len("ROLL_TABLE"):].strip())
                    odds = {int(k): [float(x) for x in v][:5] for k, v in (table.get("shop_odds") or {}).items()}
                    pool = {int(k): int(v) for k, v in (table.get("pool_copies") or {}).items()}
                    if odds and all(len(v) == 5 for v in odds.values()):
                        self.roll_info["shop_odds"] = {**rollodds.DEFAULT_SHOP_ODDS, **odds}
                    if len(pool) == 5:
                        self.roll_info["pool_copies"] = pool
                    print("Roll table from patch notes:", table)
                except (ValueError, TypeError, AttributeError) as e:
                    print("Bad ROLL_TABLE line:", e)

    def read_screen(self, frame):
        if not self.ocr_ok:
            return [], {}
        try:
            readings = hudread.read(frame)
            return readings, hudread.parse(readings)
        except Exception as e:
            print("OCR unavailable, continuing without it:", repr(e))
            self.ocr_ok = False
            return [], {}

    def analyse(self, frame, why: str = "") -> dict | None:
        from google.genai import types
        meta = f"CURRENT META (from a web search today):\n{self.meta}\n\n" if self.meta else ""
        previous = ""
        if self.last:
            previous = ("YOUR PREVIOUS ANALYSIS (keep the plan consistent unless the game changed; "
                        f"don't repeat orders the player already followed):\n{notes_text(self.last)}\n\n")
        readings, facts = self.read_screen(frame)
        prompt = ANALYST_PROMPT.format(
            meta=meta, previous=previous,
            setdata=f"{self.set_text}\n\n" if self.set_text else "",
            ocr=f"{hudread.prompt_text(readings, facts)}\n\n" if readings else "",
            why=f"WHY NOW: {why}\n" if why else "")
        resp = self._call(
            [types.Part.from_bytes(data=encode_hi(frame), mime_type="image/jpeg"), prompt],
            {"response_mime_type": "application/json",
             "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
             "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
             "temperature": 0.3})
        analysis = parse_analysis(resp.text)
        if analysis:
            for key in ("stage", "level"):
                if facts.get(key) is not None:
                    analysis[key] = facts[key]       # OCR beats the model on digits
            analysis["why"] = why
            if isinstance(analysis.get("roll_target"), dict):
                target = dict(analysis["roll_target"])
                if facts.get("level"):
                    target["level"] = facts["level"]
                analysis["roll_math"] = rollodds.summary(target, self.roll_info)
        return analysis

    # ---------- loop ----------
    def _run(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        try:
            self.meta = self.research_meta()
            print("Meta notes loaded:", len(self.meta), "chars")
        except Exception as e:
            print("Meta lookup failed:", repr(e))
        last_t = -1e9
        last_sig = None
        last_stage = None
        last_stage_check = 0.0
        while not self._stop.is_set():
            frame, seq = self.get_frame()
            if frame is None or self.is_paused():
                self._stop.wait(CHECK_EVERY)
                continue
            now = time.monotonic()
            why = ""
            if self.ocr_ok and now - last_stage_check >= STAGE_EVERY:
                last_stage_check = now
                stage = hudread.read_stage(frame)
                if stage and stage != last_stage:
                    why = f"new round {stage}" if last_stage else f"round {stage}"
                    last_stage = stage
            sig = signature(frame)
            if last_sig is not None and not why:
                full, bottom = change(last_sig, sig)
                if full > FULL_CHANGE:
                    why = "the screen changed (augment, carousel or a new phase)"
                elif bottom > BOTTOM_CHANGE:
                    why = "the shop or bench changed"
            gap = now - last_t
            if (why and gap >= MIN_GAP) or gap >= MAX_GAP:
                last_t, last_sig = now, sig
                try:
                    analysis = self.analyse(frame, why)
                    self.calls += 1
                    if analysis:
                        self.last = analysis
                        self.on_analysis(analysis)
                except Exception as e:
                    self.failures += 1
                    print("Analysis failed:", repr(e)[:300])
            self._stop.wait(CHECK_EVERY)


def signature(frame):
    """Tiny grayscale thumbnails of the whole screen and of the shop/bench strip."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    full = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    bottom = cv2.resize(gray[int(h * 0.72):], (96, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    return full, bottom


def change(a, b):
    return float(np.abs(a[0] - b[0]).mean()), float(np.abs(a[1] - b[1]).mean())
