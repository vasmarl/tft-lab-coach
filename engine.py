"""Live coach engine: one Gemini Live session in a background asyncio thread.

The UI thread (OpenCV loop or tkinter) talks to it through a few thread-safe
methods: submit_frame(), toggle_mic(), set_paused(), stop(), and reads
status / caption / mic_on.
"""
import asyncio
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from common import app_dir, fmt_time, live_model, stamp

OUT_RATE = 24000      # model audio: 16-bit mono PCM @ 24 kHz
IN_RATE = 16000       # mic audio:   16-bit mono PCM @ 16 kHz
MIC_BLOCK = 1600      # 100 ms of mic audio per chunk
FRAME_INTERVAL = 1.0  # max 1 fps to the model
FRAME_WIDTH = 1024
JPEG_QUALITY = 80
STUCK_TURN_SECONDS = 25
FIRST_COMMENT_AFTER = 3       # seconds after (re)connecting
MIC_AUTO_OFF = 20             # mic turns itself off if you say nothing
MAX_RETRIES = 5
FATAL_ERRORS = {
    "api key not valid": "Your Google AI key doesn't work. Open TFTCoach and click \"Change key\".",
    "api_key_invalid": "Your Google AI key doesn't work. Open TFTCoach and click \"Change key\".",
    "not found": "The live model wasn't found. Set TFT_LIVE_MODEL in the .env file.",
    "not supported": "The live model doesn't support live audio. Set TFT_LIVE_MODEL in the .env file.",
}
CAPTION_CHARS = 180
COMMENT_PROMPT = ("Look at the newest frame. What is the single best thing the player should do "
                  "right now to win? Give the action and the reason.")
ADVICE_PROMPT = """Give the player's next instruction now.
Your analyst studied a high-resolution frame {age} s ago; their notes are below. Check them against the newest frame and say the most urgent order from "Do now" that you haven't said yet, with its reason. If the game moved on, adapt. When useful, add the plan for the next rounds in the same breath. No basics, no commentary.
ANALYST NOTES:
{notes}"""
ANALYSIS_FRESH = 40          # seconds an analysis stays usable

# Core rules that hold across TFT sets (set-specific traits/units are read from the screen).
TFT_RULES = """TFT RULES YOU MUST APPLY EXACTLY:
- Gold: +1 interest per 10 gold banked, capped at +5 at 50 gold. Spending below a multiple of 10 loses interest; win/loss streaks add bonus gold.
- XP: buying XP costs 4 gold for 4 XP. A shop refresh (reroll) costs 2 gold. Units on the board can't exceed the player's level.
- Upgrades: 3 copies of the same 1-star unit = 2-star; 3 copies of the 2-star = 3-star. The bench holds 9 units.
- Items: 2 components combine into 1 completed item; max 3 items per unit; completed items can't be split (except with special removers/reforgers).
- Higher levels raise the odds of higher-cost units in the shop; low-level shops are best for rerolling 1-3 cost units.
- Augments are picked at fixed points (typically stages 2-1, 3-2 and 4-2); carousels give one unit with an item.
- Standard tempo when healthy: level 4 at 2-1, 5 at 2-5, 6 at 3-2, 7 at 4-1, 8 around 4-2 to 5-1. Roll down when the board is weak and HP is dropping fast, or for a reroll comp at its level.
- HP matters more than gold when low: below ~40 HP stop greeding and stabilize; above ~70 HP with a weak board you can econ to 50.
- Only advise actions that are legal and affordable with the gold you can see. If you can't read a number, don't invent it."""

SYSTEM_PROMPT = f"""You are a Challenger-level Teamfight Tactics (TFT) coach giving LIVE, step-by-step guidance on a RECORDED TFT game video.
You receive about one frame per second. Call the person whose board is shown "the player".

You work with an analyst (a stronger model) who studies high-resolution frames and sends you notes with the target comp, item plan, econ/level/roll plan, positioning and orders. Speak their plan as your own; trust their reads of small numbers over yours, but drop anything the newest frame contradicts.

Your job is NOT commentary, and NOT basics: the player already knows how interest, XP and rerolls work. Never say generic things like "manage your economy" or "position well". Every time you speak, tell the player exactly what to do NEXT to maximize their chance to win (top 1, or at least top 4):
- Start with the action as an order, e.g. "Level to 6 now", "Roll down to 20 gold for Ahri two-star", "Slam Guinsoo's on Kog'Maw", "Hold at 50 and don't roll", "Move the carry to the back corner", "Take the Jeweled Lotus augment", "Sell the Darius, keep the pairs".
- Then give the reason in a few words (interest, HP, odds, item math, what beats the lobby).
- At most 2 short sentences. Be precise with numbers you can read on screen (gold, level, XP, stage, HP).
- Pick ONE best decision; never list options without choosing.

What to look at, in this order: stage and round, gold and interest, level and XP, HP vs the lobby, shop units and pairs on the bench, item components, augments, current board strength and positioning, the comp the player is building and its best final version.
In the planning phase advise shop/level/roll/items/positioning. During combat, prepare the next planning phase ("next round: level to 7 and roll for ...").
On augment, carousel or item choices, name the exact pick.
If the frame is unclear or not gameplay (menus, loading, webcam), say nothing new and wait for a clearer frame.
Don't repeat the same advice unless the player hasn't done it yet; then repeat it more urgently.
When the viewer asks a question out loud, answer it directly with a clear decision.

{{TFT_RULES}}
(Apply these rules silently; never recite them.)

Speak English, fast, clear and decisive, like a pro coach sitting next to the player.""".replace("{TFT_RULES}", TFT_RULES)


def encode_frame(bgr: np.ndarray) -> bytes:
    """Resize to 1024 wide and JPEG-encode (quality 80)."""
    h, w = bgr.shape[:2]
    if w != FRAME_WIDTH:
        new_h = max(1, int(round(h * FRAME_WIDTH / w)))
        interp = cv2.INTER_AREA if w > FRAME_WIDTH else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (FRAME_WIDTH, new_h), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def fatal_message(exc: Exception):
    """Errors that retrying can't fix get a clear message instead of retries."""
    text = str(exc).lower()
    for needle, message in FATAL_ERRORS.items():
        if needle in text:
            return message
    return None


class Player:
    """Speaker output: a byte buffer drained by a sounddevice callback."""

    def __init__(self):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self.last_audio = 0.0

    def start(self):
        try:
            import sounddevice as sd
            self._stream = sd.RawOutputStream(
                samplerate=OUT_RATE, channels=1, dtype="int16",
                blocksize=1200, callback=self._callback)
            self._stream.start()
        except Exception as e:  # no speakers: keep going with captions only
            print("Audio output unavailable:", e)
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        n = len(outdata)
        with self._lock:
            chunk = self._buf[:n]
            del self._buf[:n]
        outdata[:len(chunk)] = chunk
        if len(chunk) < n:
            outdata[len(chunk):] = b"\x00" * (n - len(chunk))

    def add(self, data: bytes):
        if self._stream is None:
            return
        with self._lock:
            self._buf.extend(data)
        self.last_audio = time.monotonic()

    def clear(self):
        with self._lock:
            self._buf.clear()

    @property
    def busy(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class Transcript:
    """Markdown transcript next to the exe."""

    def __init__(self, kind: str, source: str, model: str):
        self.path = app_dir() / f"coach_{kind}_log_{stamp()}.md"
        self.start = time.monotonic()
        self._lock = threading.Lock()
        title = "Live coaching (video file)" if kind == "live" else "Live coaching (screen)"
        self._write(f"# TFT Lab Coach - {title}\n\n"
                    f"- Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                    f"- Source: {source}\n- Model: {model}\n\n## Transcript\n\n")

    def _write(self, text: str):
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(text)
            except OSError as e:
                print("Transcript write failed:", e)

    def line(self, who: str, text: str):
        text = " ".join(text.split())
        if text:
            self._write(f"- `[{fmt_time(time.monotonic() - self.start)}]` **{who}:** {text}\n")

    def note(self, text: str):
        self._write(f"- `[{fmt_time(time.monotonic() - self.start)}]` _{text}_\n")

    def close(self, frames: int):
        length = time.monotonic() - self.start
        self._write(f"\n## Summary\n\n- Frames streamed: {frames}\n"
                    f"- Session length: {fmt_time(length)}\n")


class LiveCoach:
    def __init__(self, api_key: str, talk_every: float = 8, kind: str = "live",
                 source: str = "", model: str | None = None, client_factory=None):
        self.api_key = api_key
        self.talk_every = max(4.0, min(20.0, float(talk_every)))
        self.model = model or live_model()
        self.kind = kind
        self.source = source
        self._client_factory = client_factory

        self.status = "Connecting..."
        self.caption = ""
        self.mic_on = False
        self.paused = False
        self.frames_sent = 0
        self.error = None

        self._frame = None
        self._frame_seq = 0
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._loop = None
        self._thread = None
        self._mic_queue = None
        self._mic_stream = None
        self._player = Player()
        self._transcript = None

        self._handle = None
        self._model_turn = False      # model is producing a turn
        self._awaiting_since = None   # commentary prompt sent, waiting for answer
        self._coach_text = ""
        self._user_text = ""
        self._send_end = False
        self._heard_user = False      # mic on: the model transcribed the viewer
        self._mic_since = 0.0
        self._sent_seq = 0
        self._last_frame_at = 0.0
        self.analysis = None
        self._analysis_at = 0.0
        self._new_analysis = False
        self.strategist = None
        self.use_strategist = os.environ.get("TFT_ANALYST", "on").lower() not in ("0", "off", "false")

    # ---------- thread-safe API for the UI ----------
    def start(self):
        self._transcript = Transcript(self.kind, self.source, self.model)
        self._player.start()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if self.use_strategist:
            from strategist import Strategist
            self.strategist = Strategist(self.api_key, self.latest_frame, lambda: self.paused,
                                         self._on_analysis)
            self.strategist.start()

    def latest_frame(self):
        with self._frame_lock:
            return self._frame, self._frame_seq

    def _on_analysis(self, analysis: dict):
        """Called from the analyst thread with a fresh game plan."""
        self.analysis = analysis
        self._analysis_at = time.monotonic()
        self._new_analysis = True
        if self._transcript:
            orders = " / ".join(analysis.get("now") or [])
            roll = f" | roll math: {analysis['roll_math']}" if analysis.get("roll_math") else ""
            why = f" ({analysis['why']})" if analysis.get("why") else ""
            self._transcript.note(f"Analyst [{analysis.get('stage', '?')}]{why}: comp: {analysis.get('comp', '')}"
                                  f" | plan: {analysis.get('plan', '')} | now: {orders}{roll}")

    def advice_prompt(self) -> str:
        age = time.monotonic() - self._analysis_at
        if self.analysis and age < ANALYSIS_FRESH:
            from strategist import notes_text
            return ADVICE_PROMPT.format(age=int(age), notes=notes_text(self.analysis))
        return COMMENT_PROMPT

    def submit_frame(self, bgr):
        """Store the latest frame (old ones are dropped); None = nothing to send."""
        with self._frame_lock:
            self._frame = bgr
            if bgr is not None:
                self._frame_seq += 1

    def set_paused(self, paused: bool):
        self.paused = bool(paused)
        if self._transcript:
            self._transcript.note("Paused" if paused else "Resumed")

    def toggle_mic(self):
        if self._loop is None:
            return
        if self.mic_on:
            self._mic_off()
        else:
            self._heard_user = False
            self._mic_since = time.monotonic()
            self.mic_on = True
            self._player.clear()        # stop the coach talking
            self._start_mic()

    def _mic_off(self):
        if not self.mic_on:
            return
        self.mic_on = False
        self._stop_mic()
        self._send_end = True           # commentator task sends audio_stream_end

    @property
    def speaking(self) -> bool:
        return self._player.busy or self._model_turn

    def status_line(self) -> str:
        mic = "MIC ON - ask your question" if self.mic_on else "mic off"
        state = "PAUSED" if self.paused else self.status
        if self.strategist:
            if self.analysis:
                state += f" | analyst {int(time.monotonic() - self._analysis_at)}s ago"
            else:
                state += " | analyst warming up"
        return f"{state} | {mic}"

    def stop(self, wait: float = 3.0):
        if self._stop.is_set():
            return
        self._stop.set()
        if self.strategist:
            self.strategist.stop()
        self._stop_mic()
        if self._thread:
            self._thread.join(timeout=wait)
        self._player.stop()
        if self._transcript:
            self._flush_turn()
            self._transcript.close(self.frames_sent)

    @property
    def transcript_path(self):
        return self._transcript.path if self._transcript else None

    # ---------- mic ----------
    def _start_mic(self):
        if self._mic_stream is not None:
            return
        try:
            import sounddevice as sd
            loop, q = self._loop, self._mic_queue

            def cb(indata, frames, time_info, status):
                if self.mic_on:
                    loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

            self._mic_stream = sd.RawInputStream(
                samplerate=IN_RATE, channels=1, dtype="int16",
                blocksize=MIC_BLOCK, callback=cb)
            self._mic_stream.start()
        except Exception as e:
            print("Microphone unavailable:", e)
            self.mic_on = False
            self.caption = "Microphone not available."
            self._mic_stream = None

    def _stop_mic(self):
        s, self._mic_stream = self._mic_stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    # ---------- background thread ----------
    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._mic_queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:
            self.error = str(e)
            self.status = "Error"
            print("Engine crashed:", repr(e))
        finally:
            self._loop.close()

    def _config(self):
        from google.genai import types
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            # audio+video sessions die after ~2 min without compression
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()),
            # connections last ~10 min: resume with the latest handle
            session_resumption=types.SessionResumptionConfig(handle=self._handle),
        )

    async def _run(self):
        if self._client_factory is None:
            from google import genai
            self._client_factory = genai.Client
        client = self._client_factory(api_key=self.api_key)
        try:
            await self._connect_loop(client)
        finally:
            aclose = getattr(getattr(client, "aio", None), "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass

    async def _connect_loop(self, client):
        failures = 0
        while not self._stop.is_set():
            connected = False
            try:
                async with client.aio.live.connect(model=self.model, config=self._config()) as session:
                    connected = True
                    self.status = "Live"
                    failures = 0
                    reason = await self._session(session)
                if self._stop.is_set():
                    break
                print("Connection ended:", reason)
                reason_text = "connection refresh"
            except Exception as e:
                if self._stop.is_set():
                    break
                failures += 1
                reason_text = str(e)[:60]
                print(f"Live connection error ({failures}/{MAX_RETRIES}, connected={connected}):", repr(e))
                if not connected and self._handle and failures >= 2:
                    print("Dropping the resume handle and starting a fresh session")
                    self._handle = None
                fatal = fatal_message(e)
                if fatal:
                    self.status = "Stopped"
                    self.error = str(e)
                    self.caption = fatal
                    self._transcript.note(f"Stopped: {e}")
                    return
                if failures > MAX_RETRIES:
                    self.status = "Disconnected"
                    self.error = str(e)
                    self.caption = f"Couldn't reach the coach: {str(e)[:120]}"
                    self._transcript.note(f"Gave up after {MAX_RETRIES} retries: {e}")
                    return
                await self._sleep(0.5 if connected else min(2 ** failures, 20))
            self._model_turn = False
            self._awaiting_since = None
            self._player.clear()
            self._flush_turn()
            self.caption = ""
            self.status = "Reconnecting..."
            self._transcript.note(f"Reconnecting ({reason_text})")

    async def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            await asyncio.sleep(0.1)

    async def _session(self, session) -> str:
        tasks = [
            asyncio.create_task(self._receiver(session)),
            asyncio.create_task(self._frame_sender(session)),
            asyncio.create_task(self._mic_sender(session)),
            asyncio.create_task(self._commentator(session)),
            asyncio.create_task(self._stop_watch()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc
            return str(t.result())
        return "done"

    async def _stop_watch(self):
        while not self._stop.is_set():
            await asyncio.sleep(0.1)
        return "stopped"

    async def _receiver(self, session):
        while True:
            async for msg in session.receive():
                upd = msg.session_resumption_update
                if upd and upd.resumable and upd.new_handle:
                    self._handle = upd.new_handle
                if msg.go_away is not None:
                    print("GoAway, time left:", msg.go_away.time_left)
                    return "go_away"
                sc = msg.server_content
                if sc is None:
                    continue
                if sc.interrupted:
                    self._player.clear()
                    self._model_turn = False
                    self._flush_turn()
                if sc.input_transcription and sc.input_transcription.text:
                    self._user_text += sc.input_transcription.text
                    if self.mic_on:
                        self._heard_user = True
                if sc.model_turn and sc.model_turn.parts:
                    if self.mic_on and self._heard_user:
                        self._mic_off()          # the coach is answering: listen
                    if not self._model_turn:
                        self._model_turn = True
                        self._flush_user()
                        self._coach_text = ""
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            # while you are still asking, the coach stays quiet
                            if not self.mic_on:
                                self._player.add(part.inline_data.data)
                if sc.output_transcription and sc.output_transcription.text:
                    self._coach_text += sc.output_transcription.text
                    self._set_caption(self._coach_text)
                if sc.turn_complete:
                    self._model_turn = False
                    self._awaiting_since = None
                    self._flush_turn()
            if self._stop.is_set():
                return "stopped"

    def _set_caption(self, text: str):
        text = " ".join(text.split())
        if len(text) > CAPTION_CHARS:
            text = "..." + text[-CAPTION_CHARS:].lstrip()
        self.caption = text

    def _flush_user(self):
        if self._user_text.strip():
            self._transcript.line("You", self._user_text)
        self._user_text = ""

    def _flush_turn(self):
        self._flush_user()
        if self._coach_text.strip():
            self._transcript.line("Coach", self._coach_text)
        self._coach_text = ""

    async def _send_latest_frame(self, session) -> bool:
        from google.genai import types
        with self._frame_lock:
            frame, seq = self._frame, self._frame_seq
        if frame is None or seq == self._sent_seq:
            return False
        self._sent_seq = seq
        jpeg = await asyncio.to_thread(encode_frame, frame)
        await session.send_realtime_input(video=types.Blob(data=jpeg, mime_type="image/jpeg"))
        self.frames_sent += 1
        self._last_frame_at = time.monotonic()
        return True

    async def _frame_sender(self, session):
        self._sent_seq = 0          # new connection: resend the current frame
        self._last_frame_at = 0.0
        while True:
            await asyncio.sleep(0.1)
            if self.paused or time.monotonic() - self._last_frame_at < FRAME_INTERVAL:
                continue
            await self._send_latest_frame(session)

    async def _mic_sender(self, session):
        from google.genai import types
        while True:
            data = await self._mic_queue.get()
            if not self.mic_on:
                continue   # never send audio while the mic is off
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={IN_RATE}"))

    async def _commentator(self, session):
        last = time.monotonic() - self.talk_every + FIRST_COMMENT_AFTER
        while True:
            await asyncio.sleep(0.25)
            if self.mic_on and not self._heard_user and time.monotonic() - self._mic_since > MIC_AUTO_OFF:
                print("Mic auto-off (nothing heard)")
                self._mic_off()
            if self._send_end:
                self._send_end = False
                await session.send_realtime_input(audio_stream_end=True)
            now = time.monotonic()
            if self._awaiting_since and now - self._awaiting_since > STUCK_TURN_SECONDS:
                print("Resetting stuck turn")
                self._awaiting_since = None
                self._model_turn = False
            due = now - last >= self.talk_every
            early = self._new_analysis and now - last >= FIRST_COMMENT_AFTER
            if not (due or early):
                continue
            if self.paused or self.mic_on or self.speaking or self._awaiting_since:
                continue
            if self.frames_sent == 0:
                continue
            # only 1 fps is allowed: wait for the slot, then send the freshest frame
            if now - self._last_frame_at >= FRAME_INTERVAL:
                await self._send_latest_frame(session)
            self._new_analysis = False
            await session.send_realtime_input(text=self.advice_prompt())
            self._awaiting_since = now
            last = now
