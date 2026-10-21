"""Engine runs against a fake Live session (no network, no audio devices)."""
import asyncio
import time
from types import SimpleNamespace as NS

import numpy as np

import engine


def msg(**kw):
    base = dict(session_resumption_update=None, go_away=None, server_content=None)
    base.update(kw)
    return NS(**base)


def content(**kw):
    base = dict(interrupted=False, input_transcription=None, output_transcription=None,
                model_turn=None, turn_complete=False)
    base.update(kw)
    return NS(**base)


class FakeSession:
    def __init__(self, log, go_away_after=None):
        self.log = log
        self.inbox = asyncio.Queue()
        self.go_away_after = go_away_after
        self.texts = 0

    async def send_realtime_input(self, **kw):
        self.log.append(kw)
        if kw.get("text"):
            self.texts += 1
            if self.go_away_after and self.texts >= self.go_away_after:
                await self.inbox.put([msg(go_away=NS(time_left="5s"))])
                return
            part = NS(inline_data=NS(data=b"\x00\x00" * 100))
            await self.inbox.put([
                msg(server_content=content(model_turn=NS(parts=[part]))),
                msg(server_content=content(output_transcription=NS(text="Slam the sword now."))),
                msg(server_content=content(turn_complete=True)),
            ])

    async def receive(self):
        yield msg(session_resumption_update=NS(resumable=True, new_handle="h1"))
        for m in await self.inbox.get():
            yield m


class FakeClient:
    def __init__(self, **kw):
        self.sends = []
        self.configs = []
        self.sessions = 0
        outer = self

        class Ctx:
            def __init__(self, model, config):
                outer.configs.append(config)
                outer.sessions += 1
                self.s = FakeSession(outer.sends, go_away_after=2 if outer.sessions == 1 else None)

            async def __aenter__(self):
                return self.s

            async def __aexit__(self, *a):
                return False

        self.aio = NS(live=NS(connect=lambda model, config: Ctx(model, config)))


def test_encode_frame_is_1024_wide_jpeg():
    data = engine.encode_frame(np.zeros((720, 1280, 3), np.uint8))
    assert data[:2] == b"\xff\xd8"
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 1024


def test_engine_comments_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(engine, "FRAME_INTERVAL", 0.05)
    fake = FakeClient()
    coach = engine.LiveCoach("k", talk_every=4, kind="live", source="x.mp4",
                             client_factory=lambda api_key: fake)
    coach.talk_every = 0.2
    coach._player._stream = object()   # pretend a speaker exists (buffer only)
    coach._player.stop = lambda: None
    coach._transcript = None
    coach._transcript = engine.Transcript("live", "x.mp4", "m")
    coach._thread = __import__("threading").Thread(target=coach._thread_main, daemon=True)
    coach._thread.start()
    coach.submit_frame(np.zeros((360, 640, 3), np.uint8))
    deadline = time.time() + 5
    while time.time() < deadline and fake.sessions < 2:
        coach._player.clear()
        time.sleep(0.05)
    time.sleep(0.6)
    coach.stop()

    assert any("video" in s for s in fake.sends)
    assert any(s.get("text") == engine.COMMENT_PROMPT for s in fake.sends)
    assert fake.sessions >= 2                       # reconnected after GoAway
    assert fake.configs[1].session_resumption.handle == "h1"
    assert fake.configs[0].context_window_compression.sliding_window is not None
    assert not any("audio" in s for s in fake.sends)  # mic off -> no audio
    log = coach.transcript_path.read_text()
    assert "**Coach:** Slam the sword now." in log
    assert "Frames streamed" in log


def test_bad_key_stops_without_retrying():
    err = Exception("1007 None. API key not valid. Please pass a valid API key.")
    assert "Change key" in engine.fatal_message(err)
    assert engine.fatal_message(Exception("1011 internal error")) is None
