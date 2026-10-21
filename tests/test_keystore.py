import sys
from types import SimpleNamespace as NS

import pytest
from google.genai import errors

import common
import keystore


@pytest.fixture
def exe_dir(tmp_path, monkeypatch):
    """Pretend we are a frozen exe living in tmp_path."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "TFTCoach.exe"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return tmp_path


def test_env_is_next_to_exe(exe_dir):
    assert common.app_dir() == exe_dir
    assert common.env_path() == exe_dir / ".env"


def test_save_and_load_key(exe_dir):
    assert keystore.load_key() == ""
    (exe_dir / ".env").write_text("TFT_LIVE_MODEL=other\nGOOGLE_API_KEY=old\n")
    keystore.save_key("  AIzaNEWKEY123  ")
    text = (exe_dir / ".env").read_text()
    assert "GOOGLE_API_KEY=AIzaNEWKEY123" in text
    assert "old" not in text
    assert "TFT_LIVE_MODEL=other" in text
    assert keystore.load_key() == "AIzaNEWKEY123"


def client_raising(exc):
    def factory(api_key):
        def list_models(config=None):
            raise exc
        return NS(models=NS(list=list_models))
    return factory


def test_invalid_key_message():
    exc = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid.",
                                             "status": "INVALID_ARGUMENT"}})
    ok, msg = keystore.check_key("bad", client_factory=client_raising(exc))
    assert not ok
    assert msg == "This key doesn't work. Copy it again from Google AI Studio."


def test_network_error_message():
    ok, msg = keystore.check_key("k", client_factory=client_raising(ConnectionError("offline")))
    assert not ok
    assert msg == "Couldn't check the key (internet?)"


def test_valid_key():
    factory = lambda api_key: NS(models=NS(list=lambda config=None: iter([NS(name="models/x")])))
    assert keystore.check_key("good", client_factory=factory) == (True, keystore.MSG_OK)
