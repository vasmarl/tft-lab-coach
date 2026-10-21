"""Google AI key: save/load in a .env next to the exe, and validation."""
import os

from common import env_path

KEY_NAME = "GOOGLE_API_KEY"

MSG_OK = "Key works."
MSG_INVALID = "This key doesn't work. Copy it again from Google AI Studio."
MSG_NETWORK = "Couldn't check the key (internet?)"


def load_env() -> None:
    """Load the .env next to the exe into os.environ (does not override)."""
    path = env_path()
    if path.exists():
        from dotenv import load_dotenv
        load_dotenv(path, override=False)


def load_key() -> str:
    path = env_path()
    if path.exists():
        from dotenv import dotenv_values
        value = dotenv_values(path).get(KEY_NAME)
        if value:
            return value.strip()
    return (os.environ.get(KEY_NAME) or os.environ.get("GEMINI_API_KEY") or "").strip()


def save_key(key: str) -> None:
    """Write/replace GOOGLE_API_KEY in the .env, keeping any other lines."""
    key = key.strip()
    path = env_path()
    lines = []
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                 if not ln.strip().startswith(KEY_NAME + "=")]
    lines.append(f"{KEY_NAME}={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[KEY_NAME] = key


def check_key(key: str, client_factory=None) -> tuple:
    """Validate a key by listing one model. Returns (ok, message)."""
    key = (key or "").strip()
    if not key:
        return False, MSG_INVALID
    from google.genai import errors
    if client_factory is None:
        from google import genai
        client_factory = genai.Client
    try:
        client = client_factory(api_key=key)
        pager = client.models.list(config={"page_size": 1})
        next(iter(pager), None)
        return True, MSG_OK
    except errors.ClientError:
        # 400 "API key not valid", 401, 403 ...
        return False, MSG_INVALID
    except errors.APIError:
        return False, MSG_NETWORK
    except Exception:
        # httpx connect errors, DNS failures, timeouts
        return False, MSG_NETWORK
