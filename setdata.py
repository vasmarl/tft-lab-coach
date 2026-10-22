"""Current-set data (champions, traits, item recipes) for the analyst.

tft_set_data.json is built from CommunityDragon at build time and bundled in
the exe. A newer copy placed next to the exe wins (for a new patch without a
rebuild).
"""
import json
import sys
from pathlib import Path

from common import app_dir

FILE = "tft_set_data.json"


def _candidates():
    yield app_dir() / FILE
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        yield Path(bundle) / FILE
    yield Path(__file__).resolve().parent / FILE


def load() -> dict:
    for path in _candidates():
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print("Set data unreadable:", path, e)
    return {}


def prompt_text(data: dict) -> str:
    """Compact reference the model can't misremember."""
    if not data.get("champions"):
        return ""
    s = data.get("set") or {}
    lines = [f"SET DATA (TFT set {s.get('number')}, patch {data.get('version', '')[:5]}) - "
             "use these exact names, costs, trait breakpoints and recipes; never invent others."]
    by_cost = {}
    for c in data["champions"]:
        by_cost.setdefault(c["cost"], []).append(f"{c['name']} ({'/'.join(c['traits'])})")
    for cost in sorted(by_cost):
        lines.append(f"{cost}-cost: " + "; ".join(by_cost[cost]))
    traits = [f"{t['name']} {'/'.join(map(str, t['breakpoints']))}" for t in data.get("traits") or []
              if t.get("breakpoints")]
    if traits:
        lines.append("Traits (breakpoints): " + "; ".join(traits))
    items = [f"{i['name']} = {' + '.join(i['from'])}" for i in data.get("items") or []]
    if items:
        lines.append("Item recipes: " + "; ".join(items))
    return "\n".join(lines)
