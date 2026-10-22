"""Build tft_set_data.json (current TFT set) from CommunityDragon.

Run at build time (CI) so the exe ships the set data of the current patch:
    python tools/build_set_data.py [cdragon_en_us.json] [out.json]
Without an input file it downloads the latest data from CommunityDragon.
"""
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

CDRAGON = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
META = "https://raw.communitydragon.org/latest/content-metadata.json"


def fetch(url: str, timeout: int = 600):
    req = urllib.request.Request(url, headers={"User-Agent": "tft-lab-coach-build"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def current_set(set_data: list) -> dict:
    """The main set with the highest number (skip tutorials/PvE/revival modes)."""
    candidates = []
    for s in set_data:
        mutator = str(s.get("mutator") or "")
        if not re.fullmatch(r"TFTSet\d+", mutator):
            continue
        champs = [c for c in s.get("champions") or [] if 1 <= int(c.get("cost") or 0) <= 5 and c.get("traits")]
        if len(champs) < 30:
            continue
        candidates.append((int(s.get("number") or 0), s))
    if not candidates:
        raise SystemExit("No TFT set found in CommunityDragon data")
    return max(candidates, key=lambda x: x[0])[1]


def build(raw: dict, version: str = "") -> dict:
    s = current_set(raw.get("setData") or [])
    number = int(s.get("number") or 0)

    trait_names = {}
    traits = []
    for t in s.get("traits") or []:
        name = clean(t.get("name"))
        if not name:
            continue
        trait_names[t.get("apiName")] = name
        breakpoints = sorted({int(e["minUnits"]) for e in t.get("effects") or [] if e.get("minUnits")})
        traits.append({"name": name, "breakpoints": breakpoints})

    champions = []
    for c in s.get("champions") or []:
        cost = int(c.get("cost") or 0)
        if not (1 <= cost <= 5) or not c.get("traits"):
            continue
        champions.append({"name": clean(c.get("name")), "cost": cost,
                          "traits": [trait_names.get(t, clean(t)) for t in c["traits"]]})
    champions.sort(key=lambda c: (c["cost"], c["name"]))

    items_by_api = {i.get("apiName"): i for i in raw.get("items") or []}
    prefix_ok = ("TFT_Item_", f"TFT{number}_Item_")
    items = []
    seen = set()
    for i in raw.get("items") or []:
        api = str(i.get("apiName") or "")
        comp = i.get("composition") or []
        if i.get("isAugment") or len(comp) != 2 or not api.startswith(prefix_ok):
            continue
        name = clean(i.get("name"))
        parts = [clean((items_by_api.get(p) or {}).get("name") or p) for p in comp]
        if not name or name in seen or name.lower().startswith("tft_item_name"):
            continue
        seen.add(name)
        items.append({"name": name, "from": parts})
    items.sort(key=lambda i: i["name"])

    # distinct shop units per cost (variants like "Lux (Fae)" share one pool entry)
    base = {(re.sub(r"\s*\(.*\)$", "", c["name"]), c["cost"]) for c in champions}
    per_cost = {}
    for _, cost in base:
        per_cost[str(cost)] = per_cost.get(str(cost), 0) + 1

    return {"source": "CommunityDragon", "version": version, "built": f"{date.today():%Y-%m-%d}",
            "set": {"number": number, "name": clean(s.get("name")), "mutator": s.get("mutator")},
            "units_per_cost": per_cost, "champions": champions, "traits": traits, "items": items}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tft_set_data.json")
    raw = json.loads(Path(src).read_text(encoding="utf-8")) if src else fetch(CDRAGON)
    try:
        version = fetch(META, 30).get("version", "")
    except Exception:
        version = ""
    data = build(raw, version)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Set {data['set']['number']} ({data['set']['name']}): {len(data['champions'])} champions, "
          f"{len(data['traits'])} traits, {len(data['items'])} items -> {out}")


if __name__ == "__main__":
    main()
