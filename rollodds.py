"""Roll math: chance to hit copies of a unit when rolling down.

Shop odds and pool sizes default to the long-standing TFT values; the set data
or the analyst's meta lookup can override them when a set changes them.
"""
import numpy as np

# level -> chance per shop slot of cost 1..5 (in %)
DEFAULT_SHOP_ODDS = {
    1: [100, 0, 0, 0, 0],
    2: [100, 0, 0, 0, 0],
    3: [75, 25, 0, 0, 0],
    4: [55, 30, 15, 0, 0],
    5: [45, 33, 20, 2, 0],
    6: [30, 40, 25, 5, 0],
    7: [19, 30, 40, 10, 1],
    8: [18, 25, 32, 22, 3],
    9: [15, 20, 25, 30, 10],
    10: [5, 10, 20, 40, 25],
}
# copies of EACH unit in the shared pool, by cost
DEFAULT_POOL_COPIES = {1: 30, 2: 25, 3: 18, 4: 10, 5: 9}
REROLL_COST = 2
SLOTS = 5


def hit_chance(level: int, cost: int, need: int, gold: int, units_of_cost: int,
               copies_gone: int = 0, other_cost_gone: int = 0, shop_odds=None,
               pool_copies=None, trials: int = 4000, seed: int = 7) -> float:
    """Probability of buying `need` more copies of one `cost`-cost unit with `gold`.

    copies_gone: copies of this unit already out of the pool (yours + opponents').
    other_cost_gone: other units of the same cost out of the pool.
    units_of_cost: how many different champions of that cost exist in the set.
    Buying a copy costs `cost` gold; each refresh costs 2.
    """
    shop_odds = shop_odds or DEFAULT_SHOP_ODDS
    pool_copies = pool_copies or DEFAULT_POOL_COPIES
    if need <= 0:
        return 1.0
    p_tier = shop_odds.get(int(level), shop_odds[max(shop_odds)])[cost - 1] / 100.0
    per_unit = pool_copies[cost]
    mine_left = max(0, per_unit - copies_gone)
    tier_left = max(1, per_unit * units_of_cost - copies_gone - other_cost_gone)
    if p_tier == 0 or mine_left < need:
        return 0.0

    # all trials at once: each loop iteration is one refresh in every trial
    rng = np.random.default_rng(seed)
    g = np.full(trials, gold, dtype=np.int64)
    got = np.zeros(trials, dtype=np.int64)
    left = np.full(trials, mine_left, dtype=np.float64)
    total = np.full(trials, tier_left, dtype=np.float64)
    while True:
        active = (g >= REROLL_COST) & (got < need)
        if not active.any():
            break
        g = np.where(active, g - REROLL_COST, g)
        for _ in range(SLOTS):
            hit = (active & (got < need) & (g >= cost)
                   & (rng.random(trials) < p_tier)
                   & (rng.random(trials) < left / np.maximum(total, 1)))
            g = np.where(hit, g - cost, g)
            got += hit
            left -= hit
            total -= hit
    return float((got >= need).mean())


def summary(target: dict, set_info: dict | None = None) -> str:
    """One-line roll math for the coach, e.g. at current level vs one level up."""
    try:
        unit = str(target.get("unit") or "the unit")
        cost = int(target["cost"])
        level = int(target["level"])
        need = int(target["need"])
        gold = int(target["gold"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not (1 <= cost <= 5 and 1 <= level <= 10 and gold >= 0 and need > 0):
        return ""
    set_info = set_info or {}
    units_of_cost = (set_info.get("units_per_cost") or {}).get(str(cost)) or {1: 13, 2: 13, 3: 13, 4: 12, 5: 8}[cost]
    kw = dict(cost=cost, need=need, units_of_cost=int(units_of_cost),
              copies_gone=int(target.get("copies_gone") or 0),
              shop_odds=set_info.get("shop_odds"), pool_copies=set_info.get("pool_copies"))
    now = hit_chance(level, gold=gold, **kw)
    parts = [f"{now:.0%} to hit {need} more {unit} rolling {gold} gold at level {level}"]
    if level < 10 and gold >= 8:
        # leveling costs gold too: estimate the XP buy as ~gold/2 at most 4 buys (16 gold)
        spend = min(16, max(4, int(target.get("level_cost") or 8)))
        up = hit_chance(level + 1, gold=max(0, gold - spend), **kw)
        parts.append(f"{up:.0%} if you level to {level + 1} first ({spend} gold) then roll")
    return "; ".join(parts)
