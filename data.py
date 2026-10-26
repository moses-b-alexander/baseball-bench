"""Offline data: profile schema helpers, cutoff-safe aggregation, synthetic demo snapshot, validation.

All demo players, games, and numbers are synthetic.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA_VERSION = "0.3"
MAPPING_VERSION = "map-0.1"
PITCH_REGISTRY = {"version": "demo-registry-0.1", "codes": ["FF", "SI", "FC", "SL", "CU", "CH", "UNKNOWN"]}
OUTCOMES = ("balls", "strikes", "fouls", "hits", "in_play_outs")
HIT_TYPES = ("single", "double", "triple", "home_run")
OUT_RESULTS = ("strikeout", "in_play_out")
HAND_SIDE = {"L": "left", "R": "right"}
IDEAL_ATTACK_ANGLE_DEG = (5.0, 20.0)
HORIZONTAL_BREAK_CONVENTION = "catcher_perspective_positive_toward_first_base_side"
SHAPE_FIELDS = ("velocity_mph", "ivb_in", "horizontal_break_in")
SWING_FIELDS = ("attack_angle_deg", "swing_path_tilt_deg", "bat_speed_mph")

# Raw pitch description -> simulator outcome. Anything else (HBP, errors, interference, ...) is excluded and counted.
RAW_TO_OUTCOME = {"ball": "balls", "called_strike": "strikes", "swinging_strike": "strikes", "foul": "fouls"}

PITCH_COLUMNS = ["game_id", "date", "ts", "pa_id", "pitch_num", "pitcher_id", "p_throws", "batter_id", "stand",
                 "pitch_type", "raw_description", "outcome", "pa_result", "velocity_mph", "ivb_in",
                 "horizontal_break_in", "swing", "competitive_swing", "attack_angle_deg", "swing_path_tilt_deg",
                 "bat_speed_mph"]
PA_EVENT_COLUMNS = ["game_id", "date", "ts", "pa_id", "pitcher_id", "p_throws", "batter_id", "stand", "event"]


def map_outcome(raw, pa_result):
    if raw in RAW_TO_OUTCOME:
        return RAW_TO_OUTCOME[raw]
    if raw == "hit_into_play":
        if pa_result in HIT_TYPES:
            return "hits"
        if pa_result == "in_play_out":
            return "in_play_outs"
    return None


def resolve_side(bats, throws):
    """Batting side for one matchup; switch hitters bat opposite the pitcher's hand."""
    if bats == "S":
        return "L" if throws == "R" else "R"
    return bats


def ip_display(outs):
    return f"{outs // 3}.{outs % 3}"


def ip_to_outs(ip):
    whole, _, frac = ip.partition(".")
    frac = int(frac or 0)
    if frac not in (0, 1, 2):
        raise ValueError(f"invalid innings notation {ip!r}")
    return int(whole) * 3 + frac


# ---------------------------------------------------------------- schema primitives

def mean_metric():
    return {"mean": None, "n": 0}


def rate_metric():
    return {"value": None, "numerator": 0, "denominator": 0}


def pitcher_split(codes=None):
    codes = codes or PITCH_REGISTRY["codes"]
    return {"pitch_count": 0, "completed_pa": 0, "intentional_walks": 0, "outcomes": dict.fromkeys(OUTCOMES, 0),
            "pitch_types": {c: {"count": 0, "shape": {f: mean_metric() for f in SHAPE_FIELDS}} for c in codes}}


def batter_split():
    swing = {f: mean_metric() for f in SWING_FIELDS}
    swing["ideal_attack_angle_rate"] = rate_metric()
    return {"pitch_count": 0, "completed_pa": 0, "intentional_walks": 0, "outcomes": dict.fromkeys(OUTCOMES, 0),
            "swing_profile": swing}


def splits(make):
    return {side: make() for side in ("left", "right", "both")}


def combine(a, b):
    """Add two schema values: ints sum, means weight by their own n, rates recompute from parts."""
    if isinstance(a, dict):
        if a.keys() == {"mean", "n"}:
            n = a["n"] + b["n"]
            if n == 0:
                return mean_metric()
            total = (a["mean"] or 0.0) * a["n"] + (b["mean"] or 0.0) * b["n"]
            return {"mean": total / n, "n": n}
        if a.keys() == {"value", "numerator", "denominator"}:
            num, den = a["numerator"] + b["numerator"], a["denominator"] + b["denominator"]
            return {"value": num / den if den else None, "numerator": num, "denominator": den}
        return {k: combine(a[k], b[k]) for k in a}
    return a + b


def observe_mean(metric, x):
    return combine(metric, {"mean": x, "n": 1})


def add_pitch(split, row, role):
    split["pitch_count"] += 1
    split["outcomes"][row["outcome"]] += 1
    if role == "pitcher":
        code = row["pitch_type"] if row["pitch_type"] in split["pitch_types"] else "UNKNOWN"
        entry = split["pitch_types"][code]
        entry["count"] += 1
        for f in SHAPE_FIELDS:
            if row[f] is not None:
                entry["shape"][f] = observe_mean(entry["shape"][f], row[f])
    elif row["swing"]:
        swing = split["swing_profile"]
        for f in ("attack_angle_deg", "swing_path_tilt_deg"):
            if row[f] is not None:
                swing[f] = observe_mean(swing[f], row[f])
        if row["bat_speed_mph"] is not None and row["competitive_swing"]:
            swing["bat_speed_mph"] = observe_mean(swing["bat_speed_mph"], row["bat_speed_mph"])
        if row["attack_angle_deg"] is not None:
            lo, hi = IDEAL_ATTACK_ANGLE_DEG
            ideal = lo <= row["attack_angle_deg"] <= hi
            swing["ideal_attack_angle_rate"] = combine(
                swing["ideal_attack_angle_rate"], {"value": None, "numerator": int(ideal), "denominator": 1})


def add_pa(split, result):
    split["completed_pa"] += 1
    if result == "intentional_walk":
        split["intentional_walks"] += 1


def finalize(sp):
    sp["both"] = combine(sp["left"], sp["right"])
    return sp


# ---------------------------------------------------------------- cutoff-safe aggregation

def aggregate(pitches, pa_events, scenario, pitcher_ids, batter_ids):
    """Build season-before-game and current-game-prefix splits plus frozen simulator tables.

    Season rows must precede `historical_before`; current-game rows must precede `branch_ts`.
    Everything else (later games, the rest of the real game) is excluded and counted.
    """
    cutoff, branch, gid = scenario["historical_before"], scenario["branch_ts"], scenario["game_id"]

    def window(row):
        if row["game_id"] == gid:
            return "game_prefix" if row["ts"] < branch else None
        return "season" if row["ts"] < cutoff else None

    prof_p = {p: {"season": splits(pitcher_split), "game_prefix": splits(pitcher_split)} for p in pitcher_ids}
    prof_b = {b: {"season": splits(batter_split), "game_prefix": splits(batter_split)} for b in batter_ids}
    sim_p = {p: {"left": {}, "right": {}} for p in pitcher_ids}
    sim_b = {b: {"left": dict.fromkeys(OUTCOMES, 0), "right": dict.fromkeys(OUTCOMES, 0)} for b in batter_ids}
    reference = dict.fromkeys(OUTCOMES, 0)
    hit_types = dict.fromkeys(HIT_TYPES, 0)
    outs_before_game = Counter()
    excluded = Counter()
    used = 0

    for row in pitches:
        w = window(row)
        if w is None:
            excluded["outside_cutoff_window"] += 1
            continue
        if row["stand"] not in HAND_SIDE or row["p_throws"] not in HAND_SIDE:
            excluded["unresolved_handedness"] += 1
            continue
        if row["outcome"] is None:
            excluded[f"unsupported:{row['raw_description']}"] += 1
            continue
        used += 1
        o, pid, bid = row["outcome"], row["pitcher_id"], row["batter_id"]
        reference[o] += 1
        if row["pa_result"] in HIT_TYPES:
            hit_types[row["pa_result"]] += 1
        pside, bside = HAND_SIDE[row["stand"]], HAND_SIDE[row["p_throws"]]
        if pid in prof_p:
            add_pitch(prof_p[pid][w][pside], row, "pitcher")
            code = row["pitch_type"] if row["pitch_type"] in PITCH_REGISTRY["codes"] else "UNKNOWN"
            cell = sim_p[pid][pside].setdefault(code, dict.fromkeys(OUTCOMES, 0))
            cell[o] += 1
            if row["pa_result"]:
                add_pa(prof_p[pid][w][pside], row["pa_result"])
                if w == "season" and row["pa_result"] in OUT_RESULTS:
                    outs_before_game[pid] += 1
        if bid in prof_b:
            add_pitch(prof_b[bid][w][bside], row, "batter")
            sim_b[bid][bside][o] += 1
            if row["pa_result"]:
                add_pa(prof_b[bid][w][bside], row["pa_result"])

    for ev in pa_events:  # zero-pitch PA events (intentional walks): never fabricate pitches for them
        w = window(ev)
        if w is None:
            excluded["pa_event_outside_cutoff_window"] += 1
            continue
        pside, bside = HAND_SIDE[ev["stand"]], HAND_SIDE[ev["p_throws"]]
        if ev["pitcher_id"] in prof_p:
            add_pa(prof_p[ev["pitcher_id"]][w][pside], ev["event"])
        if ev["batter_id"] in prof_b:
            add_pa(prof_b[ev["batter_id"]][w][bside], ev["event"])

    for prof in list(prof_p.values()) + list(prof_b.values()):
        finalize(prof["season"])
        finalize(prof["game_prefix"])
    return {"pitchers": prof_p, "batters": prof_b,
            "sim": {"reference_counts": reference, "hit_type_counts": hit_types,
                    "pitchers": sim_p, "batters": sim_b},
            "outs_before_game": dict(outs_before_game), "excluded": dict(sorted(excluded.items())),
            "used_pitches": used}


def workload_days(pitches, pitcher_id, team_pitchers, before_date):
    """Five most recent completed team game-dates before `before_date`, including zero-appearance dates."""
    team_dates = {}
    for row in pitches:
        if row["pitcher_id"] in team_pitchers and row["date"] < before_date:
            team_dates.setdefault(row["date"], set()).add(row["game_id"])
    outs = Counter(row["date"] for row in pitches
                   if row["pitcher_id"] == pitcher_id and row["date"] < before_date
                   and row["outcome"] is not None and row["pa_result"] in OUT_RESULTS)
    last = sorted(team_dates)[-5:]
    if len(last) < 5:
        raise ValueError(f"fewer than five prior team game-dates for {pitcher_id}")
    return [{"date": d, "game_ids": sorted(team_dates[d]), "outs_recorded": outs[d],
             "ip_display": ip_display(outs[d]), "data_status": "known"} for d in last]


# ---------------------------------------------------------------- CSV I/O

def _parse(value, kind):
    if value == "":
        return None
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "bool":
        return value == "1"
    return value


PITCH_KINDS = {"pitch_num": "int", "velocity_mph": "float", "ivb_in": "float", "horizontal_break_in": "float",
               "attack_angle_deg": "float", "swing_path_tilt_deg": "float", "bat_speed_mph": "float",
               "swing": "bool", "competitive_swing": "bool"}


def read_csv(path, kinds=None):
    kinds = kinds or {}
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _parse(v, kinds.get(k, "str")) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, rows, columns):
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        return v
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({c: fmt(row[c]) for c in columns})


def load_records(snapshot):
    snapshot = Path(snapshot)
    return read_csv(snapshot / "pitches.csv", PITCH_KINDS), read_csv(snapshot / "pa_events.csv")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- synthetic demo generator

BASE_PROBS = {"balls": 0.36, "strikes": 0.25, "fouls": 0.18, "hits": 0.065, "in_play_outs": 0.145}

# id: throws, stuff (higher = more strikes, fewer hits), arsenal {code: (usage, velo, ivb, arm-side break)}
DEMO_PITCHERS = {
    "pitcher_01": ("R", 1.05, {"FF": (0.45, 94.6, 16.2, 8.1), "SL": (0.30, 85.4, 1.8, -5.2), "CH": (0.25, 86.3, 7.4, 14.0)}),
    "reliever_02": ("L", 1.00, {"FF": (0.55, 93.8, 17.0, 7.5), "SL": (0.45, 83.9, 0.6, -6.8)}),
    "reliever_03": ("R", 1.10, {"SI": (0.50, 95.9, 6.1, 16.3), "SL": (0.50, 86.8, 2.4, -4.1)}),
    "reliever_04": ("R", 0.95, {"FF": (0.60, 96.4, 18.3, 6.0), "CU": (0.40, 80.2, -12.5, -8.7)}),
    "reliever_05": ("L", 1.00, {"SI": (0.60, 92.1, 7.8, 15.1), "CH": (0.40, 84.0, 5.2, 13.2)}),
    "reliever_06": ("R", 1.08, {"FC": (0.55, 91.7, 10.2, -2.5), "SL": (0.45, 84.8, 0.9, -7.3)}),
    "reliever_07": ("R", 1.02, {"FF": (0.50, 97.1, 16.6, 7.0), "SL": (0.50, 87.0, 1.5, -5.5)}),
}
DEMO_STARTER = "pitcher_01"
# Usage/health fields that game logs cannot supply; all synthetic.
DEMO_HEALTH = {
    "pitcher_01": (None, "no_recorded_stint", True),
    "reliever_02": (None, "no_recorded_stint", True),
    "reliever_03": (212, "known_return", True),
    "reliever_04": (None, "no_recorded_stint", True),
    "reliever_05": (6, "known_return", True),
    "reliever_06": (None, "unavailable", True),
    "reliever_07": (None, "no_recorded_stint", False),
}
# id: bats, power (higher = more hits), swing baseline (attack angle, tilt, bat speed)
DEMO_BATTERS = {
    "batter_01": ("L", 1.00, (9.0, 31.0, 70.5)), "batter_02": ("R", 0.95, (7.5, 33.0, 69.0)),
    "batter_03": ("S", 1.10, (11.0, 30.0, 72.4)), "batter_04": ("R", 1.25, (14.0, 35.0, 75.1)),
    "batter_05": ("L", 1.15, (12.5, 29.0, 73.8)), "batter_06": ("R", 1.00, (8.0, 34.0, 71.0)),
    "batter_07": ("L", 0.90, (6.5, 28.0, 68.2)), "batter_08": ("R", 0.85, (5.0, 32.0, 67.5)),
    "batter_09": ("S", 0.80, (4.5, 30.0, 66.9)),
}
DEMO_SCENARIO_DATE = "2025-06-10"
DEMO_GAME_ID = "demo_game_001"


def _ts(day, seconds):
    return (datetime.fromisoformat(day + "T23:05:00") + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_pa(rng, game_id, day, clock, pa_id, p, b, allow_rare=True):
    """Generate one synthetic PA as pitch rows (or one zero-pitch IBB event). Returns rows, event, result, clock."""
    throws, stuff, arsenal = p["throws"], p["stuff"], p["arsenal"]
    stand = resolve_side(b["bats"], throws)
    base = {"game_id": game_id, "date": day, "pa_id": pa_id, "pitcher_id": p["id"], "p_throws": throws,
            "batter_id": b["id"], "stand": stand}
    if allow_rare and rng.random() < 0.01:
        return [], dict(base, ts=_ts(day, clock), event="intentional_walk"), "intentional_walk", clock + 30
    platoon = 0.9 if stand == throws else 1.08
    probs = dict(BASE_PROBS)
    probs["strikes"] *= stuff
    probs["hits"] *= b["power"] * platoon / stuff
    codes = sorted(arsenal)
    rows, balls, strikes = [], 0, 0
    arm = 1 if throws == "L" else -1  # arm-side break is toward first base for a LHP (catcher's view)
    while True:
        clock += 20
        code = rng.choices(codes, weights=[arsenal[c][0] for c in codes])[0]
        _, velo, ivb, hb = arsenal[code]
        row = dict(base, ts=_ts(day, clock), pitch_num=len(rows) + 1, pitch_type=code, pa_result=None,
                   velocity_mph=None if rng.random() < 0.03 else round(rng.gauss(velo, 0.9), 1),
                   ivb_in=None if rng.random() < 0.03 else round(rng.gauss(ivb, 1.6), 1),
                   horizontal_break_in=None if rng.random() < 0.03 else round(arm * rng.gauss(hb, 1.6), 1),
                   swing=False, competitive_swing=False, attack_angle_deg=None, swing_path_tilt_deg=None,
                   bat_speed_mph=None)
        if allow_rare and rng.random() < 0.004:
            row.update(raw_description="hit_by_pitch", pa_result="hit_by_pitch", outcome=None)
            rows.append(row)
            return rows, None, "hit_by_pitch", clock
        o = rng.choices(OUTCOMES, weights=[probs[k] for k in OUTCOMES])[0]
        result = None
        if o == "balls":
            raw = "ball"
            balls += 1
            result = "walk" if balls == 4 else None
        elif o == "strikes":
            raw = "swinging_strike" if rng.random() < 0.45 else "called_strike"
            strikes += 1
            result = "strikeout" if strikes == 3 else None
        elif o == "fouls":
            raw = "foul"
            strikes = min(strikes + 1, 2)
        else:
            raw = "hit_into_play"
            result = rng.choices(HIT_TYPES, weights=[64, 20, 2, 14])[0] if o == "hits" else "in_play_out"
        if raw in ("swinging_strike", "foul", "hit_into_play"):
            aa, tilt, bs = b["swing"]
            row.update(swing=True, competitive_swing=rng.random() < 0.9,
                       attack_angle_deg=None if rng.random() < 0.05 else round(rng.gauss(aa, 6.0), 1),
                       swing_path_tilt_deg=None if rng.random() < 0.05 else round(rng.gauss(tilt, 4.0), 1),
                       bat_speed_mph=None if rng.random() < 0.08 else round(rng.gauss(bs, 3.0), 1))
        row.update(raw_description=raw, pa_result=result, outcome=map_outcome(raw, result))
        rows.append(row)
        if result:
            return rows, None, result, clock


def make_demo(out_dir, seed=20250610):
    import bench  # base-state reducers, used to replay the synthetic game prefix consistently

    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pitchers = {pid: {"id": pid, "throws": t, "stuff": s, "arsenal": a} for pid, (t, s, a) in DEMO_PITCHERS.items()}
    batters = {bid: {"id": bid, "bats": h, "power": pw, "swing": sw} for bid, (h, pw, sw) in DEMO_BATTERS.items()}
    league_p = {f"league_p_{i:02d}": {"id": f"league_p_{i:02d}", "throws": "R" if i % 3 else "L",
                                      "stuff": 0.9 + 0.03 * i,
                                      "arsenal": {"FF": (0.5, 93.0 + i * 0.3, 15.0, 8.0), "SL": (0.3, 85.0, 2.0, -5.0),
                                                  "CH": (0.2, 85.5, 7.0, 13.0)}} for i in range(1, 9)}
    league_b = {f"league_b_{i:02d}": {"id": f"league_b_{i:02d}", "bats": "LRS"[i % 3], "power": 0.85 + 0.04 * i,
                                      "swing": (8.0 + i * 0.5, 31.0, 70.0)} for i in range(1, 10)}
    relievers = [p for p in DEMO_PITCHERS if p != DEMO_STARTER]
    pitches, events = [], []
    pa_n = 0

    start, scenario_day = date(2025, 4, 21), date.fromisoformat(DEMO_SCENARIO_DATE)
    days = [start + timedelta(d) for d in range(56)]  # includes days after the scenario (leakage control)
    for day_obj in days:
        day = day_obj.isoformat()
        if day == DEMO_SCENARIO_DATE:
            continue
        gid, clock = f"g_{day}", 0
        appearances = []
        if (scenario_day - day_obj).days % 5 == 0:
            appearances.append((DEMO_STARTER, 22))
        healthy = [r for r in relievers if not (r == "reliever_07" and day > "2025-05-25")]
        for r in rng.sample(healthy, 2 if appearances else 3):
            appearances.append((r, rng.randint(3, 5)))
        for pid, n in appearances:
            for _ in range(n):
                pa_n += 1
                b = league_b[rng.choice(sorted(league_b))]
                rows, ev, _, clock = _gen_pa(rng, gid, day, clock, f"pa{pa_n:06d}", pitchers[pid], b)
                pitches += rows
                events += [ev] if ev else []
        for bid in DEMO_BATTERS:  # the opponent's hitters face other clubs' pitchers
            for _ in range(3):
                pa_n += 1
                p = league_p[rng.choice(sorted(league_p))]
                rows, ev, _, clock = _gen_pa(rng, f"o_{day}", day, clock, f"pa{pa_n:06d}", p, batters[bid])
                pitches += rows
                events += [ev] if ev else []

    # Current game: the starter faces the lineup until 16 outs (top 6th, one out), then the branch point.
    order = list(DEMO_BATTERS)
    bases = {"first": None, "second": None, "third": None}
    outs = hits = pa_count = runs = 0
    idx, clock = 0, 0
    while outs < 16:
        pa_n += 1
        bid = order[idx]
        rows, _, result, clock = _gen_pa(rng, DEMO_GAME_ID, DEMO_SCENARIO_DATE, clock, f"pa{pa_n:06d}",
                                         pitchers[DEMO_STARTER], batters[bid], allow_rare=False)
        pitches += rows
        pa_count += 1
        idx = (idx + 1) % 9
        runner = {"player_id": bid, "responsible_pitcher_id": DEMO_STARTER}
        if result == "walk":
            bases, scored = bench.force_walk(bases, runner)
            runs += len(scored)
        elif result in HIT_TYPES:
            hits += 1
            bases, scored = bench.advance_hit(bases, bench.HIT_BASES[result], runner)
            runs += len(scored)
        else:
            outs += 1
            if outs % 3 == 0:
                bases = {"first": None, "second": None, "third": None}
    prefix_pitches = sum(1 for r in pitches if r["game_id"] == DEMO_GAME_ID)
    branch_idx = idx
    branch_ts = _ts(DEMO_SCENARIO_DATE, clock + 1)
    for _ in range(8):  # the rest of the real game exists in the raw file but must never be used
        pa_n += 1
        rows, ev, _, clock = _gen_pa(rng, DEMO_GAME_ID, DEMO_SCENARIO_DATE, clock + 60, f"pa{pa_n:06d}",
                                     pitchers["reliever_04"], batters[order[idx]])
        pitches += rows
        idx = (idx + 1) % 9

    scenario = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "demo_001",
        "synthetic": True,
        "game_id": DEMO_GAME_ID,
        "game_date": DEMO_SCENARIO_DATE,
        "historical_before": _ts(DEMO_SCENARIO_DATE, 0),
        "branch_ts": branch_ts,
        "controlled_team_id": "home_demo",
        "opponent_team_id": "away_demo",
        "controlled_is_home": True,
        "last_defensive_inning": 9,
        "offense_runs_after_inning": {"6": 0, "7": 1, "8": 0},
        "pitcher_pool": [DEMO_STARTER] + relievers,
        "batting_order": order,
        "initial_state": {
            "inning": outs // 3 + 1, "half": "top", "outs_this_half": outs % 3,
            "bases": bases, "score": {"controlled": runs + 1, "opponent": runs},
            "current_pitcher_id": DEMO_STARTER, "used_pitcher_ids": [DEMO_STARTER],
            "next_batter_index": branch_idx,
            "pitcher_appearances": {DEMO_STARTER: {"outs_recorded": outs, "hits_allowed": hits,
                                                   "pitches_thrown": prefix_pitches, "completed_pa": pa_count,
                                                   "entry_event_seq": 0}},
            "defense_totals": {"outs_recorded": outs, "hits_allowed": hits, "pitches_thrown": prefix_pitches,
                               "completed_pa": pa_count},
        },
        "notes": "Synthetic. Prefix bases/score are replayed with the benchmark's deterministic runner rules; "
                 "the controlled team's runs are declared.",
    }

    agg = aggregate(pitches, events, scenario, list(DEMO_PITCHERS), list(DEMO_BATTERS))
    profiles = {"pitchers": {}, "batters": {}}
    for pid, (throws, _, _) in DEMO_PITCHERS.items():
        il_days, il_status, available = DEMO_HEALTH[pid]
        profiles["pitchers"][pid] = {
            "player_id": pid, "display_name": f"Demo {pid.replace('_', ' ').title()}", "throws": throws,
            "usage_health": {
                "last_five_team_game_days": workload_days(pitches, pid, set(DEMO_PITCHERS), DEMO_SCENARIO_DATE),
                "season_outs_before_game": agg["outs_before_game"].get(pid, 0),
                "days_since_last_il_return": il_days, "il_history_status": il_status, "roster_available": available},
            **agg["pitchers"][pid]}
    for bid, (bats, _, _) in DEMO_BATTERS.items():
        profiles["batters"][bid] = {"player_id": bid, "display_name": f"Demo {bid.replace('_', ' ').title()}",
                                    "bats": bats, **agg["batters"][bid]}
    sim = dict(agg["sim"], synthetic=True, provider="pooled_frequency",
               pitcher_throws={p: DEMO_PITCHERS[p][0] for p in DEMO_PITCHERS})

    write_csv(out / "pitches.csv", pitches, PITCH_COLUMNS)
    write_csv(out / "pa_events.csv", events, PA_EVENT_COLUMNS)
    write_json(out / "scenario.json", scenario)
    write_json(out / "profiles.json", profiles)
    write_json(out / "sim_tables.json", sim)
    write_manifest(out, seed, agg)
    return out


def write_manifest(out, seed, agg):
    files = ["pitches.csv", "pa_events.csv", "scenario.json", "profiles.json", "sim_tables.json"]
    manifest = {
        "snapshot_id": "demo-synthetic-0.1",
        "synthetic": True,
        "generator": {"command": "python data.py make-demo", "seed": seed},
        "raw_sources": ["synthetic generator in data.py (no network)"],
        "schema_version": SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "outcome_mapping": dict(RAW_TO_OUTCOME, hit_into_play="hits | in_play_outs by pa_result",
                                hit_by_pitch="excluded (unsupported)"),
        "pitch_registry": PITCH_REGISTRY,
        "horizontal_break_convention": HORIZONTAL_BREAK_CONVENTION,
        "ivb_definition": "induced vertical break, gravity removed, inches",
        "ideal_attack_angle_deg": list(IDEAL_ATTACK_ANGLE_DEG),
        "workload_window": "five most recent completed team game-dates before the scenario date, "
                           "zero-appearance dates included; current game excluded",
        "il_field": "days since most recent IL return (proposed interpretation)",
        "used_pitches": agg["used_pitches"],
        "exclusions": agg["excluded"],
        "files": {f: file_sha256(out / f) for f in files},
    }
    write_json(out / "manifest.json", manifest)


# ---------------------------------------------------------------- validation

def _reject_constant(name):
    raise ValueError(f"non-finite JSON number {name}")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject_constant)


def _close(a, b, path, errors):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            errors.append(f"{path}: keys differ")
            return
        for k in a:
            _close(a[k], b[k], f"{path}.{k}", errors)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        errors.append(f"{path}: {a!r} != {b!r}")


def check_metrics(obj, path, errors):
    if isinstance(obj, dict):
        if obj.keys() == {"mean", "n"}:
            if (obj["n"] == 0) != (obj["mean"] is None):
                errors.append(f"{path}: mean must be null exactly when n == 0")
        elif obj.keys() == {"value", "numerator", "denominator"}:
            if (obj["denominator"] == 0) != (obj["value"] is None):
                errors.append(f"{path}: rate must be null exactly when denominator == 0")
            elif obj["value"] is not None and not 0 <= obj["value"] <= 1:
                errors.append(f"{path}: rate outside [0, 1]")
        for k, v in obj.items():
            check_metrics(v, f"{path}.{k}", errors)
    elif isinstance(obj, int) and not isinstance(obj, bool) and obj < 0:
        errors.append(f"{path}: negative count")


def check_splits(sp, role, path, errors):
    codes = PITCH_REGISTRY["codes"]
    for side in ("left", "right", "both"):
        s = sp[side]
        if s["pitch_count"] != sum(s["outcomes"].values()):
            errors.append(f"{path}.{side}: pitch_count != sum(outcomes)")
        if role == "pitcher":
            if sorted(s["pitch_types"]) != sorted(codes):
                errors.append(f"{path}.{side}: pitch_types must contain every registry code")
            elif s["pitch_count"] != sum(t["count"] for t in s["pitch_types"].values()):
                errors.append(f"{path}.{side}: pitch_count != sum(pitch_type counts)")
    _close(sp["both"], combine(sp["left"], sp["right"]), f"{path}.both", errors)
    check_metrics(sp, path, errors)


def validate(snapshot):
    snapshot = Path(snapshot)
    errors = []
    manifest = load_json(snapshot / "manifest.json")
    for name, digest in manifest["files"].items():
        if file_sha256(snapshot / name) != digest:
            errors.append(f"manifest hash mismatch for {name}")
    scenario = load_json(snapshot / "scenario.json")
    profiles = load_json(snapshot / "profiles.json")
    sim = load_json(snapshot / "sim_tables.json")
    pitches, events = load_records(snapshot)

    for row in pitches:
        if row["outcome"] != map_outcome(row["raw_description"], row["pa_result"]):
            errors.append(f"pitch {row['pa_id']}#{row['pitch_num']}: outcome does not match mapping")
            break
    for pid, prof in profiles["pitchers"].items():
        for w in ("season", "game_prefix"):
            check_splits(prof[w], "pitcher", f"pitchers.{pid}.{w}", errors)
        days = prof["usage_health"]["last_five_team_game_days"]
        if len(days) != 5:
            errors.append(f"pitchers.{pid}: need exactly five workload days")
        if [d["date"] for d in days] != sorted(d["date"] for d in days):
            errors.append(f"pitchers.{pid}: workload days must be oldest first")
        for d in days:
            if d["date"] >= scenario["game_date"]:
                errors.append(f"pitchers.{pid}: workload day {d['date']} is not before the game")
            if d["data_status"] == "known" and (d["outs_recorded"] is None or d["ip_display"] != ip_display(d["outs_recorded"])):
                errors.append(f"pitchers.{pid}: workload day {d['date']} outs/IP inconsistent")
            if d["data_status"] == "unavailable" and d["outs_recorded"] is not None:
                errors.append(f"pitchers.{pid}: unknown workload must be null, not a number")
    for bid, prof in profiles["batters"].items():
        for w in ("season", "game_prefix"):
            check_splits(prof[w], "batter", f"batters.{bid}.{w}", errors)

    # Re-aggregate the raw records under the cutoff rule: proves no post-cutoff rows reached the profiles.
    agg = aggregate(pitches, events, scenario, list(profiles["pitchers"]), list(profiles["batters"]))
    for pid, prof in profiles["pitchers"].items():
        for w in ("season", "game_prefix"):
            _close(prof[w], agg["pitchers"][pid][w], f"pitchers.{pid}.{w} vs raw", errors)
        if prof["usage_health"]["season_outs_before_game"] != agg["outs_before_game"].get(pid, 0):
            errors.append(f"pitchers.{pid}: season outs differ from raw records")
    for bid, prof in profiles["batters"].items():
        for w in ("season", "game_prefix"):
            _close(prof[w], agg["batters"][bid][w], f"batters.{bid}.{w} vs raw", errors)
    for key in ("reference_counts", "hit_type_counts", "pitchers", "batters"):
        _close(sim[key], agg["sim"][key], f"sim.{key} vs raw", errors)
    if scenario["branch_ts"] < scenario["historical_before"]:
        errors.append("branch_ts precedes historical_before")

    init = scenario["initial_state"]
    if not 0 <= init["outs_this_half"] <= 2:
        errors.append("initial outs must be 0-2")
    order = scenario["batting_order"]
    if len(order) != 9 or len(set(order)) != 9 or any(b not in profiles["batters"] for b in order):
        errors.append("batting order must be nine distinct known batters")
    if any(p not in profiles["pitchers"] for p in scenario["pitcher_pool"]):
        errors.append("pitcher pool contains unknown pitchers")
    prefix = profiles["pitchers"][init["current_pitcher_id"]]["game_prefix"]["both"]
    app = init["pitcher_appearances"][init["current_pitcher_id"]]
    if (prefix["pitch_count"], prefix["completed_pa"]) != (app["pitches_thrown"], app["completed_pa"]):
        errors.append("starter appearance does not match game-prefix records")
    for k in ("hits", "in_play_outs", "balls", "strikes"):
        if sim["reference_counts"][k] <= 0:
            errors.append(f"reference distribution needs positive support for {k}")
    return errors, {"pitch_rows": len(pitches), "pa_events": len(events), "used_pitches": agg["used_pitches"],
                    "excluded": agg["excluded"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-demo", help="generate the synthetic demo snapshot")
    mk.add_argument("--out", default="data/demo")
    mk.add_argument("--seed", type=int, default=20250610)
    va = sub.add_parser("validate", help="check schema identities, cutoff safety, and manifest hashes")
    va.add_argument("--snapshot", default="data/demo")
    args = ap.parse_args(argv)
    if args.cmd == "make-demo":
        print(f"wrote {make_demo(args.out, args.seed)}")
        return 0
    errors, info = validate(args.snapshot)
    print(json.dumps(info, indent=1))
    for e in errors[:50]:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
