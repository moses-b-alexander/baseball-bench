"""BaseballBench engine: state, rules, pitch/PA simulation, tools, episode loop, save/resume, summary.

Decisions happen between plate appearances; pitches are simulated underneath. See DESIGN.md.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import data

SCHEMA_VERSION = "0.3"
RULESET_VERSION = "rules-0.1"
PROVIDER_VERSION = "pooled_frequency-0.1"
OUTCOMES = data.OUTCOMES
HIT_TYPES = data.HIT_TYPES
HIT_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
BASES = ("first", "second", "third")
ACTIONS = {"continue": (), "intentional_walk": (), "switch_pitcher": ("pitcher_id",),
           "check_at_bat": ("pitcher_id", "batter_id")}
OPTIONAL_FIELDS = ("reason", "notes")
CHECKPOINT, EVENTS, DECISIONS, SUMMARY = "checkpoint.json", "events.jsonl", "decisions.jsonl", "summary.json"


class InvalidAction(ValueError):
    pass


# ---------------------------------------------------------------- loading

def load_config(path):
    cfg = data.load_json(path)
    check_config(cfg)
    return cfg


def check_config(cfg):
    """Refuse configurations whose rules are not implemented rather than silently ignoring them."""
    rules, probs = cfg["rules"], cfg["probabilities"]
    unsupported = [name for name, on in (
        ("rules.steals_enabled", rules["steals_enabled"]),
        ("rules.double_plays_enabled", rules["double_plays_enabled"]),
        ("rules.enforce_three_batter_minimum", rules["enforce_three_batter_minimum"]),
        ("probabilities.fatigue_enabled", probs["fatigue_enabled"]),
        ("lookup.repeat_paid_query_costs_pa=false", not cfg["lookup"]["repeat_paid_query_costs_pa"]),
    ) if on]
    if rules["advancement"] != "deterministic_bases":
        unsupported.append(f"rules.advancement={rules['advancement']}")
    if probs["provider"] != "pooled_frequency":
        unsupported.append(f"probabilities.provider={probs['provider']}")
    if unsupported:
        raise ValueError("not implemented in this ruleset: " + ", ".join(unsupported))


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_snapshot(path):
    path = Path(path)
    snap = {k: data.load_json(path / f) for k, f in (
        ("scenario", "scenario.json"), ("profiles", "profiles.json"), ("sim", "sim_tables.json"),
        ("manifest", "manifest.json"))}
    for k in ("hits", "in_play_outs", "balls", "strikes"):
        if snap["sim"]["reference_counts"][k] <= 0:
            raise ValueError(f"reference distribution needs positive support for {k}")
    return snap


def make_env(snapshot, config, forced=None):
    """Static episode inputs. `forced` maps draw kinds to queues of outcomes (tests only)."""
    return dict(snapshot, config=config, forced=forced)


# ---------------------------------------------------------------- state

def init_state(env):
    sc, init = env["scenario"], env["scenario"]["initial_state"]
    prof = env["profiles"]
    return {
        "scenario_id": sc["scenario_id"],
        "controlled_team_id": sc["controlled_team_id"],
        "opponent_team_id": sc["opponent_team_id"],
        "inning": init["inning"],
        "half": init["half"],
        "outs_this_half": init["outs_this_half"],
        "balls": 0,
        "strikes": 0,
        "bases": copy.deepcopy(init["bases"]),
        "score": dict(init["score"]),
        "current_pitcher_id": init["current_pitcher_id"],
        "pitcher_appearances": copy.deepcopy(init["pitcher_appearances"]),
        "used_pitcher_ids": list(init["used_pitcher_ids"]),
        "batting_order": list(sc["batting_order"]),
        "next_batter_index": init["next_batter_index"],
        "defense_totals": dict(init["defense_totals"]),
        "substitutions_at_boundary": 0,
        "free_lookups_at_boundary": 0,
        "event_seq": 0,
        "pa_seq": 0,
        "lookup_seq": 0,
        "decisions_made": 0,
        "status": "active",
        "termination_reason": None,
        # Current-game splits: real prefix plus simulated events. Season-before-game stays in env.
        "game_splits": {"pitchers": {p: copy.deepcopy(v["game_prefix"]) for p, v in prof["pitchers"].items()},
                        "batters": {b: copy.deepcopy(v["game_prefix"]) for b, v in prof["batters"].items()}},
    }


def emit(s, events, kind, **fields):
    s["event_seq"] += 1
    ev = {"event_id": f"{s['scenario_id']}-e{s['event_seq']:06d}", "seq": s["event_seq"], "type": kind,
          "inning": s["inning"], "half": s["half"], **fields}
    events.append(ev)
    return ev


def end_episode(s, events, status, reason):
    s["status"], s["termination_reason"] = status, reason
    emit(s, events, "episode_end", status=status, reason=reason, score=dict(s["score"]))


def throws_of(env, pid):
    return env["profiles"]["pitchers"][pid]["throws"]


def bats_of(env, bid):
    return env["profiles"]["batters"][bid]["bats"]


def matchup_sides(env, pid, bid):
    """(pitcher-profile split = batter's side, batter-profile split = pitcher's hand)."""
    throws = throws_of(env, pid)
    return data.HAND_SIDE[data.resolve_side(bats_of(env, bid), throws)], data.HAND_SIDE[throws]


def exposure(rec):
    return rec["outs_recorded"] + rec["hits_allowed"]


# ---------------------------------------------------------------- probabilities

def draw(env, rng, kind, options, weights):
    queue = (env.get("forced") or {}).get(kind)
    if queue:
        return queue.pop(0)
    return rng.choices(options, weights=weights)[0]


def _sum_counts(tables):
    total = dict.fromkeys(OUTCOMES, 0)
    for t in tables:
        for k in OUTCOMES:
            total[k] += t.get(k, 0)
    return total


def _first_nonempty(levels):
    for name, counts in levels:
        if sum(counts.values()) > 0:
            return name, counts
    return "reference", dict.fromkeys(OUTCOMES, 0)


def pitch_type_weights(env, pid, bat_side):
    table = env["sim"]["pitchers"][pid]
    for level, sides in (("side", [bat_side]), ("overall", ["left", "right"])):
        w = Counter()
        for side in sides:
            for code, counts in table[side].items():
                w[code] += sum(counts.values())
        if sum(w.values()) > 0:
            codes = sorted(c for c in w if w[c] > 0)
            return codes, [w[c] for c in codes], level
    return ["UNKNOWN"], [1], "reference"


def outcome_probs(env, pid, bid, bat_side, throw_side, ptype):
    """Equal-weight pooling of smoothed pitcher and batter frequencies (proposed baseline rule)."""
    cfg, sim = env["config"]["probabilities"], env["sim"]
    alpha, w = cfg["prior_strength"], cfg["pitcher_weight"]
    ref = sim["reference_counts"]
    n_ref = sum(ref.values())
    q = {k: ref[k] / n_ref for k in OUTCOMES}
    pt, bt = sim["pitchers"][pid], sim["batters"][bid]
    plevel, pc = _first_nonempty([
        ("pitch_type_side", _sum_counts([pt[bat_side].get(ptype, {})])),
        ("side", _sum_counts(pt[bat_side].values())),
        ("overall", _sum_counts(list(pt["left"].values()) + list(pt["right"].values())))])
    blevel, bc = _first_nonempty([("hand", bt[throw_side]), ("overall", _sum_counts([bt["left"], bt["right"]]))])

    def smooth(c):
        n = sum(c.values())
        return [(c[k] + alpha * q[k]) / (n + alpha) for k in OUTCOMES]

    probs = [w * a + (1 - w) * b for a, b in zip(smooth(pc), smooth(bc))]
    return probs, {"pitcher": plevel, "batter": blevel}


# ---------------------------------------------------------------- reducers

def apply_count(balls, strikes, outcome):
    """Count transition for one pitch -> (balls, strikes, terminal result or None). Hits resolve separately."""
    if outcome == "balls":
        return (balls, strikes, "walk") if balls == 3 else (balls + 1, strikes, None)
    if outcome == "strikes":
        return (balls, strikes, "strikeout") if strikes == 2 else (balls, strikes + 1, None)
    if outcome == "fouls":
        return balls, min(strikes + 1, 2), None
    if outcome == "hits":
        return balls, strikes, "hit"
    return balls, strikes, "in_play_out"


def force_walk(bases, runner):
    """Advance only forced runners; batter takes first. Returns (bases, scored runners)."""
    b, scored = dict(bases), []
    if b["first"]:
        if b["second"]:
            if b["third"]:
                scored.append(b["third"])
            b["third"] = b["second"]
        b["second"] = b["first"]
    b["first"] = runner
    return b, scored


def advance_hit(bases, n, runner):
    """Every runner and the batter advance n bases (deterministic baseline)."""
    new, scored = dict.fromkeys(BASES), []
    for i in (2, 1, 0):
        r = bases[BASES[i]]
        if r:
            if i + n >= 3:
                scored.append(r)
            else:
                new[BASES[i + n]] = r
    if n >= 4:
        scored.append(runner)
    else:
        new[BASES[n - 1]] = runner
    return new, scored


def _bump_pitch(s, pid, bid, bat_side, throw_side, ptype, outcome):
    s["pitcher_appearances"][pid]["pitches_thrown"] += 1
    s["defense_totals"]["pitches_thrown"] += 1
    for side in (bat_side, "both"):
        sp = s["game_splits"]["pitchers"][pid][side]
        sp["pitch_count"] += 1
        sp["outcomes"][outcome] += 1
        sp["pitch_types"][ptype]["count"] += 1  # no shape measurement is fabricated for simulated pitches
    for side in (throw_side, "both"):
        sp = s["game_splits"]["batters"][bid][side]
        sp["pitch_count"] += 1
        sp["outcomes"][outcome] += 1


def _bump_pa(s, pid, bid, bat_side, throw_side, result):
    s["pitcher_appearances"][pid]["completed_pa"] += 1
    s["defense_totals"]["completed_pa"] += 1
    for role, pl, side in (("pitchers", pid, bat_side), ("batters", bid, throw_side)):
        for sd in (side, "both"):
            data.add_pa(s["game_splits"][role][pl][sd], result)


def complete_pa(s, env, events, pid, bid, result, pitches, pa_id, bases_before):
    bat_side, throw_side = matchup_sides(env, pid, bid)
    runner = {"player_id": bid, "responsible_pitcher_id": pid}
    app, tot = s["pitcher_appearances"][pid], s["defense_totals"]
    scored = []
    if result in ("walk", "intentional_walk"):
        s["bases"], scored = force_walk(s["bases"], runner)
    elif result in HIT_BASES:
        s["bases"], scored = advance_hit(s["bases"], HIT_BASES[result], runner)
        app["hits_allowed"] += 1
        tot["hits_allowed"] += 1
    else:  # strikeout or basic in-play out: one out, runners hold
        s["outs_this_half"] += 1
        app["outs_recorded"] += 1
        tot["outs_recorded"] += 1
    s["score"]["opponent"] += len(scored)
    _bump_pa(s, pid, bid, bat_side, throw_side, result)
    s["next_batter_index"] = (s["next_batter_index"] + 1) % len(s["batting_order"])
    s["balls"] = s["strikes"] = 0
    s["substitutions_at_boundary"] = s["free_lookups_at_boundary"] = 0
    emit(s, events, "pa", pa_id=pa_id, pitcher_id=pid, batter_id=bid, result=result, pitches=pitches,
         bases_before=bases_before, runs_scored=scored, outs_after=s["outs_this_half"],
         bases_after=copy.deepcopy(s["bases"]), score_after=dict(s["score"]))
    if s["outs_this_half"] == 3:
        finish_half(s, env, events)


def finish_half(s, env, events):
    sc = env["scenario"]
    left = sum(1 for b in BASES if s["bases"][b])
    emit(s, events, "half_end", left_on_base=left, score=dict(s["score"]))
    s["bases"] = dict.fromkeys(BASES)
    s["outs_this_half"] = s["balls"] = s["strikes"] = 0
    if s["inning"] >= sc["last_defensive_inning"]:
        end_episode(s, events, "completed", "defensive_horizon_reached")
        return
    runs = sc["offense_runs_after_inning"].get(str(s["inning"]), 0)
    s["score"]["controlled"] += runs
    emit(s, events, "offense_runs", runs=runs, score=dict(s["score"]))
    s["inning"] += 1


def simulate_pa(s, env, rng, events):
    """Simulate one actual PA with the incumbent against the scheduled batter. Returns the PA id."""
    pid = s["current_pitcher_id"]
    bid = s["batting_order"][s["next_batter_index"]]
    bat_side, throw_side = matchup_sides(env, pid, bid)
    s["pa_seq"] += 1
    pa_id = f"{s['scenario_id']}-pa{s['pa_seq']:04d}"
    bases_before = copy.deepcopy(s["bases"])
    limit = env["config"]["execution"]["max_pitches_per_pa"]
    pitches = 0
    while True:
        if pitches >= limit:
            end_episode(s, events, "truncated", "max_pitches_per_pa")
            return pa_id
        codes, weights, type_level = pitch_type_weights(env, pid, bat_side)
        ptype = draw(env, rng, "pitch_type", codes, weights)
        probs, levels = outcome_probs(env, pid, bid, bat_side, throw_side, ptype)
        outcome = draw(env, rng, "outcome", list(OUTCOMES), probs)
        pitches += 1
        before = (s["balls"], s["strikes"])
        s["balls"], s["strikes"], result = apply_count(s["balls"], s["strikes"], outcome)
        _bump_pitch(s, pid, bid, bat_side, throw_side, ptype, outcome)
        if result == "hit":
            hw = env["sim"]["hit_type_counts"]
            result = draw(env, rng, "hit_type", list(HIT_TYPES), [hw[h] for h in HIT_TYPES])
        emit(s, events, "pitch", pa_id=pa_id, pitcher_id=pid, batter_id=bid, pitch_type=ptype, outcome=outcome,
             count_before=list(before), count_after=[s["balls"], s["strikes"]],
             fallback=dict(levels, pitch_type=type_level))
        if result:
            complete_pa(s, env, events, pid, bid, result, pitches, pa_id, bases_before)
            return pa_id


# ---------------------------------------------------------------- profiles and observation

def strip_shapes(obj):
    if isinstance(obj, dict):
        return {k: strip_shapes(v) for k, v in obj.items() if k not in ("shape", "swing_profile")}
    if isinstance(obj, list):
        return [strip_shapes(v) for v in obj]
    return obj


def _as_of(s, env):
    return {"historical_before": env["scenario"]["historical_before"], "scenario_id": s["scenario_id"],
            "simulation_event_seq": s["event_seq"]}


def _visible(obj, env):
    return obj if env["config"]["observation"]["show_shapes"] else strip_shapes(obj)


def pitcher_profile(s, env, pid):
    base = env["profiles"]["pitchers"][pid]
    game = s["game_splits"]["pitchers"][pid]
    usage = copy.deepcopy(base["usage_health"])
    app = s["pitcher_appearances"].get(pid)
    outs = usage.pop("season_outs_before_game") + (app["outs_recorded"] if app else 0)
    usage["season_outs_recorded"], usage["season_ip_display"] = outs, data.ip_display(outs)
    return _visible({
        "schema_version": SCHEMA_VERSION, "player_id": pid,
        "display_name": base["display_name"] if env["config"]["observation"]["show_player_names"] else None,
        "throws": base["throws"], "as_of": _as_of(s, env), "usage_health": usage,
        "stats": data.combine(base["season"], game), "game": copy.deepcopy(game)}, env)


def batter_profile(s, env, bid):
    base = env["profiles"]["batters"][bid]
    game = s["game_splits"]["batters"][bid]
    return _visible({
        "schema_version": SCHEMA_VERSION, "player_id": bid,
        "display_name": base["display_name"] if env["config"]["observation"]["show_player_names"] else None,
        "bats": base["bats"], "as_of": _as_of(s, env),
        "stats": data.combine(base["season"], game), "game": copy.deepcopy(game)}, env)


def entry_block(s, env, pid):
    """Reason `pid` cannot enter now, or None if it can."""
    if pid not in env["scenario"]["pitcher_pool"]:
        return "unknown_pitcher"
    if pid == s["current_pitcher_id"]:
        return "currently_pitching"
    if not env["profiles"]["pitchers"][pid]["usage_health"]["roster_available"]:
        return "not_on_active_roster"
    if pid in s["used_pitcher_ids"] and not env["config"]["rules"]["allow_pitcher_reentry"]:
        return "already_used"
    if s["substitutions_at_boundary"] >= env["config"]["rules"]["max_switches_per_boundary"]:
        return "switch_limit_this_boundary"
    return None


def lookup_block(s, env):
    lk = env["config"]["lookup"]
    if lk["cost_pa"] == 0 and s["free_lookups_at_boundary"] >= lk["max_free_per_boundary"]:
        return "free_lookup_limit_this_boundary"
    return None


def legal_actions(s, env):
    pool = env["scenario"]["pitcher_pool"]
    return {"continue": True,
            "intentional_walk": env["config"]["rules"]["intentional_walk_enabled"],
            "switch_pitcher": [p for p in pool if entry_block(s, env, p) is None],
            "check_at_bat": {"allowed": lookup_block(s, env) is None, "pitcher_ids": list(pool),
                             "batter_ids": list(s["batting_order"])}}


def observe(s, env, evidence, notes):
    cfg = env["config"]
    pid = s["current_pitcher_id"]
    app = s["pitcher_appearances"][pid]
    tot = s["defense_totals"]
    lk = cfg["lookup"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": s["scenario_id"],
        "decision_id": f"{s['scenario_id']}-d{s['decisions_made']:04d}",
        "game": {"inning": s["inning"], "half": s["half"], "outs_this_half": s["outs_this_half"],
                 "outs_to_end_inning": 3 - s["outs_this_half"], "count": {"balls": s["balls"], "strikes": s["strikes"]},
                 "bases": copy.deepcopy(s["bases"]), "score": dict(s["score"]),
                 "last_defensive_inning": env["scenario"]["last_defensive_inning"],
                 "defense_totals": dict(tot), "defense_exposure": exposure(tot),
                 "current_pitcher_exposure": exposure(app)},
        "current_pitcher": {"player_id": pid, "throws": throws_of(env, pid), "appearance": dict(app),
                            "game_splits": _visible(copy.deepcopy(s["game_splits"]["pitchers"][pid]), env)},
        "opposing_lineup": [{"player_id": b, "bats": bats_of(env, b), "lineup_position": i + 1}
                            for i, b in enumerate(s["batting_order"])],
        "next_batter_id": s["batting_order"][s["next_batter_index"]],
        "bullpen_overview": [{"player_id": p, "throws": throws_of(env, p),
                              "roster_available": env["profiles"]["pitchers"][p]["usage_health"]["roster_available"],
                              "legal_to_enter_now": entry_block(s, env, p) is None,
                              "unavailable_reason": entry_block(s, env, p)}
                             for p in env["scenario"]["pitcher_pool"] if p != pid],
        "legal_actions": legal_actions(s, env),
        "lookup_rules": {"cost_pa": lk["cost_pa"],
                         "remaining_free_lookups_this_boundary":
                             lk["max_free_per_boundary"] - s["free_lookups_at_boundary"] if lk["cost_pa"] == 0 else 0},
        "acquired_evidence": copy.deepcopy(evidence),
        "agent_notes": notes,
    }


# ---------------------------------------------------------------- actions

def validate_action(action, s, env):
    """Return an error message, or None. Never mutates state or consumes randomness."""
    if s["status"] != "active":
        return "episode is not active"
    if not isinstance(action, dict):
        return "action must be a JSON object"
    name = action.get("action")
    if name not in ACTIONS:
        return f"unknown action {name!r}; expected one of {sorted(ACTIONS)}"
    unknown = sorted(set(action) - {"action", *ACTIONS[name], *OPTIONAL_FIELDS})
    if unknown:
        return f"unknown field(s) for {name}: {unknown}"
    for f in ACTIONS[name]:
        if not isinstance(action.get(f), str):
            return f"{name} requires string field {f!r}"
    obs_cfg = env["config"]["observation"]
    for f, limit in (("reason", obs_cfg["reason_max_chars"]), ("notes", obs_cfg["notes_max_chars"])):
        if f in action and (not isinstance(action[f], str) or len(action[f]) > limit):
            return f"{f} must be a string of at most {limit} characters"
    if name == "switch_pitcher":
        block = entry_block(s, env, action["pitcher_id"])
        if block:
            return f"cannot switch to {action['pitcher_id']!r}: {block}"
    elif name == "check_at_bat":
        if action["pitcher_id"] not in env["scenario"]["pitcher_pool"]:
            return f"unknown pitcher_id {action['pitcher_id']!r}"
        if action["batter_id"] not in s["batting_order"]:
            return f"batter_id {action['batter_id']!r} is not in the opposing lineup"
        block = lookup_block(s, env)
        if block:
            return f"check_at_bat unavailable: {block}"
    elif name == "intentional_walk" and not env["config"]["rules"]["intentional_walk_enabled"]:
        return "intentional_walk is disabled in this condition"
    return None


def intentional_walk(s, env, events):
    """Award first to the scheduled batter: zero pitches, zero random draws, forced runners only."""
    pid = s["current_pitcher_id"]
    bid = s["batting_order"][s["next_batter_index"]]
    s["pa_seq"] += 1
    pa_id = f"{s['scenario_id']}-pa{s['pa_seq']:04d}"
    complete_pa(s, env, events, pid, bid, "intentional_walk", 0, pa_id, copy.deepcopy(s["bases"]))
    return pa_id


def switch_pitcher(s, env, events, new_pid):
    old = s["current_pitcher_id"]
    old_app = s["pitcher_appearances"][old]
    s["current_pitcher_id"] = new_pid
    if new_pid not in s["used_pitcher_ids"]:
        s["used_pitcher_ids"].append(new_pid)
    s["substitutions_at_boundary"] += 1
    ev = emit(s, events, "substitution", from_pitcher=old, to_pitcher=new_pid, outs_this_half=s["outs_this_half"],
              bases=copy.deepcopy(s["bases"]), incumbent_pitches=old_app["pitches_thrown"],
              incumbent_pa=old_app["completed_pa"], incumbent_exposure=exposure(old_app),
              defense_pa=s["defense_totals"]["completed_pa"], decision_index=s["decisions_made"] - 1)
    s["pitcher_appearances"].setdefault(new_pid, {"outs_recorded": 0, "hits_allowed": 0, "pitches_thrown": 0,
                                                  "completed_pa": 0, "entry_event_seq": ev["seq"]})


def check_at_bat(s, env, rng, events, action):
    """Atomic paid lookup: snapshot profiles, then the incumbent pitches cost_pa actual PAs, then deliver."""
    pid, bid = action["pitcher_id"], action["batter_id"]
    cost = env["config"]["lookup"]["cost_pa"]
    s["lookup_seq"] += 1
    receipt = {"lookup_id": f"{s['scenario_id']}-lk{s['lookup_seq']:03d}", "pitcher_id": pid, "batter_id": bid,
               "queried_at_event_seq": s["event_seq"],
               "target_lineup_distance_at_request":
                   (s["batting_order"].index(bid) - s["next_batter_index"]) % len(s["batting_order"]),
               "pitcher": pitcher_profile(s, env, pid), "batter": batter_profile(s, env, bid),
               "cost_pa": cost, "elapsed_pa_ids": []}
    if cost == 0:
        s["free_lookups_at_boundary"] += 1
    for _ in range(cost):
        if s["status"] != "active":
            break
        receipt["elapsed_pa_ids"].append(simulate_pa(s, env, rng, events))
    ev = emit(s, events, "lookup_delivered", lookup_id=receipt["lookup_id"], pitcher_id=pid, batter_id=bid,
              distance=receipt["target_lineup_distance_at_request"], elapsed_pa_ids=receipt["elapsed_pa_ids"],
              episode_active_after=s["status"] == "active")
    receipt["delivered_at_event_seq"] = ev["seq"]
    return receipt


def step(state, action, env, rng):
    """Validate and execute one action on a copy of `state`. Returns (state, events, receipt or None)."""
    err = validate_action(action, state, env)
    if err:
        raise InvalidAction(err)
    s = copy.deepcopy(state)
    events, receipt = [], None
    s["decisions_made"] += 1
    emit(s, events, "action", decision_index=s["decisions_made"] - 1, pitcher_id=s["current_pitcher_id"],
         action={k: v for k, v in action.items() if k != "notes"})
    name = action["action"]
    if name == "continue":
        simulate_pa(s, env, rng, events)
    elif name == "intentional_walk":
        intentional_walk(s, env, events)
    elif name == "switch_pitcher":
        switch_pitcher(s, env, events, action["pitcher_id"])
    else:
        receipt = check_at_bat(s, env, rng, events, action)
    return s, events, receipt


# ---------------------------------------------------------------- episode loop and persistence

def rng_state(rng):
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def set_rng_state(rng, st):
    rng.setstate((st[0], tuple(st[1]), st[2]))


def _atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(rd, state, evidence, notes, rng, pending, cfg):
    payload = {"schema_version": SCHEMA_VERSION, "ruleset_version": RULESET_VERSION,
               "provider_version": PROVIDER_VERSION, "config_hash": config_hash(cfg),
               "committed_event_seq": state["event_seq"], "state": state, "evidence": evidence, "notes": notes,
               "rng_state": rng_state(rng), "pending": pending}
    _atomic_write(rd / CHECKPOINT, json.dumps(payload, allow_nan=False))


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trim_events(path, committed_seq):
    """Drop uncommitted tail events (written after the last checkpoint) and any duplicates."""
    seen, keep = set(), []
    for ev in read_jsonl(path):
        if ev["seq"] <= committed_seq and ev["seq"] not in seen:
            seen.add(ev["seq"])
            keep.append(ev)
    _atomic_write(Path(path), "".join(json.dumps(e, sort_keys=True) + "\n" for e in keep))


def run_episode(env, decide_fn, run_dir=None, resume=False):
    """observe -> one decision -> validate -> execute once -> save, until the episode ends.

    decide_fn(observation, check) -> (action or None, attempt log); check(action) returns an error or None.
    """
    cfg = env["config"]
    rd = Path(run_dir) if run_dir else None
    all_events = []
    if resume:
        ck = data.load_json(rd / CHECKPOINT)
        if ck["config_hash"] != config_hash(cfg):
            raise ValueError("config differs from the checkpoint's; refusing to resume")
        state, evidence, notes, pending = ck["state"], ck["evidence"], ck["notes"], ck["pending"]
        rng = random.Random()
        set_rng_state(rng, ck["rng_state"])
        trim_events(rd / EVENTS, ck["committed_event_seq"])
    else:
        state, evidence, notes, pending = init_state(env), [], "", None
        rng = random.Random(cfg["execution"]["simulator_seed"])
        start = []
        emit(state, start, "episode_start", score=dict(state["score"]), bases=copy.deepcopy(state["bases"]),
             outs_this_half=state["outs_this_half"], current_pitcher_id=state["current_pitcher_id"],
             config_hash=config_hash(cfg), ruleset_version=RULESET_VERSION, provider_version=PROVIDER_VERSION)
        all_events += start
        if rd:
            rd.mkdir(parents=True, exist_ok=True)
            for name in (CHECKPOINT, EVENTS, DECISIONS, SUMMARY):
                (rd / name).unlink(missing_ok=True)
            append_jsonl(rd / EVENTS, start)
            save_checkpoint(rd, state, evidence, notes, rng, None, cfg)

    def commit(new_state, events, new_pending=None):
        all_events.extend(events)
        if rd:
            append_jsonl(rd / EVENTS, events)
            save_checkpoint(rd, new_state, evidence, notes, rng, new_pending, cfg)

    while state["status"] == "active":
        if pending is None:
            if state["decisions_made"] >= cfg["execution"]["max_decisions"]:
                state = copy.deepcopy(state)
                events = []
                end_episode(state, events, "truncated", "max_decisions")
                commit(state, events)
                break
            obs = observe(state, env, evidence, notes)
            current = state
            action, attempts = decide_fn(obs, lambda a: validate_action(a, current, env))
            err = "no valid action" if action is None else validate_action(action, state, env)
            if rd:
                append_jsonl(rd / DECISIONS, [{"decision_id": obs["decision_id"], "attempts": attempts,
                                               "accepted": err is None, "action": action}])
            if err:
                state = copy.deepcopy(state)
                events = []
                end_episode(state, events, "failed", f"invalid_action_after_repair: {err}")
                commit(state, events)
                break
            # Capture the accepted decision before executing it, so a resume never resamples it.
            pending = {"action_key": obs["decision_id"], "action": action}
            if rd:
                save_checkpoint(rd, state, evidence, notes, rng, pending, cfg)
        state, events, receipt = step(state, pending["action"], env, rng)
        if receipt:
            evidence.append(receipt)
        if "notes" in pending["action"]:
            notes = pending["action"]["notes"]
        pending = None
        commit(state, events)
    return {"state": state, "evidence": evidence, "notes": notes,
            "events": read_jsonl(rd / EVENTS) if rd else all_events}


# ---------------------------------------------------------------- summary

def summarize(events, state, evidence, decisions=None):
    start = next(e for e in events if e["type"] == "episode_start")
    pas = [e for e in events if e["type"] == "pa"]
    results = Counter(e["result"] for e in pas)
    subs = [e for e in events if e["type"] == "substitution"]
    lookups = [e for e in events if e["type"] == "lookup_delivered"]
    charged = Counter(r["responsible_pitcher_id"] for e in pas for r in e["runs_scored"])
    if subs:
        s0 = subs[0]
        first = {k: s0[k] for k in ("decision_index", "inning", "outs_this_half", "from_pitcher", "to_pitcher",
                                    "incumbent_pitches", "incumbent_pa", "incumbent_exposure", "defense_pa")}
        paid_before = sum(1 for e in lookups if e["seq"] < s0["seq"] and e["elapsed_pa_ids"])
    else:
        first, paid_before = {"censored": True, "reason": "never replaced before episode end"}, len(
            [e for e in lookups if e["elapsed_pa_ids"]])
    pairs = Counter((r["pitcher_id"], r["batter_id"]) for r in evidence)
    out = {
        "status": state["status"], "termination_reason": state["termination_reason"],
        "decisions": state["decisions_made"],
        "final_score": state["score"],
        "runs_allowed": state["score"]["opponent"] - start["score"]["opponent"],
        "runs_charged_by_pitcher": dict(charged),
        "plate_appearances": len(pas),
        "pitches": sum(1 for e in events if e["type"] == "pitch"),
        "results": dict(results),
        "hits_allowed": sum(results[h] for h in HIT_TYPES),
        "walks": results["walk"] + results["intentional_walk"],
        "intentional_walks": [{"inning": e["inning"], "batter_id": e["batter_id"], "bases_before": e["bases_before"],
                               "forced_runs": len(e["runs_scored"])} for e in pas if e["result"] == "intentional_walk"],
        "strikeouts": results["strikeout"],
        "replacements": [{"to": e["to_pitcher"], "inning": e["inning"], "outs": e["outs_this_half"]} for e in subs],
        "first_replacement": first,
        "lookups": {"count": len(evidence),
                    "unique_pitchers": sorted({r["pitcher_id"] for r in evidence}),
                    "unique_batters": sorted({r["batter_id"] for r in evidence}),
                    "lineup_distances": [r["target_lineup_distance_at_request"] for r in evidence],
                    "repeated_queries": sum(n - 1 for n in pairs.values()),
                    "no_later_action_opportunity": sum(1 for e in lookups if not e["episode_active_after"]),
                    "paid_lookups_before_first_replacement": paid_before},
    }
    if decisions is not None:
        last = {d["decision_id"]: d for d in decisions}  # a crash can duplicate a log line; keep the last
        attempts = [a for d in last.values() for a in d["attempts"]]
        model = [a for a in attempts if a.get("kind") == "model"]
        out["reliability"] = {
            "decisions_logged": len(last),
            "model_attempts": len(model),
            "invalid_attempts": sum(1 for a in model if a.get("error")),
            "infra_errors": sum(1 for a in attempts if a.get("kind") == "infra_error"),
            "prompt_tokens": sum((a.get("usage") or {}).get("prompt_tokens", 0) for a in model),
            "completion_tokens": sum((a.get("usage") or {}).get("completion_tokens", 0) for a in model),
            "latency_s": round(sum(a.get("latency_s", 0) for a in model), 3)}
    return out


def summarize_run(run_dir):
    rd = Path(run_dir)
    ck = data.load_json(rd / CHECKPOINT)
    summary = summarize(read_jsonl(rd / EVENTS), ck["state"], ck["evidence"], read_jsonl(rd / DECISIONS))
    _atomic_write(rd / SUMMARY, json.dumps(summary, indent=1, sort_keys=True))
    return summary


def print_summary(sm):
    fr = sm["first_replacement"]
    print(f"status: {sm['status']} ({sm['termination_reason']}) after {sm['decisions']} decisions")
    print(f"score: controlled {sm['final_score']['controlled']} - opponent {sm['final_score']['opponent']}"
          f"  (runs allowed this episode: {sm['runs_allowed']})")
    print(f"PA {sm['plate_appearances']}, pitches {sm['pitches']}, hits {sm['hits_allowed']}, "
          f"walks {sm['walks']} (IBB {len(sm['intentional_walks'])}), K {sm['strikeouts']}")
    print("first replacement: " + ("censored (never replaced)" if fr.get("censored") else
          f"{fr['from_pitcher']} -> {fr['to_pitcher']} in inning {fr['inning']}, {fr['outs_this_half']} out, "
          f"after {fr['incumbent_pitches']} pitches / {fr['incumbent_pa']} PA"))
    lk = sm["lookups"]
    print(f"lookups: {lk['count']} (pitchers {lk['unique_pitchers']}, batters {lk['unique_batters']}, "
          f"distances {lk['lineup_distances']})")
    if "reliability" in sm:
        print(f"reliability: {sm['reliability']}")


# ---------------------------------------------------------------- CLI

def main(argv=None):
    import agent  # agent imports bench; import lazily to keep the module graph acyclic

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config.json")
        p.add_argument("--snapshot", default=None, help="defaults to config data.snapshot")
        p.add_argument("--run-dir", default="runs/smoke" if name == "smoke" else "runs/example")
        if name == "run":
            p.add_argument("--agent", default="threshold", choices=sorted(agent.POLICIES) + ["model"])
            p.add_argument("--resume", action="store_true")
    sm = sub.add_parser("summarize")
    sm.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "summarize":
        summary = summarize_run(args.run_dir)
    else:
        cfg = load_config(args.config)
        env = make_env(load_snapshot(args.snapshot or cfg["data"]["snapshot"]), cfg)
        kind = "tour" if args.cmd == "smoke" else args.agent
        decide = agent.make_model_decide(cfg) if kind == "model" else agent.POLICIES[kind](cfg, env)
        run_episode(env, decide, args.run_dir, resume=getattr(args, "resume", False))
        summary = summarize_run(args.run_dir)
        print(f"agent: {kind}; run dir: {args.run_dir}")
    print_summary(summary)
    return 0 if summary["status"] in ("completed", "truncated") else 1


if __name__ == "__main__":
    sys.exit(main())
