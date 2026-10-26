"""Agent side: common briefing, observation formatting, one model call, response parsing,
scripted baselines, and comprehension probes.

The model adapter speaks the OpenAI-compatible /chat/completions protocol over the standard
library (Ollama, vLLM, llama.cpp server, LM Studio, and hosted open-weight providers all serve it).
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import bench
import data

BRIEFING_VERSION = "briefing-0.1"

BRIEFING = """You manage the defense of one baseball team in a simplified benchmark. You decide only between plate
appearances (PAs); pitches are simulated underneath. All players are anonymized.

RULES OF THIS BENCHMARK (simplified; not the full MLB rulebook)
- You control your team's pitching while the opponent bats. Between your defensive half-innings your team's
  offense scores a fixed, pre-declared number of runs. The episode ends after your last defensive half-inning.
- Each pitch is one of: ball, strike (called or swinging miss), foul, hit, or ball in play for an out.
  Four balls = walk; three strikes = strikeout; a foul with two strikes leaves the count unchanged.
- Runners: walks force only runners who must move; a single moves every runner one base, a double two,
  a triple scores all runners, a home run scores everyone. Outs in play: one out, runners hold.
  No steals, no double plays. Three outs clear the bases.
- A pitcher who leaves cannot return. At most one pitching change per PA boundary. Relievers marked
  roster_available=false cannot enter.

YOUR ACTIONS (exactly one per turn)
- continue: the current pitcher faces the next scheduled batter for one PA.
- switch_pitcher(pitcher_id): the new pitcher takes over immediately; no PA elapses. Bases, score,
  batting order and runner responsibility are unchanged. You then decide again at the same boundary.
- intentional_walk: the next scheduled batter goes to first immediately with zero pitches; forced runners
  advance (with the bases loaded, one run scores). Then you decide again against the following batter.
- check_at_bat(pitcher_id, batter_id): returns the historical and current-game profiles of that pitcher and
  that batter. COST: before you see the result, your CURRENT pitcher must face the ACTUAL next scheduled
  batter for one full PA. The batter you look up may be farther down the lineup; looking that batter up does not
  change who bats next. The cost cannot be cancelled. Profiles are separate: once you have seen pitcher A
  and batters X and Y, you can compare A with either without another lookup. Previous receipts stay in
  acquired_evidence for free; they are snapshots taken when requested, not live updates.

READING THE DATA
- Rates are fractions in [0, 1]. "strikes" counts called and swinging strikes only; fouls and balls in play
  are separate. Strikeouts and walks are PA results, not pitch outcomes.
- Splits: in a pitcher profile, left/right is the batter's side in that matchup; in a batter profile it is
  the pitcher's throwing hand. A switch hitter bats left against a right-handed pitcher and right against a
  left-handed pitcher. both = left + right.
- Always read sample sizes: a mean with n = 0 is null (unknown), which is different from an observed 0.
  Small samples are noisy; a few current-game PAs say less than a season of data.
- stats = season before this game + this game so far; game is a subset of stats, never add them together.
- Innings notation: 5.2 IP means 5 innings and 2 outs = 17 outs (.1 and .2 are thirds, not decimals).
- Workload lists the five most recent team game-dates before today (zero = no appearance that day).
- Pitch shape (velocity, induced vertical break, horizontal break from the catcher's view, positive toward
  first base) and swing shape (attack angle, swing-path tilt, bat speed) describe how players pitch and swing.
  Baseball relationships such as platoon advantage, fatigue, or shape matchups are tendencies, not laws, and
  this simulator may not implement them directly.
- exposure = outs recorded + hits allowed; it is not the PA count (walks add zero).
"""

ACTION_FORMAT = """
RESPONSE FORMAT
Reply with exactly one JSON object and nothing else, for example:
{"action": "continue", "reason": "short optional reason", "notes": "optional notes for your future self"}
{"action": "switch_pitcher", "pitcher_id": "<id>"}
{"action": "intentional_walk"}
{"action": "check_at_bat", "pitcher_id": "<id>", "batter_id": "<id>"}
Allowed fields only. reason <= 400 characters; notes <= 1500 characters (notes replace your previous notes
and are shown back to you; omit notes to keep them). Do not include step-by-step reasoning.
"""

SYSTEM_PROMPT = BRIEFING + ACTION_FORMAT


def format_observation(obs):
    return json.dumps(obs, sort_keys=True, separators=(",", ":"))


def build_messages(obs):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Decision {obs['decision_id']}. Observation:\n{format_observation(obs)}"}]


THINK = re.compile(r"<think>.*?</think>", re.S)
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_action(text):
    """Extract exactly one JSON object from a reply -> (object, None) or (None, error)."""
    t = THINK.sub("", text or "").strip()
    m = FENCE.match(t)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"response is not exactly one JSON object ({e.msg})"
    if not isinstance(obj, dict):
        return None, "response must be a JSON object"
    return obj, None


# ---------------------------------------------------------------- model adapter

def http_transport(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def call_model(messages, mcfg, seed=None, transport=http_transport):
    if not mcfg.get("model"):
        raise ValueError("set model.model in the config before using the model agent")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(mcfg["api_key_env"]) if mcfg.get("api_key_env") else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": mcfg["model"], "messages": messages, "temperature": mcfg["temperature"],
               "max_tokens": mcfg["max_tokens"]}
    if seed is not None:
        payload["seed"] = seed  # backend-dependent; logged, never assumed deterministic
    if mcfg.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    t0 = time.monotonic()
    resp = transport(mcfg["base_url"].rstrip("/") + "/chat/completions", payload, headers, mcfg["timeout_s"])
    return resp["choices"][0]["message"]["content"], resp.get("usage") or {}, time.monotonic() - t0


def make_model_decide(cfg, transport=http_transport):
    mcfg = cfg["model"]
    repairs = cfg["execution"]["max_repair_attempts"]
    seed = cfg["execution"]["model_seed"]

    def decide(obs, check):
        messages, attempts = build_messages(obs), []
        for _ in range(repairs + 1):
            text = None
            for _ in range(mcfg.get("infra_retries", 2) + 1):  # same request; nothing has been committed yet
                try:
                    text, usage, latency = call_model(messages, mcfg, seed, transport)
                    break
                except (OSError, ValueError, KeyError, IndexError) as e:
                    attempts.append({"kind": "infra_error", "error": f"{type(e).__name__}: {e}"})
            if text is None:
                return None, attempts
            action, err = parse_action(text)
            if err is None:
                err = check(action)
            attempts.append({"kind": "model", "raw": text, "usage": usage, "latency_s": round(latency, 3),
                             "error": err})
            if err is None:
                return action, attempts
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Rejected: {err}. Nothing happened in the game. "
                                            "Reply with exactly one valid JSON action object."}]
        return None, attempts

    return decide


def model_metadata(cfg):
    m = cfg["model"]
    return {k: m.get(k) for k in ("base_url", "model", "revision", "quantization", "temperature", "max_tokens")}


# ---------------------------------------------------------------- scripted baselines (same observation as models)

def _scripted(fn):
    def decide(obs, check):
        action = fn(obs)
        return action, [{"kind": "scripted", "action": action}]
    return decide


def never_switch(cfg, env):
    return _scripted(lambda obs: {"action": "continue"})


def pitch_count_threshold(cfg, env):
    """Replace when the incumbent reaches a pitch count; take the first legal reliever in pool order."""
    base = cfg["baselines"]
    starter = env["scenario"]["initial_state"]["current_pitcher_id"]

    def fn(obs):
        cur = obs["current_pitcher"]
        limit = base["starter_pitch_threshold"] if cur["player_id"] == starter else base["reliever_pitch_threshold"]
        options = obs["legal_actions"]["switch_pitcher"]
        if cur["appearance"]["pitches_thrown"] >= limit and options:
            return {"action": "switch_pitcher", "pitcher_id": options[0], "reason": f"pitch count >= {limit}"}
        return {"action": "continue"}
    return _scripted(fn)


def random_legal(cfg, env):
    rng = random.Random(cfg["execution"]["model_seed"])

    def fn(obs):
        legal = obs["legal_actions"]
        options = [{"action": "continue"}]
        if legal["intentional_walk"]:
            options.append({"action": "intentional_walk"})
        options += [{"action": "switch_pitcher", "pitcher_id": p} for p in legal["switch_pitcher"]]
        if legal["check_at_bat"]["allowed"]:
            options.append({"action": "check_at_bat", "pitcher_id": rng.choice(legal["check_at_bat"]["pitcher_ids"]),
                            "batter_id": rng.choice(legal["check_at_bat"]["batter_ids"])})
        return rng.choice(options)
    return _scripted(fn)


def tour(cfg, env):
    """Smoke-test policy: exercise lookup, intentional walk, and switch once, then the threshold rule."""
    threshold = pitch_count_threshold(cfg, env)
    plan = []

    def decide(obs, check):
        i = int(obs["decision_id"].rsplit("-d", 1)[1])
        if i == 0:
            reliever = obs["legal_actions"]["switch_pitcher"][0]
            lineup = [b["player_id"] for b in obs["opposing_lineup"]]
            later = lineup[(lineup.index(obs["next_batter_id"]) + 2) % len(lineup)]
            plan[:] = [
                {"action": "check_at_bat", "pitcher_id": reliever, "batter_id": later,
                 "reason": "scripted: look two hitters ahead", "notes": f"looked up {reliever} vs {later}"},
                {"action": "intentional_walk", "reason": "scripted: exercise the intentional walk"},
                {"action": "switch_pitcher", "pitcher_id": reliever, "reason": "scripted: use the looked-up reliever"}]
        if i < len(plan):
            return plan[i], [{"kind": "scripted", "action": plan[i]}]
        return threshold(obs, check)
    return decide


def fixed_sequence(actions):
    """Test helper: play the given actions in order, then continue."""
    queue = list(actions)
    return _scripted(lambda obs: queue.pop(0) if queue else {"action": "continue"})


POLICIES = {"never_switch": never_switch, "threshold": pitch_count_threshold, "random": random_legal, "tour": tour}


# ---------------------------------------------------------------- comprehension probes

PROBE_FORMAT = 'Answer with exactly one JSON object: {"answer": "<answer>"}.'


def make_probes(seed=7, n=3):
    """Held-out factual/procedural probes with fresh numbers; ground truth comes from engine/data helpers."""
    rng = random.Random(seed)
    probes = []

    def add(kind, question, answer):
        probes.append({"id": f"{kind}-{len(probes):03d}", "kind": kind, "question": question, "answer": str(answer)})

    for _ in range(n):
        outs = rng.randint(4, 60)
        add("ip_to_outs", f"A pitcher has thrown {data.ip_display(outs)} innings. How many outs is that? "
                          "Answer with an integer.", outs)
        outs = rng.randint(4, 60)
        add("outs_to_ip", f"A pitcher has recorded {outs} outs. Write the innings pitched in standard "
                          "baseball notation (e.g. 4.1).", data.ip_display(outs))
        den = rng.randint(20, 300)
        num = rng.randint(0, den // 3)
        add("rate", f"A split shows {num} hits in {den} pitches. What fraction of pitches were hits? "
                    "Answer as a decimal rounded to three places.", f"{num / den:.3f}")
        seq = [rng.choice(data.OUTCOMES[:3]) for _ in range(rng.randint(3, 8))]
        balls = strikes = 0
        result = "in progress"
        for i, o in enumerate(seq):
            balls, strikes, r = bench.apply_count(balls, strikes, o)
            if r:
                result, seq = r, seq[:i + 1]
                break
        answer = result if result != "in progress" else f"{balls}-{strikes}"
        add("count", "Starting from a 0-0 count, the pitches are: " + ", ".join(seq) +
            ". If the PA ended, answer 'walk' or 'strikeout'; otherwise answer the count as balls-strikes.", answer)
        lineup = [f"batter_{i:02d}" for i in rng.sample(range(10, 99), 9)]
        nxt, d = rng.randrange(9), rng.randint(1, 8)
        add("lookup_timing", f"The opposing lineup in order is {', '.join(lineup)}. {lineup[nxt]} is due up next. "
                             f"You call check_at_bat for a reliever and {lineup[(nxt + d) % 9]}. Which batter does "
                             "your current pitcher face during the lookup? Answer with the batter id.", lineup[nxt])
    add("null_vs_zero", 'A metric reads {"mean": null, "n": 0}. Does this mean the observed value was zero? '
                        "Answer yes or no.", "no")
    add("split_pitcher", "In a right-handed pitcher's profile, which split (left or right) describes results "
                         "against a switch hitter?", "left")
    add("split_batter", "In a left-handed batter's profile, which split (left or right) describes results "
                        "against a right-handed pitcher?", "right")
    add("legal_switch", "You already changed pitchers at this PA boundary and the limit is one per boundary. "
                        "Is switch_pitcher legal right now? Answer yes or no.", "no")
    add("legal_ibb", "The bases are loaded with one out. Is intentional_walk a legal action? Answer yes or no.", "yes")
    add("ibb_runs", "The bases are loaded and you issue an intentional walk. How many runs score? "
                    "Answer with an integer.", 1)
    return probes


def _norm(s):
    return str(s).strip().lower()


def run_probes(cfg, probes, transport=http_transport):
    results = []
    for p in probes:
        messages = [{"role": "system", "content": BRIEFING + "\n" + PROBE_FORMAT},
                    {"role": "user", "content": f"Probe {p['id']}: {p['question']}"}]
        try:
            text, usage, latency = call_model(messages, cfg["model"], cfg["execution"]["model_seed"], transport)
            obj, err = parse_action(text)
            got = obj.get("answer") if obj else None
        except (OSError, ValueError, KeyError, IndexError) as e:
            text, err, got = None, f"{type(e).__name__}: {e}", None
        results.append({"id": p["id"], "kind": p["kind"], "expected": p["answer"], "got": got,
                        "correct": got is not None and _norm(got) == _norm(p["answer"]), "error": err, "raw": text})
    return results


def oracle_transport(probes):
    """Mock backend that answers probes correctly: checks the harness and scoring, not any model."""
    answers = {p["id"]: p["answer"] for p in probes}

    def transport(url, payload, headers, timeout):
        pid = re.search(r"Probe (\S+):", payload["messages"][-1]["content"]).group(1)
        return {"choices": [{"message": {"content": json.dumps({"answer": answers[pid]})}}], "usage": {}}
    return transport


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probes", help="run held-out comprehension probes")
    pr.add_argument("--config", default="config.json")
    pr.add_argument("--mock", action="store_true", help="use an oracle mock backend (no network)")
    pr.add_argument("--seed", type=int, default=7)
    pr.add_argument("--out", default=None)
    sub.add_parser("briefing", help="print the common briefing")
    args = ap.parse_args(argv)
    if args.cmd == "briefing":
        print(f"[{BRIEFING_VERSION}]\n{SYSTEM_PROMPT}")
        return 0
    cfg = bench.load_config(args.config)
    probes = make_probes(args.seed)
    if args.mock:
        cfg = dict(cfg, model=dict(cfg["model"], model=cfg["model"].get("model") or "mock-oracle"))
    results = run_probes(cfg, probes, oracle_transport(probes) if args.mock else http_transport)
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])
    for kind, ok in sorted(by_kind.items()):
        print(f"{kind:14s} {sum(ok)}/{len(ok)}")
    total = sum(r["correct"] for r in results)
    print(f"total {total}/{len(results)}" + ("  (mock oracle: verifies the harness, not a model)" if args.mock else ""))
    if args.out:
        Path(args.out).write_text(json.dumps({"briefing_version": BRIEFING_VERSION, "seed": args.seed,
                                              "model": model_metadata(cfg), "results": results}, indent=1),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
