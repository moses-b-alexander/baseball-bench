"""Focused tests for counts, bases, lookups, intentional walks, persistence, the model adapter, and data cutoffs."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import agent
import bench
import data

ROOT = Path(__file__).resolve().parent
SNAP = bench.load_snapshot(ROOT / "data" / "demo")
CFG = bench.load_config(ROOT / "config.json")
LAST = SNAP["scenario"]["last_defensive_inning"]


def make_env(forced=None, **sections):
    cfg = copy.deepcopy(CFG)
    for section, values in sections.items():
        cfg[section].update(values)
    return bench.make_env(SNAP, cfg, forced=forced)


def runner(pid):
    return {"player_id": pid, "responsible_pitcher_id": "pitcher_01"}


def set_bases(s, *occupied):
    s["bases"] = {b: runner(f"r_{b}") if b in occupied else None for b in bench.BASES}


def kinds(events, kind):
    return [e for e in events if e["type"] == kind]


CONTINUE = {"action": "continue"}
IBB = {"action": "intentional_walk"}


class CountAndBases(unittest.TestCase):
    def play(self, outcomes, bases=(), hit_types=(), outs=None):
        env = make_env(forced={"outcome": list(outcomes), "hit_type": list(hit_types)})
        s = bench.init_state(env)
        set_bases(s, *bases)
        if outs is not None:
            s["outs_this_half"] = outs
        new, events, _ = bench.step(s, CONTINUE, env, random.Random(0))
        return s, new, events

    def test_four_balls_make_one_walk(self):
        s, new, events = self.play(["balls"] * 4)
        self.assertEqual(len(kinds(events, "pitch")), 4)
        pas = kinds(events, "pa")
        self.assertEqual([p["result"] for p in pas], ["walk"])
        self.assertEqual(new["bases"]["first"]["player_id"], pas[0]["batter_id"])
        self.assertEqual((new["balls"], new["strikes"]), (0, 0))
        self.assertEqual(new["score"], s["score"])

    def test_strikeout_and_two_strike_foul(self):
        s, new, events = self.play(["strikes", "strikes", "fouls", "fouls", "strikes"])
        pitches = kinds(events, "pitch")
        self.assertEqual([p["count_after"] for p in pitches[2:4]], [[0, 2], [0, 2]])
        self.assertEqual(kinds(events, "pa")[0]["result"], "strikeout")
        self.assertEqual(new["outs_this_half"], s["outs_this_half"] + 1)

    def test_loaded_walk_forces_one_run(self):
        s, new, _ = self.play(["balls"] * 4, bases=bench.BASES)
        self.assertEqual(new["score"]["opponent"], s["score"]["opponent"] + 1)
        self.assertEqual([new["bases"][b]["player_id"] for b in ("second", "third")], ["r_first", "r_second"])
        ids = [new["bases"][b]["player_id"] for b in bench.BASES]
        self.assertEqual(len(set(ids)), 3)

    def test_home_run_clears_bases(self):
        s, new, _ = self.play(["hits"], bases=("first", "third"), hit_types=["home_run"])
        self.assertEqual(new["score"]["opponent"], s["score"]["opponent"] + 3)
        self.assertEqual(new["bases"], dict.fromkeys(bench.BASES))

    def test_third_out_resets_and_advances_horizon(self):
        s, new, events = self.play(["in_play_outs"], bases=("second",), outs=2)
        self.assertEqual(new["bases"], dict.fromkeys(bench.BASES))
        self.assertEqual((new["outs_this_half"], new["balls"], new["strikes"]), (0, 0, 0))
        self.assertEqual(new["inning"], s["inning"] + 1)
        self.assertEqual(new["next_batter_index"], (s["next_batter_index"] + 1) % 9)
        runs = SNAP["scenario"]["offense_runs_after_inning"][str(s["inning"])]
        self.assertEqual(new["score"]["controlled"], s["score"]["controlled"] + runs)
        self.assertEqual(len(kinds(events, "half_end")), 1)

    def test_exposure_counts_outs_and_hits_not_pa(self):
        env = make_env(forced={"outcome": ["hits", "balls", "balls", "balls", "balls", "strikes", "strikes",
                                           "strikes"], "hit_type": ["single"]})
        s = bench.init_state(env)
        rng = random.Random(0)
        e0 = bench.exposure(s["defense_totals"])
        s, _, _ = bench.step(s, CONTINUE, env, rng)
        self.assertEqual(bench.exposure(s["defense_totals"]), e0 + 1)
        s, _, _ = bench.step(s, CONTINUE, env, rng)  # walk
        self.assertEqual(bench.exposure(s["defense_totals"]), e0 + 1)
        s, _, _ = bench.step(s, CONTINUE, env, rng)  # strikeout
        self.assertEqual(bench.exposure(s["defense_totals"]), e0 + 2)
        self.assertEqual(s["defense_totals"]["completed_pa"], SNAP["scenario"]["initial_state"]["defense_totals"]["completed_pa"] + 3)

    def test_long_pa_truncates_instead_of_forcing_an_out(self):
        env = make_env(forced={"outcome": ["strikes", "strikes"] + ["fouls"] * 20}, execution={"max_pitches_per_pa": 10})
        s = bench.init_state(env)
        new, events, _ = bench.step(s, CONTINUE, env, random.Random(0))
        self.assertEqual(new["status"], "truncated")
        self.assertEqual(new["outs_this_half"], s["outs_this_half"])
        self.assertEqual(kinds(events, "pa"), [])


class IntentionalWalk(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.s = bench.init_state(self.env)
        self.rng = random.Random(42)

    def test_invariants_and_no_randomness(self):
        s, env = self.s, self.env
        set_bases(s, *bench.BASES)
        pid, bid = s["current_pitcher_id"], s["batting_order"][s["next_batter_index"]]
        before = self.rng.getstate()
        new, events, _ = bench.step(s, IBB, env, self.rng)
        self.assertEqual(self.rng.getstate(), before)
        self.assertEqual(new["score"]["opponent"], s["score"]["opponent"] + 1)
        self.assertEqual(new["bases"]["first"], {"player_id": bid, "responsible_pitcher_id": pid})
        for key in ("pitches_thrown", "hits_allowed", "outs_recorded"):
            self.assertEqual(new["defense_totals"][key], s["defense_totals"][key])
        self.assertEqual(new["defense_totals"]["completed_pa"], s["defense_totals"]["completed_pa"] + 1)
        self.assertEqual(new["next_batter_index"], (s["next_batter_index"] + 1) % 9)
        self.assertEqual(kinds(events, "pitch"), [])
        self.assertEqual([p["result"] for p in kinds(events, "pa")], ["intentional_walk"])
        bat_side, throw_side = bench.matchup_sides(env, pid, bid)
        for role, player, side in (("pitchers", pid, bat_side), ("batters", bid, throw_side)):
            for sd in (side, "both"):
                old, cur = s["game_splits"][role][player][sd], new["game_splits"][role][player][sd]
                self.assertEqual(cur["intentional_walks"], old["intentional_walks"] + 1)
                self.assertEqual(cur["completed_pa"], old["completed_pa"] + 1)
                self.assertEqual(cur["pitch_count"], old["pitch_count"])
                self.assertEqual(cur["outcomes"], old["outcomes"])

    def test_runner_on_second_only_stays(self):
        set_bases(self.s, "second")
        new, _, _ = bench.step(self.s, IBB, self.env, self.rng)
        self.assertEqual(new["bases"]["second"]["player_id"], "r_second")
        self.assertIsNone(new["bases"]["third"])
        self.assertIsNotNone(new["bases"]["first"])
        self.assertEqual(new["score"], self.s["score"])

    def test_walk_does_not_pay_a_later_lookup(self):
        s = self.s
        following = s["batting_order"][(s["next_batter_index"] + 1) % 9]
        target = s["batting_order"][(s["next_batter_index"] + 4) % 9]
        s1, _, _ = bench.step(s, IBB, self.env, self.rng)
        s2, events, receipt = bench.step(s1, {"action": "check_at_bat", "pitcher_id": "reliever_03",
                                              "batter_id": target}, self.env, self.rng)
        pas = kinds(events, "pa")
        self.assertEqual(len(pas), 1)
        self.assertEqual((pas[0]["batter_id"], pas[0]["pitcher_id"]), (following, s["current_pitcher_id"]))
        self.assertEqual(receipt["elapsed_pa_ids"], [pas[0]["pa_id"]])


class Lookup(unittest.TestCase):
    def test_later_target_still_faces_next_batter(self):
        env = make_env()
        s = bench.init_state(env)
        nxt = s["batting_order"][s["next_batter_index"]]
        target = s["batting_order"][(s["next_batter_index"] + 5) % 9]
        new, events, receipt = bench.step(s, {"action": "check_at_bat", "pitcher_id": "reliever_04",
                                              "batter_id": target}, env, random.Random(3))
        pas = kinds(events, "pa")
        self.assertEqual(len(pas), 1)
        self.assertEqual((pas[0]["batter_id"], pas[0]["pitcher_id"]), (nxt, s["current_pitcher_id"]))
        self.assertEqual(receipt["target_lineup_distance_at_request"], 5)
        self.assertEqual(receipt["cost_pa"], 1)
        self.assertEqual(new["batting_order"], s["batting_order"])
        self.assertEqual(new["current_pitcher_id"], s["current_pitcher_id"])
        # Snapshot was taken before the elapsed PA.
        self.assertEqual(receipt["pitcher"]["as_of"]["simulation_event_seq"], receipt["queried_at_event_seq"])
        self.assertGreater(receipt["delivered_at_event_seq"], receipt["queried_at_event_seq"])
        self.assertNotIn("season_outs_before_game", receipt["pitcher"]["usage_health"])

    def test_receipt_profile_is_pre_action(self):
        env = make_env()
        s = bench.init_state(env)
        pid, nxt = s["current_pitcher_id"], s["batting_order"][s["next_batter_index"]]
        before = s["game_splits"]["batters"][nxt]["both"]["pitch_count"]
        _, _, receipt = bench.step(s, {"action": "check_at_bat", "pitcher_id": pid, "batter_id": nxt},
                                   env, random.Random(3))
        self.assertEqual(receipt["batter"]["game"]["both"]["pitch_count"], before)

    def test_lookup_on_last_pa_is_charged_and_ends_episode(self):
        env = make_env(forced={"outcome": ["in_play_outs"]})
        env["scenario"] = dict(env["scenario"], initial_state=dict(env["scenario"]["initial_state"],
                                                                     inning=LAST, outs_this_half=2))
        calls = []

        def decide(obs, check):
            calls.append(obs["decision_id"])
            return {"action": "check_at_bat", "pitcher_id": "reliever_02", "batter_id": obs["next_batter_id"]}, []

        out = bench.run_episode(env, decide)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["state"]["status"], "completed")
        self.assertEqual(len(out["evidence"]), 1)
        self.assertEqual(len(out["evidence"][0]["elapsed_pa_ids"]), 1)
        sm = bench.summarize(out["events"], out["state"], out["evidence"])
        self.assertEqual(sm["lookups"]["no_later_action_opportunity"], 1)

    def test_free_lookup_consumes_no_time_or_randomness_and_is_capped(self):
        env = make_env(lookup={"cost_pa": 0})
        s = bench.init_state(env)
        rng = random.Random(9)
        before = rng.getstate()
        action = {"action": "check_at_bat", "pitcher_id": "reliever_02", "batter_id": s["batting_order"][0]}
        cur = s
        for _ in range(CFG["lookup"]["max_free_per_boundary"]):
            cur, events, receipt = bench.step(cur, action, env, rng)
            self.assertEqual(kinds(events, "pa"), [])
            self.assertEqual(receipt["elapsed_pa_ids"], [])
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(cur["game_splits"], s["game_splits"])
        self.assertEqual(cur["defense_totals"], s["defense_totals"])
        with self.assertRaises(bench.InvalidAction):
            bench.step(cur, action, env, rng)
        cur, _, _ = bench.step(cur, CONTINUE, env, rng)  # a completed PA resets the boundary budget
        bench.step(cur, action, env, rng)


class Switching(unittest.TestCase):
    def test_switch_costs_zero_pa_and_new_pitcher_throws_next(self):
        env = make_env()
        s = bench.init_state(env)
        set_bases(s, "first", "third")
        new, events, _ = bench.step(s, {"action": "switch_pitcher", "pitcher_id": "reliever_03"}, env,
                                    random.Random(0))
        self.assertEqual(kinds(events, "pa") + kinds(events, "pitch"), [])
        self.assertEqual(new["bases"], s["bases"])
        self.assertEqual(new["defense_totals"], s["defense_totals"])
        self.assertEqual(new["next_batter_index"], s["next_batter_index"])
        self.assertEqual(new["current_pitcher_id"], "reliever_03")
        _, events, _ = bench.step(new, CONTINUE, env, random.Random(0))
        self.assertEqual({p["pitcher_id"] for p in kinds(events, "pitch")}, {"reliever_03"})

    def test_one_switch_per_boundary_and_no_reentry(self):
        env = make_env()
        s, _, _ = bench.step(bench.init_state(env), {"action": "switch_pitcher", "pitcher_id": "reliever_03"},
                             env, random.Random(0))
        self.assertEqual(bench.legal_actions(s, env)["switch_pitcher"], [])
        self.assertIn("switch_limit", bench.validate_action({"action": "switch_pitcher", "pitcher_id": "reliever_04"},
                                                            s, env))
        s, _, _ = bench.step(s, IBB, env, random.Random(0))
        self.assertIn("already_used", bench.validate_action({"action": "switch_pitcher", "pitcher_id": "pitcher_01"},
                                                            s, env))
        self.assertIn("not_on_active_roster", bench.validate_action(
            {"action": "switch_pitcher", "pitcher_id": "reliever_07"}, s, env))


class Validation(unittest.TestCase):
    def test_invalid_actions_change_nothing(self):
        env = make_env()
        s = bench.init_state(env)
        snapshot = copy.deepcopy(s)
        rng = random.Random(5)
        before = rng.getstate()
        bad = [{"action": "continue", "extra": 1}, {"action": "bunt"}, "continue",
               {"action": "switch_pitcher", "pitcher_id": "reliever_99"},
               {"action": "check_at_bat", "pitcher_id": "reliever_02", "batter_id": "nobody"},
               {"action": "check_at_bat", "pitcher_id": "reliever_02"},
               {"action": "continue", "reason": "x" * 401}]
        for action in bad:
            with self.assertRaises(bench.InvalidAction):
                bench.step(s, action, env, rng)
        self.assertEqual(s, snapshot)
        self.assertEqual(rng.getstate(), before)

    def test_initial_observation_hides_detailed_profiles(self):
        env = make_env()
        obs = bench.observe(bench.init_state(env), env, [], "")
        self.assertEqual(obs["acquired_evidence"], [])
        text = json.dumps(obs)
        self.assertNotIn("usage_health", text)
        self.assertNotIn('"stats"', text)
        for entry in obs["bullpen_overview"]:
            self.assertEqual(set(entry), {"player_id", "throws", "roster_available", "legal_to_enter_now",
                                          "unavailable_reason"})
        self.assertIsNone(json.loads(text)["game"]["bases"]["first"])

    def test_hidden_shapes_condition(self):
        env = make_env(observation={"show_shapes": False})
        s = bench.init_state(env)
        obs = bench.observe(s, env, [], "")
        _, _, receipt = bench.step(s, {"action": "check_at_bat", "pitcher_id": "reliever_02",
                                       "batter_id": s["batting_order"][0]}, env, random.Random(0))
        for blob in (obs, receipt):
            text = json.dumps(blob)
            self.assertNotIn("velocity_mph", text)
            self.assertNotIn("bat_speed_mph", text)


class Persistence(unittest.TestCase):
    ACTIONS = [{"action": "check_at_bat", "pitcher_id": "reliever_02", "batter_id": "batter_03"}, IBB,
               {"action": "switch_pitcher", "pitcher_id": "reliever_02"}, CONTINUE, CONTINUE]

    def test_replay_is_identical(self):
        a = bench.run_episode(make_env(), agent.fixed_sequence(self.ACTIONS))
        b = bench.run_episode(make_env(), agent.fixed_sequence(self.ACTIONS))
        self.assertEqual(a["events"], b["events"])

    def test_resume_with_pending_action_does_not_double_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir, crash_dir = Path(tmp, "ref"), Path(tmp, "crash")
            bench.run_episode(make_env(), agent.fixed_sequence(self.ACTIONS), ref_dir)

            real_step, calls = bench.step, []

            def crashing_step(*args):
                calls.append(1)
                if len(calls) == 4:  # crash after the 4th decision was captured but before it committed
                    raise KeyboardInterrupt
                return real_step(*args)

            bench.step = crashing_step
            try:
                with self.assertRaises(KeyboardInterrupt):
                    bench.run_episode(make_env(), agent.fixed_sequence(self.ACTIONS), crash_dir)
            finally:
                bench.step = real_step
            ck = data.load_json(crash_dir / bench.CHECKPOINT)
            self.assertEqual(ck["pending"]["action"], CONTINUE)
            # Simulate a torn write: an uncommitted event past the checkpoint.
            bench.append_jsonl(crash_dir / bench.EVENTS, [{"seq": ck["committed_event_seq"] + 1, "type": "pitch"}])

            seen = []

            def after_resume(obs, check):
                seen.append(obs["decision_id"])
                return CONTINUE, []

            bench.run_episode(make_env(), after_resume, crash_dir, resume=True)
            self.assertEqual(seen[0], "demo_001-d0004")  # pending decision 3 was executed, not re-asked
            ref, got = bench.read_jsonl(ref_dir / bench.EVENTS), bench.read_jsonl(crash_dir / bench.EVENTS)
            self.assertEqual([e["seq"] for e in got], list(range(1, len(got) + 1)))
            self.assertEqual(got, ref)

    def test_resume_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_env(execution={"max_decisions": 2})
            bench.run_episode(env, agent.fixed_sequence([]), tmp)
            with self.assertRaises(ValueError):
                bench.run_episode(make_env(), agent.fixed_sequence([]), tmp, resume=True)

    def test_decision_limit_truncates(self):
        out = bench.run_episode(make_env(execution={"max_decisions": 3}), agent.fixed_sequence([]))
        self.assertEqual((out["state"]["status"], out["state"]["termination_reason"]), ("truncated", "max_decisions"))
        self.assertEqual(out["state"]["decisions_made"], 3)


def mock_transport(replies):
    replies = list(replies)
    sent = []

    def transport(url, payload, headers, timeout):
        sent.append(payload)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    transport.sent = sent
    return transport


class ModelAdapter(unittest.TestCase):
    def cfg(self):
        return dict(CFG, model=dict(CFG["model"], model="mock-model"))

    def test_parse_handles_fences_and_think_blocks(self):
        self.assertEqual(agent.parse_action('```json\n{"action": "continue"}\n```')[0], CONTINUE)
        self.assertEqual(agent.parse_action('<think>hmm</think>{"action": "continue"}')[0], CONTINUE)
        self.assertIsNotNone(agent.parse_action('{"action": "continue"} {"action": "continue"}')[1])
        self.assertIsNotNone(agent.parse_action('["continue"]')[1])

    def test_repair_then_accept(self):
        t = mock_transport(['{"action": "switch_pitcher", "pitcher_id": "reliever_07"}', '{"action": "continue"}'])
        env = make_env()
        s = bench.init_state(env)
        action, attempts = agent.make_model_decide(self.cfg(), t)(bench.observe(s, env, [], ""),
                                                                  lambda a: bench.validate_action(a, s, env))
        self.assertEqual(action, CONTINUE)
        self.assertEqual([bool(a["error"]) for a in attempts], [True, False])
        self.assertIn("not_on_active_roster", t.sent[1]["messages"][-1]["content"])
        self.assertEqual(t.sent[0]["seed"], CFG["execution"]["model_seed"])

    def test_infra_retry_then_success(self):
        t = mock_transport([OSError("connection refused"), '{"action": "continue"}'])
        env = make_env()
        s = bench.init_state(env)
        action, attempts = agent.make_model_decide(self.cfg(), t)(bench.observe(s, env, [], ""), lambda a: None)
        self.assertEqual(action, CONTINUE)
        self.assertEqual([a["kind"] for a in attempts], ["infra_error", "model"])

    def test_repeated_invalid_marks_episode_failed(self):
        t = mock_transport(["not json", "still not json"])
        with tempfile.TemporaryDirectory() as tmp:
            out = bench.run_episode(make_env(), agent.make_model_decide(self.cfg(), t), tmp)
            self.assertEqual(out["state"]["status"], "failed")
            self.assertEqual(out["state"]["decisions_made"], 0)
            sm = bench.summarize_run(tmp)
            self.assertEqual(sm["reliability"]["invalid_attempts"], 2)
            self.assertEqual(sm["plate_appearances"], 0)

    def test_probe_harness_scores_oracle(self):
        probes = agent.make_probes(seed=11)
        results = agent.run_probes(self.cfg(), probes, agent.oracle_transport(probes))
        self.assertTrue(all(r["correct"] for r in results))
        wrong = mock_transport(['{"answer": "banana"}'] * len(probes))
        self.assertFalse(any(r["correct"] for r in agent.run_probes(self.cfg(), probes, wrong)))


def row(game_id, ts, outcome_raw="ball", pa_result=None, stand="L", throws="R", velo=None, pitcher="p1",
        batter="b1", ptype="FF"):
    return {"game_id": game_id, "date": ts[:10], "ts": ts, "pa_id": "x", "pitch_num": 1, "pitcher_id": pitcher,
            "p_throws": throws, "batter_id": batter, "stand": stand, "pitch_type": ptype,
            "raw_description": outcome_raw, "outcome": data.map_outcome(outcome_raw, pa_result), "pa_result": pa_result,
            "velocity_mph": velo, "ivb_in": None, "horizontal_break_in": None, "swing": False,
            "competitive_swing": False, "attack_angle_deg": None, "swing_path_tilt_deg": None, "bat_speed_mph": None}


class Data(unittest.TestCase):
    def test_split_identities_and_weighted_means(self):
        left, right = data.pitcher_split(), data.pitcher_split()
        left["pitch_count"], left["outcomes"]["balls"] = 5, 5
        left["pitch_types"]["FF"] = {"count": 5, "shape": {"velocity_mph": {"mean": 90.0, "n": 1},
                                                          "ivb_in": data.mean_metric(),
                                                          "horizontal_break_in": data.mean_metric()}}
        right["pitch_count"], right["outcomes"]["hits"] = 1, 1
        right["pitch_types"]["FF"] = {"count": 1, "shape": {"velocity_mph": {"mean": 94.0, "n": 3},
                                                           "ivb_in": data.mean_metric(),
                                                           "horizontal_break_in": data.mean_metric()}}
        both = data.combine(left, right)
        self.assertEqual(both["pitch_count"], 6)
        self.assertEqual(both["outcomes"]["balls"] + both["outcomes"]["hits"], 6)
        self.assertEqual(both["pitch_types"]["FF"]["shape"]["velocity_mph"], {"mean": 93.0, "n": 4})  # not by pitch count
        self.assertEqual(both["pitch_types"]["FF"]["shape"]["ivb_in"], {"mean": None, "n": 0})

    def test_null_is_not_zero(self):
        self.assertEqual(data.combine(data.rate_metric(), data.rate_metric())["value"], None)
        observed_zero = {"mean": 0.0, "n": 2}
        self.assertEqual(data.combine(observed_zero, data.mean_metric()), {"mean": 0.0, "n": 2})
        errors = []
        data.check_metrics({"m": {"mean": 0.0, "n": 0}}, "x", errors)
        self.assertTrue(errors)

    def test_innings_notation(self):
        self.assertEqual(data.ip_to_outs("5.2"), 17)
        self.assertEqual(data.ip_display(17), "5.2")
        with self.assertRaises(ValueError):
            data.ip_to_outs("5.3")

    def test_switch_hitter_side(self):
        self.assertEqual(data.resolve_side("S", "R"), "L")
        self.assertEqual(data.resolve_side("S", "L"), "R")
        env = make_env()
        self.assertEqual(bench.matchup_sides(env, "pitcher_01", "batter_03"), ("left", "right"))

    def test_cutoff_and_game_subset(self):
        sc = {"historical_before": "2025-06-10T23:05:00Z", "branch_ts": "2025-06-10T23:40:00Z", "game_id": "g"}
        pitches = [row("old", "2025-06-01T23:10:00Z", velo=90.0),
                   row("g", "2025-06-10T23:20:00Z", "hit_into_play", "single", velo=92.0),
                   row("g", "2025-06-10T23:50:00Z", "foul"),            # rest of the real game: excluded
                   row("new", "2025-06-12T23:10:00Z", "foul"),          # future game: excluded
                   row("old", "2025-06-02T23:10:00Z", "hit_by_pitch", "hit_by_pitch")]  # unsupported
        agg = data.aggregate(pitches, [], sc, ["p1"], ["b1"])
        season, game = agg["pitchers"]["p1"]["season"], agg["pitchers"]["p1"]["game_prefix"]
        self.assertEqual((season["both"]["pitch_count"], game["both"]["pitch_count"]), (1, 1))
        self.assertEqual(agg["excluded"], {"outside_cutoff_window": 2, "unsupported:hit_by_pitch": 1})
        stats = data.combine(season, game)
        self.assertEqual(stats["both"]["pitch_count"], 2)
        self.assertEqual(stats["left"]["pitch_types"]["FF"]["shape"]["velocity_mph"], {"mean": 91.0, "n": 2})
        self.assertEqual(agg["batters"]["b1"]["game_prefix"]["right"]["completed_pa"], 1)

    def test_zero_pitch_ibb_event_adds_no_pitches(self):
        sc = {"historical_before": "2025-06-10T23:05:00Z", "branch_ts": "2025-06-10T23:40:00Z", "game_id": "g"}
        ev = {"game_id": "old", "date": "2025-06-01", "ts": "2025-06-01T23:10:00Z", "pa_id": "y", "pitcher_id": "p1",
              "p_throws": "R", "batter_id": "b1", "stand": "L", "event": "intentional_walk"}
        agg = data.aggregate([], [ev], sc, ["p1"], ["b1"])
        both = agg["pitchers"]["p1"]["season"]["both"]
        self.assertEqual((both["pitch_count"], both["completed_pa"], both["intentional_walks"]), (0, 1, 1))

    def test_demo_snapshot_validates(self):
        errors, _ = data.validate(ROOT / "data" / "demo")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
