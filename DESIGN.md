# BaseballBench — design and implementation status

This file has two parts:

- **Part A, Implementation status:** the ground truth for what exists in code. Update it whenever code merges.
- **Part B, Design handoff v0.3:** the original design spec, unchanged. It describes intended contracts, and parts of it are not built. If Part A and Part B disagree, Part A describes the code.

---

# Part A — Implementation status

Last updated 2026-10-26. Code: branch [`v0`](https://github.com/moses-b-alexander/baseball-bench/tree/v0), commit `d476d8b`. `main` holds only this document and the README so far. Work on the data specs goes on branch `data-specs`, which starts from `v0`.

## A.1 Milestone status

| Milestone (Part B §10) | Status |
| --- | --- |
| 1. Offline core | **Done.** A scripted episode runs end to end with no network or GPU. |
| 2. Agent loop and shared briefing | **Done, except that no real model has run.** The adapter is tested only with mock transports. |
| 3. Frozen data pipeline | **Partial.** Cutoff-safe aggregation and validation work on the synthetic data. There is no real-data import, team identity table, or dataset split. |
| 4. Comparison runner and curriculum pilot | **Not started.** Each `run` executes one episode. There is no multi-scenario or multi-seed runner and no curriculum. |
| 5. Optional extensions | **Not started.** |

## A.2 Verified on 2026-10-26 (Windows, Python 3, standard library only)

| Command | Result |
| --- | --- |
| `python -m unittest test_bench` | 35 tests pass |
| `python data.py validate --snapshot data/demo` | OK: 6,960 pitches used. Excluded: 801 pitch rows after the cutoff, 1 PA event after the cutoff, 30 hit-by-pitch |
| `python bench.py smoke` | `completed (defensive_horizon_reached)`, 25 decisions, 22 PA, 70 pitches, 1 lookup, 1 intentional walk, 1 switch (`pitcher_01` to `reliever_02` in inning 6) |
| `python agent.py probes --mock` | 21/21. The mock returns the correct answers itself, so this checks the harness, not a model |

Other commands are `python data.py make-demo` (regenerates `data/demo` with seed 20250610), `python bench.py run --agent {never_switch,threshold,random,tour,model} [--resume]`, `python bench.py summarize --run-dir DIR`, and `python agent.py briefing`.

## A.3 Files

| File | Lines | Contents |
| --- | --- | --- |
| `bench.py` | 803 | Config checks, state, pooled-frequency probabilities, count/runner reducers, PA simulation, observation, action validation and execution (`step`), episode loop with checkpoint/resume, summary, CLI |
| `agent.py` | 365 | Briefing (`briefing-0.1`), observation formatting, OpenAI-compatible model adapter, parsing and repair, scripted baselines, comprehension probes, CLI |
| `data.py` | 704 | Schema primitives, cutoff-safe `aggregate`, `workload_days`, CSV I/O, synthetic snapshot generator, `validate`, CLI |
| `test_bench.py` | 497 | `unittest` suite |
| `config.json` | 62 | The Part B §9 defaults plus `data`, `baselines` and `model` blocks |
| `data/demo/` | — | Synthetic `pitches.csv`, `pa_events.csv`, `profiles.json`, `sim_tables.json`, `scenario.json`, `manifest.json` (with file hashes) |

Version strings: schema `0.3`, ruleset `rules-0.1`, provider `pooled_frequency-0.1`, mapping `map-0.1`, pitch registry `demo-registry-0.1` (FF, SI, FC, SL, CU, CH, UNKNOWN), briefing `briefing-0.1`. Dependencies: standard library only.

## A.4 What the code implements

**Episode.** Fixed defensive horizon (Part B §2). The demo scenario `demo_001` is synthetic. The controlled team is home and starts defending in the top of the 6th with 1 out, starter `pitcher_01` on the mound, through the end of the 9th. The controlled team's runs between innings come from `offense_runs_after_inning` (`{"6": 0, "7": 1, "8": 0}`). The pitcher pool is the starter plus six relievers. `reliever_07` has `roster_available: false`.

**Simulator.** Each pitch samples a pitch type (by batter side, falling back to overall, then `UNKNOWN`), then one of five outcomes. Outcome probabilities pool smoothed pitcher and batter frequencies (`alpha = 20`, `w = 0.5`) toward a frozen reference distribution. The pitcher side falls back from pitch type and side, to side, to overall; the batter side from hand to overall. Each pitch event logs the fallback levels used. Hit subtypes come from the frozen `hit_type_counts`. Four balls make a walk, three strikes a strikeout, and a foul with two strikes leaves the count unchanged. A PA reaching `max_pitches_per_pa` truncates the episode rather than forcing an out.

**Baserunning.** Deterministic. A walk or intentional walk forces only the runners who must move. On a hit, every runner moves n bases. A strikeout or in-play out records one out and runners hold. There are no steals, double plays, errors or HBP. Runners carry `responsible_pitcher_id`. A third out ends the half, clears the bases, applies the offense entry, and either advances the inning or ends the episode after the last defensive inning.

**Actions** (`bench.step` works on a deep copy, so a failed validation never mutates state):
- `continue`: simulates one PA.
- `intentional_walk`: zero pitches and zero RNG draws. It reuses `force_walk` and counts as a completed PA and an IBB in the splits.
- `switch_pitcher`: zero PA. Limited to `max_switches_per_boundary` (1), with no reentry, and the reliever must be roster-available.
- `check_at_bat`: snapshots both profiles first, then simulates `cost_pa` PAs (1) with the incumbent against the actual next batter, then delivers a `LookupReceipt`. Setting `cost_pa = 0` gives the free-lookup treatment, capped at 8 per boundary and using no PA and no RNG.
- Validation rejects unknown actions, unknown fields, wrong types, over-length `reason`/`notes` (400/1,500 characters), and unknown or illegal IDs.

**Observation** (`bench.observe`). It contains the game state, outs to end the inning, defense and current-pitcher exposure (outs + hits), the current pitcher's appearance and game splits, the ordered opposing lineup, and a bullpen overview (ID, hand, availability, legal now, and the reason if not). It also has the legal actions, lookup rules, every acquired receipt (snapshots, never truncated) and the agent's notes. It does not include detailed reliever or batter profiles, usage, or health; those come only through lookups. `show_shapes: false` removes pitch and swing shapes everywhere. `show_player_names: false` sets `display_name` to null.

**Profiles.** `stats = season-before-game + game`, where `game = real prefix + simulated events`. Simulated pitches increase counts but never add shape or swing measurements. Season outs and the IP display string are derived from outs.

**Persistence.** `checkpoint.json` is written atomically and holds the state, evidence, notes, RNG state, config hash, committed event sequence, and any pending accepted action. `events.jsonl` has stable IDs and is trimmed to the committed sequence on resume. `decisions.jsonl` records every attempt with raw text, usage and latency. `summary.json` holds the summary. Resume refuses a changed config and replays a pending action without asking the model again.

**Agent side.**
- The model adapter POSTs to `{base_url}/chat/completions` with `urllib`. It sends the seed and `response_format: json_object` when `json_mode` is on, and reads an API key from the environment variable named in `api_key_env`.
- The parser strips `<think>` blocks and code fences and requires exactly one JSON object.
- Each decision gets one repair attempt with the error message, plus up to 2 infrastructure retries before anything is committed. A decision still invalid after repair marks the episode `failed`.
- Baselines: `never_switch`; `threshold` (95 pitches for the starter, 25 for a reliever, then the first legal reliever in pool order); `random` (seeded random legal action); `tour` (the smoke script).
- Probes: 21 per seed, covering IP and outs, rates, count resolution, lookup timing, null versus zero, splits, legality and intentional-walk runs. The answers are generated from engine helpers.

**Summary metrics.** Status, score, runs allowed, runs charged per pitcher, PA, pitches, results, hits, walks (with intentional walks listed separately), and strikeouts. Replacements, with the first replacement (or `censored`). Lookups: count, unique pitchers and batters, lineup distances, repeated queries, lookups with no later action opportunity, and paid lookups before the first replacement. Reliability: attempts, invalid attempts, infrastructure errors, tokens and latency.

**Data pipeline** (synthetic only).
- `aggregate` builds season splits (rows before `historical_before`) and a current-game prefix (rows before `branch_ts`). It drops later games and the rest of the real game, and counts every exclusion.
- Pitch and swing means are weighted by their own `n`; rates are recomputed from numerator and denominator; `both` is `left + right`.
- Bat speed counts only competitive swings. The ideal attack-angle band is 5–20°.
- Zero-pitch intentional walks are stored in `pa_events.csv` and never turned into pitches.
- `validate` checks manifest hashes, re-aggregates the raw rows to prove nothing after the cutoff reached the profiles, and checks the split identities, null-iff-zero rules, the workload window, the batting order, and the starter's prefix against the scenario.

## A.5 Interpretations and deviations from Part B

- `check_config` **refuses to run** with any flag that has no code behind it: steals, double plays, three-batter minimum, fatigue, `repeat_paid_query_costs_pa=false`, other advancement modes, or other providers.
- The demo prefix's bases and score are replayed with the benchmark's deterministic runner rules, not taken from real play-by-play.
- Workload: five completed team game-dates before the game, zero-appearance dates included. Fewer than five raises an error, because no incomplete-data cohort exists yet. The synthetic data never produces `unavailable` days.
- `days_since_last_il_return`, `il_history_status` and `roster_available` are hand-set synthetic values in `data.DEMO_HEALTH`.
- `notes` replace the previous notes when an action includes them, and leaving them out keeps the old notes. `notes` are left out of the `action` event but kept in the checkpoint.
- In the `random` baseline, every legal switch is a separate option, so it switches more often than it would with uniform action types.
- Part B §6 also lists dataset splits, team metadata and curriculum generation as `data.py` responsibilities. None of these exist yet.

## A.6 Not implemented

- Real-data import (Statcast or another source), field and unit verification (IVB versus gravity-included drop, horizontal break sign), and a crosswalk.
- The team and franchise identity table (§8.1), chronological train/dev/test splits (§8.5), and division sampling (§8.4).
- A multi-scenario, multi-seed comparison runner; analysis across episodes; search-pattern metrics (several pitchers per batter versus several batters per pitcher); matched single-decision sensitivity probes.
- A curriculum or SFT data generator, and any training code.
- Any run against a real model endpoint. `model.model` is `null` in `config.json`.
- Fatigue, shape-conditioned probabilities, steals, double plays, the three-batter minimum, extra-inning runner, walk-offs, and full-game mode.

## A.7 Next up

- **Data specs (branch `data-specs`).** Decide the real-data source and season range, verify source fields, extend the pitch registry, map unsupported events (HBP, errors, fielder's choice, interference), define team identity, and write cutoff and split rules. Then build one frozen real-data snapshot that passes `data.py validate`.
- **Open decisions** are Part B §11. None have been resolved since the handoff.

---

# Part B — Design handoff v0.3 (unchanged spec)

Implementation handoff for Claude Code · 2026-10-21 · design version 0.3

## Start here

Build a small, reproducible Python benchmark for comparing how open-weight language-model agents acquire matchup information and decide when to replace a pitcher. The research object is the agent's behavior. The simulator should be transparent and inexpensive, not a separate predictive-model research project.

This is a design handoff, not an existing implementation. Names, interfaces, configuration values, and commands below are proposed contracts. Implement and test them before describing them as working. All illustrative player IDs and numbers are synthetic.

Read the entire document, inspect the repository if one exists, and implement incrementally. Preserve existing useful code. Start with a working offline episode using synthetic fixtures and a scripted agent, then connect one configurable model adapter to a plain Python loop. Do not stop at a plan. Do not spend money on inference or download large model weights without the user's authorization. Do not overwrite an existing CLAUDE.md; link this document from it if appropriate.

Use the defaults explicitly marked **proposed** when implementation needs a choice. Keep them in configuration and document them. These defaults are not prior user commitments. Ask only when a missing decision prevents a coherent implementation; routine choices should not block progress.

**User priority: simplicity. Minimize code, files, and dependencies.** The earlier multi-package architecture is superseded by section 6. LangGraph is optional portfolio work only if it later provides a real demonstrated integration; it is not a required execution path. The schemas below describe contracts, not a demand for a class or file per object.

## 1. Research goal and scope

Primary question: how do different open-weight agents decide to retain, investigate, or replace a pitcher, or intentionally walk a batter, and which evidence changes their actions?

The main information-gathering action has an opportunity cost: while an agent requests information about a candidate reliever and a selected batter, its current pitcher must face the actual next scheduled batter for one plate appearance (PA). The selected lookup batter may be farther ahead in the lineup. The lookup does not simulate an audition or change the batting order.

Measure replacement timing, reliever choice, information-search breadth/depth, lookahead distance, and responsiveness to evidence. Simulated runs allowed are a secondary outcome, conditional on the chosen simple simulator. Brief stated reasons are supplementary reports, not reliable access to a model's internal reasoning.

### Confirmed design requirements

| Area | Requirement |
| --- | --- |
| Agents | Compare multiple open-weight models through the same action and observation interface. |
| Decisions | Between plate appearances; simulate individual pitches underneath. |
| Main tools | `check_at_bat(pitcher_id, batter_id)` and `switch_pitcher(pitcher_id)`. |
| Retaining | An explicit `continue` action advances play without buying information. |
| Intentional walk | `intentional_walk` immediately awards first base to the next scheduled batter, completing that PA with zero pitches; force runners only as required. |
| Lookup meaning | Return statistical evidence for a candidate and a selected current/future batter. |
| Lookup cost | Main treatment: one actual PA with the incumbent before the next agent decision. |
| Pitch outcomes | Ball, strike, hit, and ball put in play for an out; include foul as a separate outcome. |
| Count | Maintain balls and strikes within the PA; reset after its completion. |
| Game memory | Score, inning/half, outs, bases, current pitcher, upcoming batters, and current-pitcher game statistics. |
| Exposure | Track outs and hits separately and expose their sum as an additional duration/exposure variable. |
| Inning boundary | `outs_to_end_inning = 3 - outs_this_half`; clear bases at the third out. |
| Pitcher profile | Usage/health; historical and current-game L/R/both splits; per-pitch-type counts and shapes. |
| Batter profile | The same outcome categories and L/R/both splits; aggregate swing shape, without pitch-type stratification. |
| Important correction | The user's earlier “K–BB” meant balls and strikes thrown, not strikeouts minus walks. Use explicit field names. |
| Statistical tools | Hypothesis testing belongs in evaluation, not in the agent's toolset. `pystats` was intended for this purpose, not baseball retrieval. |
| Simplicity | Use explicit stochastic frequencies and rules; no learned outcome model or collision-physics model is required. |
| Implementation | Minimal Python harness; standard library first, one model client/backend, training dependencies only when training is undertaken. |

### Proposed initial defaults

| Choice | Initial default | Why it is separate from the agreement |
| --- | --- | --- |
| Episode | One team's defensive management from a supplied game state to a fixed final defensive inning | Whole-game and season objectives have not been settled. |
| Other team's pitching / controlled team's batting | No model decisions; use a frozen schedule of offensive runs between defensive innings | Gives score context without building a second manager or full offense simulator. |
| Matchup probabilities | Equal-weight pooling of smoothed pitcher and batter pitch-outcome frequencies | The pooling rule was discussed but not selected. |
| Fatigue | Off in the first correctness baseline | Workload is observed; an outcome penalty has not been specified. |
| Shape effects | Profiles visible; no direct shape adjustment to outcome probabilities in the first baseline | A physical/empirical compatibility function has not been selected. |
| Baserunning | Deterministic advancement; steals and double plays off initially | State must support them, but their probabilities have not been specified. |
| Eligibility | Available roster members, no reentry, one substitution per decision boundary | A full MLB ruleset has not been requested. |
| Injury timing | Days since most recent IL return | The user's wording was “days since last IR stint”; return date is an explicit proposed interpretation. |
| Memory | Authoritative state + all acquired evidence + bounded agent notes | Prevent forgotten score arithmetic from becoming the main task. |
| Free lookup control | Implement as a configurable treatment | Proposed comparison, not a replacement for the main one-PA treatment. |

Do not label this initial ruleset an exact MLB simulator. In particular, automatic runners, three-batter minimum, rare pitch events, walk-offs, and full regulation-game termination are not implemented implicitly.

## 2. What one episode does

An episode contains an initial state, two ordered rosters as needed for identification, a controlled team's pitcher pool, a fixed opposing batting order, historical data frozen at the scenario cutoff, simulator parameters, and seeds.

At each eligible boundary:

1. Construct the public observation from authoritative state and already acquired evidence.
2. Ask the agent for exactly one structured action.
3. Validate the action without changing the game.
4. Execute the validated action once.
5. Log the action, resulting pitches/events, state changes, and information delivered.
6. Save the next boundary state and continue until the episode ends.

Do not request an LLM call for each pitch. `continue` and paid `check_at_bat` simulate a PA; `intentional_walk` completes one immediately without pitches; `switch_pitcher` changes who will pitch next without completing a PA.

### Defensive-episode convention

For the proposed first runner, all simulated PAs belong to the opponent's offense. When the defense records the third out, reset bases and count, apply the scenario's declared intervening offensive-run entry, increment the inning, and resume the next defensive half-inning. Preserve the opponent's batting-order position. The tracked half is `top` if the controlled team is home, otherwise `bottom`.

End immediately after completing the configured last defensive half-inning. Do not apply an extra offensive-run entry after termination. This is a fixed-horizon defensive experiment: score contextualizes decisions, but this mode does not claim regulation win/loss outcomes. All conditions receive the same offensive-run schedule. A full-game runner can be added behind the same environment interface later.

## 3. Schemas and invariants

Use ordinary dictionaries with type hints and a few explicit boundary checks; a dataclass for central game state is optional. Keep this schema and representative fixtures consistent with the code. Do not build a generic schema registry or add Pydantic just to mirror these tables. Reject unknown fields in action payloads; never coerce an unknown player ID into another player.

Use stable string IDs. JSON numbers contain no NaN or Infinity. Counts are nonnegative integers. Rates are fractions in `[0, 1]`, not percentages. Units belong in field names or versioned metadata. Missing is `null`; observed zero is `0`.

### 3.1 Shared statistical primitives

```text
OutcomeCounts:
  balls: int >= 0
  strikes: int >= 0
  fouls: int >= 0
  hits: int >= 0
  in_play_outs: int >= 0

MeanMetric:
  mean: float | null
  n: int >= 0

RateMetric:
  value: float in [0, 1] | null
  numerator: int >= 0
  denominator: int >= 0

PitchShape:
  velocity_mph: MeanMetric
  ivb_in: MeanMetric
  horizontal_break_in: MeanMetric

PitchTypeStats:
  count: int >= 0
  shape: PitchShape

SwingProfile:
  attack_angle_deg: MeanMetric
  swing_path_tilt_deg: MeanMetric
  bat_speed_mph: MeanMetric
  ideal_attack_angle_rate: RateMetric

PitcherSplit:
  pitch_count: int >= 0
  completed_pa: int >= 0
  intentional_walks: int >= 0
  outcomes: OutcomeCounts
  pitch_types: dict[PitchCode, PitchTypeStats]

BatterSplit:
  pitch_count: int >= 0
  completed_pa: int >= 0
  intentional_walks: int >= 0
  outcomes: OutcomeCounts
  swing_profile: SwingProfile

Splits[T]:
  left: T
  right: T
  both: T
```

`strikes` means called strikes plus swinging misses in this five-category simulator. Fouls and balls in play are distinct categories. Do not call this field an official broadcast “total strikes” statistic. Strikeouts and ordinary walks emerge from the count. Intentional walks are direct PA-level actions, recorded in `intentional_walks` and the event log; they add no pitch outcomes. Never fabricate four thrown balls for an intentional walk. Derive total walks from ordinary-walk and intentional-walk PA events, with the intentional component available separately.

For the initial five-outcome schema:

```text
pitch_count = balls + strikes + fouls + hits + in_play_outs
pitch_count = sum(pitch_types[type].count)  # pitchers only
both.outcome_counts = left.outcome_counts + right.outcome_counts
both.pitch_count = left.pitch_count + right.pitch_count
both.intentional_walks = left.intentional_walks + right.intentional_walks
```

One hit is one terminal pitch outcome, regardless of single/double/triple/home run. One double-play ball, if enabled later, is one `in_play_outs` event and can produce two actual outs. Actual outs belong in game/pitcher-appearance accounting, not this pitch-outcome count.

Split meaning is role-dependent:

| Profile | `left` / `right` refers to |
| --- | --- |
| Pitcher | The batter's actual batting side for that matchup |
| Batter | The pitcher's throwing hand |

Resolve switch-hitter side for each actual or hypothetical matchup. Do not use a static `S` as a third statistical split. Rows without resolvable handedness need an explicit exclusion report or a later schema version; do not silently put them into `both` while claiming `both = left + right`.

Combine means using their own valid measurement counts, not overall pitch counts. Some swings/pitches have one measurement but not another. A metric with `n = 0` has `mean = null`; a rate with denominator zero has `value = null`. For trimmed or filtered statistics, aggregate qualifying raw observations under the declared rule; simple weighted means are valid only when qualification is unchanged across groups.

### 3.2 Pitch repertoire and swing metrics

Use a versioned pitch-code registry covering all types in the chosen dataset, plus `UNKNOWN`. Every pitcher split contains every registry key. An unused type has count zero and all shape means null with measurement counts zero. The registry is a dataset contract; do not guess that a hard-coded example list contains every Statcast type.

The confirmed minimum movement profile is IVB and horizontal break. Retain velocity as the previously proposed companion measurement. Adopt one documented, signed horizontal coordinate system, for example the catcher's perspective, and include pitcher/batter handedness. Do not average absolute break magnitudes, and do not store hitter-relative break inside a mixed-handedness `both` aggregate. A display can derive toward/away information for a particular matchup.

IVB must represent induced movement, not gravity-included vertical drop. The importer must verify source-field definitions and units. Preserve raw columns and document conversions; do not assign `api_break_z_with_gravity` to `ivb_in`. Statcast's CSV documentation distinguishes these fields and coordinate conventions. [S4, S5]

Hitter swing profile is aggregated across pitch types within each handedness split. Use direct swing metrics rather than treating launch angle as swing plane: attack angle concerns local bat direction; tilt concerns swing-plane orientation. The ideal-attack-angle band is 5–20 degrees. [S4, S6] Preserve the source's eligibility/competitive-swing definition and sample counts, especially for bat speed. If an aggregate source cannot supply L/R swing splits, leave those metrics null rather than copy `both` into each split as if observed.

The public pitcher `pitch_types` mapping initially exposes counts and shapes, not per-type outcome probabilities. Internal simulator tables may contain per-type outcomes; keep that distinction explicit. The public batter profile contains no pitch-type-stratified outcomes or swing shapes.

### 3.3 Pitcher profile

```text
PitcherProfile:
  schema_version: str
  player_id: str
  display_name: str | null
  throws: 'L' | 'R'
  as_of: DataCutoff
  usage_health: UsageHealth
  stats: Splits[PitcherSplit]
  game: Splits[PitcherSplit]

UsageHealth:
  last_five_team_game_days: list[WorkloadDay]  # exactly five, oldest first
  season_outs_recorded: int >= 0
  season_ip_display: str                      # e.g. '42.2', never float 42.2
  days_since_last_il_return: int | null
  il_history_status: 'known_return' | 'no_recorded_stint' | 'unavailable'
  roster_available: bool

WorkloadDay:
  date: ISO date
  game_ids: list[str]
  outs_recorded: int >= 0 | null
  ip_display: str | null
  data_status: 'known' | 'unavailable'

DataCutoff:
  historical_before: ISO timestamp
  scenario_id: str
  simulation_event_seq: int >= 0
```

Workload days mean the five most recent completed team game-dates before the episode's game, including known zero-appearance dates. Combine doubleheaders on a date and retain their game IDs. Exclude the current date's game from this array to avoid an ambiguous moving window; current-game workload is exposed separately. This completed-date convention is proposed and must be in configuration/data documentation.

Unknown workload is not zero. If fewer than five prior dates are available, do not invent real dates; reject that scenario for the complete-window cohort or mark it as a separately declared incomplete-data cohort. Store outs internally and derive innings strings with `outs // 3` and `outs % 3`.

Roster availability is not identical to legal substitution eligibility. The latter can change with current state and belongs in `legal_actions`/the bullpen overview. Neither recent IL return nor workload automatically causes a simulator penalty in the baseline.

### 3.4 Batter profile

```text
BatterProfile:
  schema_version: str
  player_id: str
  display_name: str | null
  bats: 'L' | 'R' | 'S'
  as_of: DataCutoff
  stats: Splits[BatterSplit]
  game: Splits[BatterSplit]
```

The outcome structure is symmetric with the pitcher's. “Symmetric” does not mean adding a pitcher-style repertoire to the batter. Pitcher workload/IL fields also need not be mirrored onto a batter for this experiment.

### 3.5 Historical and simulated accumulation

Use a single cutoff before the branch point. Store season-before-game observations and the real current-game prefix up to the branch point separately. After branching, append only simulated events. Never append the rest of the historical game.

At any decision point:

```text
game = historical_game_prefix + simulated_game_events_so_far
stats = historical_season_before_game + game
```

Thus `game` is a subset of `stats`. Do not add it twice. Usage season outs follow the same convention. Immutable simulator probability tables are fitted from allowed pre-branch data and stay fixed during the baseline episode; updating displayed outcome counters does not refit the simulator after every pitch.

New simulated shape observations require an explicit generator. For the first baseline, simulated pitches have a pitch type but shape measurements may be null; simulated swing measurements are null. Counts still update. Historical shape means and their measurement counts remain unchanged when no new shape measurements are generated. Do not fabricate observed bat speed, tilt, or movement simply because an outcome was simulated. A later empirical shape sampler can populate those fields and must label them as simulated measurements.

### 3.6 Game state

```text
GameState:
  scenario_id: str
  controlled_team_id: str
  opponent_team_id: str
  inning: int >= 1
  half: 'top' | 'bottom'
  outs_this_half: int in [0, 2]          # at nonterminal public boundaries
  balls: int in [0, 3]                  # internal within-PA state
  strikes: int in [0, 2]
  bases: {first: Runner | null, second: Runner | null, third: Runner | null}
  score: {controlled: int >= 0, opponent: int >= 0}
  current_pitcher_id: str
  pitcher_appearances: dict[str, PitcherAppearance]
  used_pitcher_ids: list[str]
  batting_order: list[str]              # nine stable batter IDs
  next_batter_index: int in [0, 8]
  defense_totals: DefenseTotals
  substitutions_at_boundary: int
  event_seq: int
  status: 'active' | 'completed' | 'failed' | 'truncated'
  termination_reason: str | null

Runner:
  player_id: str
  responsible_pitcher_id: str

PitcherAppearance:
  outs_recorded: int >= 0
  hits_allowed: int >= 0
  pitches_thrown: int >= 0
  completed_pa: int >= 0
  entry_event_seq: int

DefenseTotals:
  outs_recorded: int >= 0
  hits_allowed: int >= 0
  pitches_thrown: int >= 0
  completed_pa: int >= 0
```

Derived fields exposed to the agent:

```text
outs_to_end_inning = 3 - outs_this_half
defense_exposure = defense_totals.outs_recorded + defense_totals.hits_allowed
current_pitcher_exposure = appearance.outs_recorded + appearance.hits_allowed
```

Defense totals are for the controlled team's pitching side, not both offenses combined. Preserve them across substitutions and inning resets. Current-pitcher exposure is a separate scope. The scenario seeds counters from the prior game prefix; it does not reset cumulative statistics at the branch.

`outs + hits` is intentionally not PA count: a walk adds zero, a hit adds one, and a double play can add two. Retain pitches, PA count, and inning alongside it. Never price a lookup using `outs + hits`; its clock is completed PAs.

The reducer may transiently reach three outs while resolving a play, then immediately finalize the half-inning. Never expose a playable state with three outs. Runs must be resolved under the selected play rule before clearing bases. Substitution does not clear runners, alter the batting order, or erase inherited-runner responsibility.

### 3.7 Public observation and acquired evidence

```text
AgentObservation:
  schema_version: str
  scenario_id: str
  decision_id: str
  game: PublicGameState
  current_pitcher: {player_id, throws, appearance, game_splits}
  opposing_lineup: ordered list[{player_id, bats, lineup_position}]
  next_batter_id: str
  bullpen_overview: list[{player_id, throws, roster_available,
                          legal_to_enter_now, unavailable_reason}]
  legal_actions: LegalActions
  lookup_rules: {cost_pa, remaining_free_lookups_this_boundary}
  acquired_evidence: list[LookupReceipt]
  agent_notes: str
```

`PublicGameState` includes inning/half, score, bases, outs remaining, cumulative exposure, current-pitcher exposure, pitch totals, and completed PAs. At ordinary between-PA decisions the count is 0–0. The full ordered lineup supports queries beyond the next hitter.

**Proposed information budget:** current-pitcher game splits are automatically visible. Bullpen identity/handedness/eligibility is free. Detailed historical candidate/batter profiles and candidate usage/health arrive through lookup. Do not dump every reliever's detailed profile into the default observation, which would remove much of the lookup's value. Whether workload should also be free is an explicit future treatment.

Previous lookup receipts persist for the episode. They are snapshots, not live-updating subscriptions. The current pitcher's public game statistics do refresh automatically. Keep all authoritative data separate from bounded, model-written notes; the model cannot overwrite counters or tool receipts.

Use a consistent deterministic serialization order and identical evidence content for all agents within a condition. Do not silently truncate acquired evidence. Select episodes and context budgets that fit the smallest participating context window, or declare a separate fixed eviction policy before evaluation. Long prompts and formatting failures are measured rather than hidden.

## 4. Action and tool contracts

Action is a discriminated union. Exactly one action is accepted per model response.

```json
{
  "action": "check_at_bat",
  "pitcher_id": "reliever_03",
  "batter_id": "batter_07",
  "reason": "Compare this reliever with the left-handed hitter due up later.",
  "notes": "One lookup before reconsidering the incumbent."
}
```

```json
{
  "action": "switch_pitcher",
  "pitcher_id": "reliever_03",
  "reason": "Use the retrieved profile for the upcoming sequence.",
  "notes": "Reassess after the next plate appearance."
}
```

```json
{
  "action": "continue",
  "reason": "Keep the incumbent for the next batter.",
  "notes": "No new information requested."
}
```

```json
{
  "action": "intentional_walk",
  "reason": "Concede first base to avoid this matchup before researching a reliever.",
  "notes": "Reassess against the next scheduled hitter."
}
```

Reason is short and optional; do not request chain-of-thought. Proposed limits: reason 400 characters, notes 1,500 characters. The prompt explains all operational rules, including the one-PA cost.

| Action | Validated behavior | Time |
| --- | --- | --- |
| `check_at_bat` | Snapshot selected profiles, simulate the next actual PA with the incumbent, then deliver the receipt and refreshed state | 1 PA in the main treatment |
| `switch_pitcher` | Replace incumbent with an available, eligible, not-previously-used reliever | 0 PA |
| `continue` | Simulate the next actual PA with the incumbent | 1 PA |
| `intentional_walk` | Award first base to the next scheduled batter; force runners as required; refresh the observation | Completes 1 PA immediately; 0 pitches; no additional waiting PA |

The selected lookup pitcher may be roster-available but temporarily ineligible to enter under a future ruleset. Lookups do not override eligibility. The target batter must be in the opposing lineup. Looking up a batter farther ahead does not skip the current batter.

### Immediate intentional walk

`intentional_walk` needs no target argument: it applies only to the next scheduled batter at the current between-PA decision boundary. It is legal while the episode is active, including with occupied or loaded bases. No prior lookup is required.

Reuse the ordinary walk's forced-runner advancement function. Charge the new runner to the incumbent, preserve responsibility for existing runners, increment completed-PA counters and the relevant pitcher/batter intentional-walk split counters, advance the batting order once, and keep the count at 0–0. Record a distinct PA event. Pitches, pitch-type counts, thrown-ball/strike counts, hits, outs, and outs+hits exposure do not increase. A loaded-bases intentional walk scores one run. It cannot itself end the half-inning by recording an out.

The action is immediate in the sense of avoiding a pitched confrontation, not in the sense of leaving the batting order or game state unchanged. Return the new observation immediately, reset per-PA boundary budgets, and allow another decision. It consumes no simulator random draws in the deterministic runner ruleset. Consecutive intentional walks are allowed; they accumulate runners and runs, and the existing episode decision limit remains an explicit truncation guard.

Example: dangerous hitter A is next, with B following. The agent can intentionally walk A, then request a lookup. The lookup still costs a full simulated PA with the incumbent against B. The walk supplies no information and does not pay or shorten a later lookup's cost. Once a paid lookup begins, its PA is atomic: there is no mid-lookup intentional-walk interruption. Thus the added flexibility exchanges an immediate baserunner for avoiding A's pitched matchup, while keeping the information-delay rule intact.

### Atomic paid lookup

1. Validate arguments and assign a unique action execution key.
2. Snapshot both profiles at the pre-action event sequence.
3. Advance exactly one actual PA with the incumbent and scheduled next batter.
4. Deliver the snapshotted information, the elapsed-PA summary, and post-action state.
5. Persist the complete transition once.

The agent cannot read the result and cancel the elapsed PA. If that PA completes the last defensive half-inning, still record the lookup and its result, then terminate; there is no further agent action. If it ends an earlier inning, the next decision is at the next defensive boundary under the episode convention. An inning-ending PA still costs one PA.

```text
LookupReceipt:
  lookup_id: str
  pitcher_id: str
  batter_id: str
  queried_at_event_seq: int
  delivered_at_event_seq: int
  target_lineup_distance_at_request: int  # next batter = 0, then 1..8
  pitcher: PitcherProfile
  batter: BatterProfile
  cost_pa: int
  elapsed_pa_ids: list[str]
```

The lookup returns evidence only: no “best pitcher,” forecasted win probability, secretly sampled audition result, or outcome-model probability vector.

**Important information-cost consequence:** profiles are separable. Once the agent has seen pitcher A and batter X, it can reuse those profiles; once it also knows batter Y, it can compare A with Y without another lookup. This follows from the requested schema and is allowed. The tool is not a pair-specific oracle. Log newly revealed pitcher IDs and batter IDs, not only pair counts. A repeated paid query still costs one PA; rereading an existing receipt is free.

### Zero-time actions and validation

After a successful switch, provide a refreshed observation at the same PA boundary. Proposed rule: allow at most one switch before a PA advances, avoiding chains of zero-time substitutions. The agent can still look up, continue, or intentionally walk the next batter. An intentional walk completes a PA and therefore opens a new boundary. This is a benchmark rule, not a claim about MLB regulations.

For the optional free-information treatment, keep a finite per-boundary cap (proposed: 8 lookups). A free lookup updates only evidence, not pitcher/batter statistics or RNG. After the cap, require switch, continue, or intentional walk, subject to the switch limit. Reset boundary budgets only after a PA completes, including an intentional walk. Report cap hits and check that the cap is not driving the substantive result.

Invalid actions and parsing errors consume no baseball time and no simulator RNG. Permit one structured repair attempt with the error message. After repeated failure, mark the episode failed and report it; do not silently turn failures into `continue`. Log all model attempts, token usage, and latency. Infrastructure retries reuse the same action key and never replay a committed PA.

## 5. Simple pitch simulator

### 5.1 Sampling

At each pitch:

1. Resolve pitcher hand and batter side.
2. Sample a pitch type from the pitcher's repertoire frequencies for that batter side, using documented fallbacks if sparse.
3. Obtain a five-outcome probability vector for ball, strike, foul, hit, and in-play out.
4. Draw one categorical outcome.
5. Apply the count/base/out/score transition and emit the event.

Independent Poisson counts are unnecessary when exactly one outcome is needed per pitch. The user's original competing-event intuition can be implemented as normalized rates with one categorical draw. Do not introduce continuous waiting times.

**Proposed baseline probability rule:** for category `k`, smooth each side's counts toward a frozen reference distribution `q`:

```text
p_pitcher[k] = (c_pitcher[k] + alpha * q[k]) / (n_pitcher + alpha)
p_batter[k]  = (c_batter[k]  + alpha * q[k]) / (n_batter  + alpha)
p_matchup[k] = w * p_pitcher[k] + (1 - w) * p_batter[k]
```

Initial configurable values: `alpha = 20` pseudo-pitches and `w = 0.5`. These are engineering defaults, not empirically validated estimates. The reference distribution must be valid and pre-cutoff, with positive support for terminating outcomes. Run basic calibration checks and a modest sensitivity check before interpreting performance.

The pitcher table may condition on pitch type and batter side. Batter frequencies condition on pitcher hand and remain aggregated over pitch type for this baseline. For absent pitcher-type data, back off to the pitcher's handedness split, then overall, then reference. For absent batter splits, back off to overall, then reference. Store the selected fallback level in diagnostic logs. Pitch-type usage has its own fallback; do not sample zero-use unsupported types merely because the registry contains them.

This mixture is deliberately simple. It does not by itself encode swing-plane/IVB compatibility, velocity effects, count-dependent approach, or fatigue. Those should be separately versioned probability providers, not undocumented adjustments scattered through the reducer.

### 5.2 Count and terminal outcomes

| Pitch event | Transition |
| --- | --- |
| Ball with fewer than 3 balls | Increment balls. |
| Fourth ball | Walk; force runners as required; finish PA. |
| Strike with fewer than 2 strikes | Increment strikes. |
| Third strike | Strikeout; record one out; finish PA. |
| Foul with fewer than 2 strikes | Increment strikes. |
| Foul with 2 strikes | Count unchanged; pitch total still increases. |
| Hit | Sample hit subtype; move runners and score; finish PA. |
| In-play out | Record out(s) under the selected baserunning rule; finish PA. |

After a completed PA, advance batting order exactly once, increment completed-PA counters, and reset balls/strikes. Then finalize an inning if needed. Fouls allow PAs longer than six pitches. A generous safety limit may mark an episode truncated; never silently force an out to end a long PA.

### 5.3 Baserunning defaults

Use a separately testable runner reducer:

- Walk: force only the runners who must advance; with loaded bases exactly one run scores.
- Intentional walk: reuse the same forced advancement directly, with zero pitches and a separate PA event; no new runner algorithm or probability model.
- Single: each existing runner advances one base; batter occupies first.
- Double: each existing runner advances two bases; batter occupies second.
- Triple: all existing runners score; batter occupies third.
- Home run: all runners and batter score.
- Basic in-play out: one out, runners hold.
- Strikeout: one out, runners hold.

Sample hit subtype from a frozen empirical distribution or clearly labeled synthetic fixture distribution. The deterministic advancement above is a proposed simplification. All subtype and advancement choices belong in a versioned ruleset.

Support runner identity now so steals, caught stealing, inherited runners, and more detailed advancement can be added later. Do not enable steals without specifying the PA-cost boundary: a caught-stealing third out can end an inning without completing the current PA. Initial `steals_enabled = false` keeps strict one-completed-PA lookup semantics unambiguous. The state can also initialize with any legitimate occupied-base pattern.

When adding double plays, distinguish pitch outcomes from actual outs and resolve third-out/run timing explicitly. Do not allow a two-out state to record a fourth out. No new baserunning behavior should be guessed from a generic `in_play_outs` count.

### 5.4 Shape and fatigue treatments

Begin with one ordinary `outcome_probs` function implementing the pooled-frequency rule. A later shape-conditioned function can use coarse pre-cutoff tables over pitch family, movement, velocity, location, and hitter-profile groups, with explicit sparse-data fallback. Add that function when the treatment is actually implemented; no provider-class hierarchy or neural model is required.

In the baseline, shape may influence the agent's choice while having no direct causal effect in the simulator beyond correlations already represented by observed rates. Performance gains therefore do not establish learned swing–pitch compatibility. Measure behavioral response to shape disclosure separately. Apply the same caution to workload and IL information while fatigue/injury effects are off.

Do not secretly retune outcomes to reward plausible-sounding baseball reasoning. Preserve the main goal of comparing evidence use.

## 6. Minimal code architecture

A plain Python loop owns the experiment. Prefer direct functions over framework nodes, plugin registries, abstract base classes, or dependency injection. Preserve logical boundaries without making each boundary a package.

Start with four Python files, plus this design/README, one JSON configuration, fixture data, and output files:

| File | Responsibility |
| --- | --- |
| `bench.py` | State, rules, pitch/PA simulation, tool execution, main loop, save/resume, basic summary |
| `agent.py` | Common briefing, observation formatting, one model call function, response parsing, scripted baselines |
| `data.py` | Offline normalization, profiles, team metadata, dataset splits, factual curriculum generation |
| `test_bench.py` | Focused count/base/lookup/cutoff/replay tests using `unittest` |

The core simulator can run with the standard library alone: `random.Random`, `dataclasses` if useful, `json`, `argparse`, `pathlib`, `collections`, `statistics`, and `unittest`. Use `random.Random.choices` for categorical sampling. No generic probability-provider hierarchy is needed: use an ordinary function and add another only when implementing another treatment.

Minimal interfaces:

```python
def observe(state, evidence, notes, config): ...
def decide(observation, model_config): ...
def validate_action(action, state, config): ...
def step(state, action, profiles, rng, config): ...
def run_episode(initial_state, profiles, decide_fn, config): ...
```

These are signatures, not implemented functions. `step` returns updated state, emitted events, and any lookup receipt. `decide_fn` can be a scripted policy or a model call. Tests can supply a fixed sequence of actions. Keep arithmetic and state transitions out of prompts and model adapters.

The loop is observe → one model response → validate → execute once → save → repeat. Paid lookup is one atomic environment action. No agent autonomy outside these defined actions, no concurrent tool calls within a game, and no LLM call per pitch.

Use a fresh state value per step where practical so failed validation or an uncommitted transition cannot mutate the last saved boundary. Save a JSON snapshot with state, acquired evidence, notes, event sequence, boundary budgets, configuration hash, action ID, and RNG state. Write to a temporary file then atomically replace the checkpoint. Store events in JSONL with stable IDs; the checkpoint's committed sequence is authoritative on resume. Discard or deduplicate uncommitted tail events. Capture an accepted model response before execution so resuming does not silently resample the decision. This needs a few explicit checks, not a transaction platform.

Split a file only after it becomes hard to read or independently useful elsewhere. Add `train.py` only when post-training is actually selected, calling an existing trainer rather than implementing a new trainer. A standalone `analyze.py` is optional if analysis outgrows the basic summary.

LangGraph is not needed to run the benchmark. If later added for portfolio coverage, a small optional adapter must genuinely execute the same tested observe/decide/step loop and have a reproducible example. An unused import is not a meaningful integration. Do not implement a second simulator or introduce it into the main dependency path merely for resume keywords.

## 7. Minimal dependencies

- Core engine, files, configuration, and tests: Python standard library.
- Model inference: one chosen client SDK or one local inference backend, selected after the target model. Do not require both local and hosted inference implementations initially.
- Data preparation: begin with CSV/JSON downloads and standard-library readers. Add pandas, a baseball-data client, or BeautifulSoup only if a concrete input format makes it worthwhile; do not install all three by default.
- Analysis: basic counts and paired differences first; add SciPy only when a specific statistical test is needed. `pystats` was a suggested testing library, not a required dependency.
- Post-training: an existing supported training stack, separate from the harness. Torch and model/adapter libraries are training dependencies only when actually needed. Do not hand-roll training or quantization to reduce a dependency count.

Use JSON configuration instead of introducing YAML support. Type hints and a few explicit field/range checks are sufficient for the small action contract. No LangChain, Pydantic, database, cloud tracing service, web framework, or GPU dependency is required for the smoke run.

Make model ID/revision, endpoint, quantization, sampling settings, timeout, and token limit configurable. Record resolved versions and training settings for actual runs. The agent receives no shell, arbitrary Python, browser, or statistical-test tool.

## 8. Data preparation and leakage controls

Start with synthetic fixtures that cover the rules. Then add one frozen real-data import path. The first deliverable should not wait for scraping several seasons.

Normalized internal pitch records should retain game/PA/pitch IDs, timestamp/order, pitcher/batter IDs and handedness, pitch type, raw outcome description, mapped outcome, optional shape values, terminal PA result, and source provenance. Keep event-level runner/outs information when available. Use source IDs consistently; retain an explicit crosswalk if combining datasets. Retain zero-pitch intentional walks as PA events outside the physical-pitch table; do not drop them or inject synthetic balls into pitch frequencies. If historical intentional walks contain actual pitches, retain those observed pitches under the documented mapping and count the terminal intentional walk once.

A mapping table must document real-data events outside the five-outcome simulation: HBP, errors, fielder's choices, interference, automatic count awards, and unusual foul events. Do not silently call an error a hit or assume every non-hit ball in play produces one out. Proposed initial approach: build probability tables from supported classified physical-pitch outcomes, report excluded counts/rates, and use only supported scenarios for the first benchmark. Profile `pitch_count` then means classified supported pitches; preserve raw/excluded counts in provenance. If excluded rates are material, extend the schema before drawing conclusions from apparent workload.

Freeze data at the episode cutoff, including health/workload and swing profiles. A full-season leaderboard downloaded after the scenario date may leak future performance even if the game prefix is correct. Never use the historical game's eventual result or later bullpen choices as hidden guidance for the agent.

Store a manifest with data snapshot ID/hash, raw sources and fetch time, statistical window, branch point, mapping version, pitch registry version, ruleset version, probability-provider version, eligibility policy, and exclusions. No network retrieval during benchmark episodes.

Anonymized stable IDs are a useful optional condition to reduce reliance on famous-player knowledge. Name visibility must be consistent within a condition. Avoid claims about abstract reasoning if one model recognizes the real game and another does not.

### 8.1 Temporal structure and team identity

Keep one canonical event hierarchy:

```text
season → game → half-inning → plate appearance → pitch
```

Team/franchise, division, park, and pitcher appearance are annotations or views over that hierarchy, not additional exclusive parents. Each game belongs to two team-season views. Store it once by `game_id`; a defensive scenario additionally identifies the controlled team. Pitcher appearances can span innings, so do not force them under a single half-inning node.

Use this small dated identity table:

| Field | Meaning |
| --- | --- |
| `franchise_id` | Internal stable continuity identifier; verify the historical crosswalk rather than assume source team codes are stable |
| `team_season_id` | Franchise-season identity for sampling and summaries |
| `valid_from`, `valid_to` | Effective interval of metadata |
| `source_team_id` | Dataset code with source namespace |
| `league_id`, `division_id` | Membership at the game date |
| `display_name`, `city` | Labels at the game date, not primary keys |
| `park_id` | Actual game venue, stored on the game; home venue can change within a season |

Preserve `player_id` across trades and store dated team membership. A franchise rename or move should not accidentally create a new independent team, while a stable franchise ID does not make its roster, park, or era unchanged. Keep identities explicit and initially restrict to a compact time range with adequate measurement coverage.

Season membership alone is not a sequential-learning objective. For the initial experiment, seasons organize data; game scenarios are episodes; PAs are decision/delay units; pitches are simulated transitions. Freeze model weights during evaluation and reset episode notes/evidence between games. Training a long-horizon season manager would additionally require cross-game resource carryover and a season-level objective; do not introduce that implicitly.

### 8.2 Three separate uses of data

Keep these logically distinct, even if they live in one script:

1. **Environment data:** frozen profiles, workload histories, simulator frequencies, and scenario initialization.
2. **Agent learning data:** examples used for a common briefing or actual weight updates.
3. **Evaluation data:** untouched game/scenario groups used to compare decisions.

An environment may know a player's profile for simulating an episode while the agent must pay to retrieve it. That is separate from whether an agent memorized that player during post-training. Report overlap; do not equate a held-out game with an unseen player.

### 8.3 Inducing shared baseball knowledge — proposed research plan

The user wants a common conceptual foundation for open-weight models. This has not yet been selected as a specific training method. Begin with explicit instruction and competency measurement, then decide whether updating weights adds value.

**Stage A: common briefing, no weight updates.** Put one concise, versioned primer in the model prompt. Cover:

- Objects and rules: pitcher/batter/runner, handedness, inning/outs/count, lineups, bullpen availability, innings notation, and the benchmark's simplified rules.
- Data interpretation: ball/strike versus walk/strikeout, sample sizes, split denominators, recent workload, missingness, and historical versus current-game evidence.
- Baseball concepts: repertoire, movement versus velocity, pitch shape versus swing shape, platoon context, fatigue as an uncertain relationship, relief roles, and short-term results versus underlying ability. Clearly distinguish empirical tendencies from universal rules and from effects actually implemented by this simulator.
- Tool semantics: one PA elapses before paid lookup evidence can inform another action; the selected hitter is not necessarily the next hitter; past profiles can be reused.

Avoid a prescribed replacement threshold or reliever ranking. The curriculum should enable the model to interpret the decision, while leaving the decision policy available for study. Even factual fine-tuning can change behavior; preserve the common-briefing-only baseline rather than assuming the intervention leaves preferences untouched.

Measure comprehension with held-out factual and procedural probes: identify the correct handedness split, convert 5.2 IP to 17 outs, calculate a rate with the proper denominator, distinguish null from zero, resolve a supplied pitch sequence, identify legal actions, and identify which actual batter the incumbent faces during a future-hitter lookup. Include paraphrases and unseen numbers rather than exact training repeats. Scores describe competence; identical briefing does not guarantee identical internal knowledge.

**Stage B: supervised fine-tuning if needed or as a declared treatment.** Generate instruction/answer examples from the same engine and normalized records. Derive mechanical targets programmatically and inspect a sample. Historical examples can teach profile extraction, state reconstruction, comparison of observed rates/sample sizes, and factual summaries. Add curated explanations for concepts that cannot be verified by arithmetic alone. Do not use another LLM's unchecked baseball advice as ground truth.

For example, input supplies a count and an observed pitch sequence; target gives the resulting count/outs. Another input supplies two splits with denominators; target identifies the relevant split and reports both the observed difference and its sample sizes. It need not choose a pitcher.

Do not train on manager replacement choices as optimal labels in this foundational stage. That would make imitation of historical policy part of the treatment. Likewise, unmasked action transcripts can teach imitation even without an explicit “best action” label. Exclude those targets or declare a separate imitation condition. Do not train on evaluation episodes or use evaluation probes to repair training examples.

LoRA/QLoRA are optional ways to run weight updates, not the curriculum itself. QLoRA trains adapters through a frozen quantized model [S8]. Choose a supported trainer only after selecting a model and compute budget; no claim is made that a particular model/context fits an 8 GB GPU.

**Stage C: broader domain adaptation only if justified.** Continued language-model training on baseball text or serialized events is possible, but it is less targeted than verified instruction examples for these specific schema gaps. Domain/task-adaptive pretraining has research precedent [S7]; that does not establish its benefit for this benchmark or guarantee equal knowledge across different models. Keep it optional.

For each base model, retain the original checkpoint and fork independent curriculum variants from the same starting checkpoint. Compare common-briefing-only against briefing-plus-post-training. Record dataset hash, objective, exact examples, trained tokens, optimizer settings, adapter settings, and revisions. Choose training settings on development data, not the test games. Fixed exposure and train-to-competence are different experiments: the latter spends different training budgets and must report that difference.

### 8.4 All-team versus division-focused exposure

Treat corpus scope as an experimental variable, independent of the knowledge-teaching method. Use the same task templates and target types across scopes.

| Training scope | Sampling rule | Question |
| --- | --- | --- |
| All-team | Balanced controlled-team-season sampling across the eligible league corpus | How does broad experience affect decisions and transfer? |
| One division | Controlled team belongs to division D at game time; balance its team-seasons | Does concentrated experience improve familiar contexts or narrow generalization? |
| Leave one division out | Controlled teams exclude D; evaluate D later | Does behavior transfer to held-out controlled-team contexts? |

These default definitions concern the **controlled team**, not every entity in the records. Teams play outside their division, so division-focused training ordinarily includes outside batters, pitchers in source records, and parks. An “unseen division” claim requires stronger filtering: exclude games involving any held-out division team from training, audit aggregate-profile provenance as well as explicit events, and separately measure player overlap due to trades. Even then pretrained models may already know the teams; describe exposure as held out from this project's post-training.

For a scope comparison within one base model, hold the calendar window, task mixture, briefing, optimizer recipe, and trained-token budget as constant as practical. Report unique games/examples and repetitions too; equal tokens with repeated narrow data is not equal diversity. Match tokens rather than epochs over differently sized corpora. Begin with one all-team arm and one division arm; expand to six division arms only after a useful pilot.

Sample team-season, then game, then a bounded number of windows. Do not sample all PAs uniformly and accidentally overweight long/hit-heavy games. Give both arms comparable mixes of inning, score margin, workload, base occupancy, handedness, and bullpen availability, without conditioning on future outcomes. Keep within-game windows chronologically ordered where a target needs history; reset state between unrelated examples.

Evaluation should include later games from the training division and later games from other divisions. Eventually a training-division × evaluation-division matrix can show specialization or transfer. A same-sized random group of teams is a useful later control: if it performs similarly to a division grouping, the effect may concern sample size or roster concentration rather than division structure.

City moves are only one possible source of distribution shift. Track park, roster/manager era where available, rule/schedule era, and measurement availability. MLB reduced scheduled divisional games in 2023 from 76 to 52 [S9], so division-based exposure depends on the selected date range. A division is a sampling stratum, not an inherently homogeneous baseball style.

### 8.5 Splitting and freezing

Choose chronological train/development/test windows. Keep an entire `game_id`, every PA/window derived from it, both team views, and every simulated branch in the same split. When adjacency matters for bullpen workload or opponent familiarity, group by series or use a temporal gap. Do not randomly split PAs from the same game.

Freeze models during the test run. Player profiles may use only evidence available at their decision cutoff; no held-out future outcomes enter simulator fitting or derived training examples. A rolling profile that includes earlier test-period observations is an explicit online-information protocol, not additional weight training. For the simplest first evaluation, freeze profile snapshots at the declared cutoff for each scenario and log all sources.

Temporal-only test splits can answer later-game generalization with familiar players. Team/division holdouts answer different questions and may still share players. Use independent flags rather than a single misleading “unseen” label. Keep factual/procedural probe holdouts separate from strategic test games, and do not drop low-scoring models after seeing their strategy results.

## 9. Experiment configuration and evaluation

Illustrative configuration; values here are proposed initial defaults:

```json
{
  "schema_version": "0.3",
  "episode_mode": "defensive_horizon",
  "lookup": {
    "cost_pa": 1,
    "max_free_per_boundary": 8,
    "repeat_paid_query_costs_pa": true
  },
  "rules": {
    "intentional_walk_enabled": true,
    "max_switches_per_boundary": 1,
    "allow_pitcher_reentry": false,
    "steals_enabled": false,
    "double_plays_enabled": false,
    "enforce_three_batter_minimum": false,
    "advancement": "deterministic_bases"
  },
  "probabilities": {
    "provider": "pooled_frequency",
    "pitcher_weight": 0.5,
    "prior_strength": 20,
    "fatigue_enabled": false
  },
  "observation": {
    "show_shapes": true,
    "show_player_names": false,
    "notes_max_chars": 1500,
    "reason_max_chars": 400
  },
  "execution": {
    "max_repair_attempts": 1,
    "max_decisions": 500,
    "max_pitches_per_pa": 100,
    "simulator_seed": 1234,
    "model_seed": 5678
  },
  "knowledge": {
    "mode": "common_briefing",
    "curriculum_id": null,
    "scope": "all_team",
    "weights_frozen_during_evaluation": true
  }
}
```

Guards cause explicit `truncated`/`failed` status; they are not normal baseball endings. Model seed support is backend-dependent and must be logged. Never promise bitwise LLM determinism.

### Initial comparisons

Start with a fixed scenario set, one model, deterministic baselines, and a common briefing. Measure comprehension before choosing a post-training treatment. Pilot one all-team versus one division-focused curriculum within the same base model, retaining the no-weight-update baseline. Broader model comparisons, paid/free information, and shapes-visible/hidden controls can follow independently. Hidden shapes must be removed consistently from lookups and ordinary observations; keep the simulator fixed across visibility conditions.

Baselines can include never-switch, a configured pitch-count threshold with a fixed reliever ordering, and a seeded random legal policy. Their information access must be explicit. Any baseline allowed hidden probabilities is a privileged comparator, not a fair same-information agent.

### Metrics

| Group | Measures |
| --- | --- |
| Replacement | First replacement boundary; inning/outs, pitches, PAs, and outs+hits at replacement; chosen reliever; number of changes |
| Information | Lookup count; unique candidates; unique batters; new profiles revealed; lineup distance; repeated queries; lookups with no later action opportunity |
| Search pattern | Multiple pitchers for one batter versus multiple batters for one pitcher; paid PAs before first replacement |
| Intentional walks | Frequency, handedness/matchup, base/score state, immediate forced runs, and use before a lookup or replacement; report separately from ordinary walks |
| Evidence sensitivity | Action changes in matched static decision states when workload, results, handedness sequence, bullpen options, or disclosure changes |
| Outcomes | Runs/hits/walks/strikeouts allowed; pitches and PAs; results conditional on simulator/ruleset |
| Reliability/cost | Invalid responses, repairs, failures, truncations, context size, generated/input tokens, latency |

Keep timing for “never replaced” separate or censored at the episode horizon; do not report the final pitch count as if a replacement occurred. Unavailable relievers are not part of that decision's choice set.

Use scenario/game as the unit of comparison or clustering. Repeated seeds and multiple PAs from one game are not independent games. Report paired effect sizes and uncertainty across scenarios before emphasizing p-values. Statistical testing happens after episodes and is never a tool available to agents. Choose specific hypotheses before a large run; broad exploratory plots can precede that.

Use separate simulator and model random streams. With identical state, action sequence, simulator seed, and code version, engine replay should match exactly. Across different agent actions, the same initial RNG seed does not imply identical baseball opportunities: branches change the number of pitches, count states, and batters. Shared seeds provide pairing, not magical counterfactual equality. If stronger common-random-number coupling is later needed, implement indexed event streams and document the coupling; do not overclaim from ordinary sequential seeds.

For clean evidence-sensitivity experiments, use matched single-decision probes in addition to branching episodes. Change one coherent feature or feature bundle, retain valid counts and denominators, and hold unrelated state fixed. Do not alter pitch count while leaving its component counts inconsistent, or present a workload manipulation as a physiological effect when fatigue is disabled.

## 10. Implementation order and acceptance criteria

### Milestone 1 — executable offline core

Implement schemas, synthetic profiles, one scenario, categorical simulator, runner/count reducers, `continue`, switching, intentional walk, and paid lookup. A scripted agent must finish an episode with no LLM, network, or GPU. Export events and a readable summary.

Meaningful tests must show:

- Four balls cause one walk, not four separate baserunning events.
- Three non-foul strikes cause one strikeout; a two-strike foul leaves the count unchanged.
- Loaded-bases walk scores one forced run without deleting or duplicating runners.
- Intentional walk advances the batter/PA and forced runners immediately, without changing pitches, thrown balls, hits, outs, outs+hits, or RNG; record it once in pitcher/batter split statistics.
- With loaded bases it forces one run; with only second base occupied it leaves that runner on second and puts the batter on first.
- Intentional walk followed by paid lookup makes the incumbent pitch to the following batter; the walk does not satisfy the lookup's PA cost.
- A home run clears occupied bases and scores all runners plus batter.
- Third out clears bases/count, preserves batting order progress, and advances the defensive horizon correctly.
- One paid lookup simulates exactly one PA with the old pitcher against the actual next batter, even when the query targets a later hitter.
- Switching costs zero PA, preserves bases/counters, and makes the replacement throw the next pitch.
- A lookup on the last PA is logged and charged without granting another action.
- Outs+hits increments by the actual defined events and does not substitute for PA timing.

### Milestone 2 — agent loop and shared briefing

Add one configurable model function to the ordinary Python loop, mock responses, bounded repairs, a common baseball/tool briefing, held-out comprehension probes, and per-run metadata. Save state and RNG at committed decision boundaries; do not add an orchestration framework.

Verify no double-advance after retry/resume, no RNG consumption on free lookup or invalid action, no unbounded zero-time action loops, and no hidden candidate profiles in the initial observation. Test the adapter using mock responses before a paid request. Use comprehension probes to diagnose rules/interface problems before interpreting strategy differences.

### Milestone 3 — frozen data pipeline

Add normalization, cutoff-aware aggregation, usage/health provenance, repertoire registry, missing-value handling, and a small real-data snapshot if available.

Verify L/R/both count identities, metric-specific weighted aggregates, unknown versus zero handling, innings-as-outs conversion, season/game subset accounting, supported-outcome mapping, and the absence of post-cutoff records. Existing real data should not be claimed complete merely because a CSV loaded successfully.

### Milestone 4 — comparison runner and curriculum pilot

Run deterministic baselines and one configured model once authorized. Establish the common-briefing baseline first. If needed, add a small supervised curriculum pilot; only then compare all-team and division-focused exposure. Add paid/free information and shape disclosure as separate comparisons later, rather than multiplying every condition immediately. Keep incomplete/failed episodes visible.

### Milestone 5 — optional extensions

Only after the above works: explicit fatigue, shape-conditioned frequencies, more realistic runner movement, steals, MLB eligibility rules, full-game/season budgets, or multi-agent coordination. These are research extensions, not prerequisites for the first prototype.

### Suggested CLI contract

Implement commands equivalent to:

```bash
python bench.py smoke
python data.py validate --snapshot data/demo
python bench.py run --config config.json
python bench.py summarize --run-dir runs/example
python -m unittest test_bench
```

These are target commands, not claims that a package currently exists. Use the small flat layout in section 6. A completed first handoff should include working setup instructions, fixture data, exact reproducible smoke commands, test results, and a short list of remaining assumptions. Avoid implementing a dashboard before the episode runner and logs work.

## 11. Decisions to revisit after the first smoke run

These are implementation notes, not a request to stop and ask all questions now:

1. Is the defensive fixed horizon sufficient, or should episodes run to an actual game ending?
2. Should recent bullpen workload be visible for free, or obtained only through lookup?
3. Is “time since IL return” the intended injury field, or should last stint start/end both be supplied?
4. Which seasons have usable swing tracking and cutoff-safe splits? Missing measurements must remain explicit.
5. Should shape and workload be descriptive evidence only initially, or enter the generative probabilities in the main experiment?
6. Which real-data rare events merit schema extensions rather than exclusion?
7. Does the one-switch-per-boundary simplification suffice, or should a documented era-specific eligibility ruleset replace it?
8. After observing prompt sizes, should the public JSON carry all zero-count pitch types, or should a fixed lossless compact representation be a separately declared format? Preserve the requested full dictionary initially.

## 12. Primary documentation checked for this handoff

These sources support source-field definitions and optional framework/training methods. The experimental choices and code architecture above are proposed for this project, not claims endorsed by the sources. Consult current API references during implementation.

- **S1:** [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- **S2:** [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- **S3:** [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- **S4:** [Statcast CSV data dictionary](https://baseballsavant.mlb.com/csv-docs)
- **S5:** [Statcast pitch movement](https://baseballsavant.mlb.com/leaderboard/pitch-movement)
- **S6:** [MLB ideal attack angle definition](https://www.mlb.com/glossary/statcast/ideal-attack-angle)
- **S7:** [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/)
- **S8:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **S9:** [MLB announces the 2023 schedule](https://www.mlb.com/press-release/press-release-mlb-announces-2023-regular-season-schedule)

## Paste into Claude Code

> Read `BaseballBench_Claude_Code_Handoff.md` and inspect the repository. Prioritize simplicity: few files, direct functions, standard library first, and a plain Python loop. Use section 6; do not add frameworks, generic abstraction layers, or a database without concrete need. Build the offline core with fixtures and a scripted agent, then one model adapter and a common baseball/tool briefing. Preserve strict one-PA lookup timing and shared outcome schemas. Follow section 8 for cutoff-safe data, factual curriculum, and team/division curation; post-training methods remain proposals until selected. Run the meaningful tests. Do not use paid inference, download large weights, or start training without authorization. Report working commands and remaining assumptions.
