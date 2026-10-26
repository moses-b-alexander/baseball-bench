# baseball-bench

Evaluating open-weight LLM agents on baseball pitcher-change decisions.

An agent manages one team's pitching between plate appearances (PAs) using four actions:
`continue`, `switch_pitcher`, `intentional_walk`, and `check_at_bat` — a statistical lookup whose
cost is one actual PA with the incumbent pitcher before the evidence arrives. Pitches are simulated
underneath with a transparent pooled-frequency model. The research object is the agent's
information-gathering and replacement behavior, not the simulator.

The design contract is [DESIGN.md](DESIGN.md) (handoff v0.3). All demo data are synthetic.

## Setup

Python 3.10+ and the standard library only. No install step.

## Commands

```bash
python data.py make-demo                      # regenerate the synthetic snapshot in data/demo (seeded)
python data.py validate --snapshot data/demo  # schema identities, null-vs-zero, cutoff safety, hashes
python bench.py smoke                         # scripted episode exercising every action -> runs/smoke
python bench.py run --config config.json --agent threshold --run-dir runs/example
python bench.py run --agent threshold --run-dir runs/example --resume
python bench.py summarize --run-dir runs/example
python -m unittest test_bench -v
python agent.py briefing                      # print the common briefing given to every model
python agent.py probes --mock                 # comprehension-probe harness with an oracle mock (no network)
```

Agents: `never_switch`, `threshold` (pitch-count rule, first legal reliever in pool order),
`random` (seeded random legal action), `tour` (smoke script), and `model`.

### Running a model

`--agent model` calls any OpenAI-compatible `/chat/completions` endpoint (Ollama, vLLM, llama.cpp
server, LM Studio, or a hosted open-weight provider). Set `model.base_url` and `model.model` in
`config.json` (and `model.api_key_env` to the *name* of an environment variable holding a key, if
needed). One repair attempt is allowed per decision; repeated invalid output marks the episode `failed`.
`python agent.py probes --config config.json` runs the probes against the same endpoint.

## Files

| File | Responsibility |
|---|---|
| `bench.py` | state, rules, pitch/PA simulation, tool execution, episode loop, checkpoint/resume, summary |
| `agent.py` | common briefing, observation formatting, model call + parsing, scripted baselines, probes |
| `data.py` | profile schema helpers, cutoff-safe aggregation, synthetic snapshot generator, validation |
| `test_bench.py` | `unittest` suite |
| `config.json` | experiment configuration (proposed defaults from DESIGN.md §9) |
| `data/demo/` | synthetic raw records (`pitches.csv`, `pa_events.csv`), profiles, frozen simulator tables, scenario, manifest |

A run directory holds `checkpoint.json` (state, evidence, notes, RNG state, pending decision; written
atomically), `events.jsonl` (stable event IDs; trimmed to the committed sequence on resume),
`decisions.jsonl` (every model attempt with tokens and latency), and `summary.json`.

## Implemented ruleset (`rules-0.1`) — not an MLB simulator

- Fixed defensive horizon: the controlled team defends from the scenario state through
  `last_defensive_inning`; its own runs come from a declared schedule between defensive halves.
- Five pitch outcomes (ball, strike, foul, hit, in-play out); count-driven walks and strikeouts.
- Deterministic baserunning; no steals, double plays, errors, HBP, three-batter minimum, or extra-inning runner.
- One switch per PA boundary, no reentry, roster-unavailable pitchers cannot enter.
- Fatigue and pitch/swing shape are visible evidence only; they do not change simulated probabilities.
- Simulated pitches update counts but never fabricate shape or swing measurements.
