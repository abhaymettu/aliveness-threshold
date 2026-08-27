# CONTRACT

Shared interface between the three agents working this repo. If you change
anything in here, change it here first, then tell the other two.

## The question

How many milliseconds of tolerated latency does one non-verbal cue buy?

**That number is not estimable from the data this repo has, and the reason is
the finding.** See README.md. The LLM judges' aliveness ratings are flat
against gap inside `cue = none`, so the horizontal shift this section describes
is a ratio with a denominator consistent with zero. The shift estimator has
been removed from `analysis/`. It remains unmeasured in humans; n humans = 0
and `web/rate.html` is the instrument for measuring it.

What downstream still serves: `actual_gap_ms` is the x-axis (never nominal
`latency_ms`), `aliveness_1_7` / `broken_1_7` / `would_wait_again_bool` are the
three DVs, and `cue` is contrasted against `none` only at 800/1200/1600 ms,
the gaps where a cue physically fits and the two share the same wait.

## File ownership

Three agents write here concurrently. Only touch what you own; `git add`
paths explicitly, never `git add -A`.

| Path | Owner |
|---|---|
| `harness/`, `scripts/`, `stimgen/`, `pyproject.toml` | stimulus/harness agent |
| `raters/`, `web/`, `data/ratings.jsonl` | rating agent |
| `data/stimuli.jsonl` | stimulus/harness agent |
| `analysis/`, `figures/`, `README.md`, `CONTRACT.md`, `SIMULATED_*` | analysis agent |
| `demo/` | stimulus/harness agent (analysis agent leaves the placeholder) |

Anything not listed: ask before creating it.

## `data/stimuli.jsonl`

One JSON object per line. One line per rendered audio stimulus.

| field | type | notes |
|---|---|---|
| `stim_id` | str | unique, stable across re-renders |
| `latency_ms` | int | nominal condition, one of `0, 200, 400, 800, 1200, 1600` |
| `cue` | str | one of `none, filled_pause, breath, backchannel, verbal_stall` |
| `exchange_id` | str | the dialogue turn this stimulus is a version of; each one appears once under every `cue` and at 5 of the 6 latencies (see the rotation note below) |
| `prompt_text` | str | what the human says |
| `response_text` | str | what the agent says after the gap |
| `wav_path` | str | repo-relative path to the rendered audio |
| `actual_gap_ms` | float | **measured** silence between prompt offset and response onset, not the nominal value |
| `cue_onset_ms` | float or null | measured cue onset relative to prompt offset; `null` when `cue == "none"` |

Notes that matter for the analysis:

- `latency_ms` is the *design* cell. `actual_gap_ms` is what the listener
  actually heard, and it is what the analysis fits on -- always, without
  exception. A row with no `actual_gap_ms` is **dropped and counted**
  (`ratings_dropped_no_measured_gap` in the results), never backfilled from
  the nominal cell. 36 of the 90 clips have a gap longer than their nominal
  cell because a cue that will not fit forces the gap open; crediting those
  clips with the wait they were designed for rather than the one they have is
  the single easiest way to get this study wrong.
- Cue duration is derived as `actual_gap_ms - cue_onset_ms` is *not* the
  cue duration. If cue duration is known, add an optional `cue_dur_ms`
  field; without it the cost side of the cost/benefit table is unreported.
- **CORRECTED 2026-08-26.** This used to say the design is crossed: "every
  `exchange_id` should appear at every `latency_ms` x `cue` cell". That was
  never true and was never built. 18 exchanges x 30 cells = 540 clips is not
  rateable, so `stimgen/design.py` runs a **rotation**:
  `latency = LATENCIES[(e + c) % 6]`, giving **90 clips**. The properties that
  do hold, all asserted in `stimgen/check.py`:
  every (latency x cue) cell has exactly 3 clips; every exchange appears once
  under **every** cue, so cue is orthogonal to content; every exchange sits at
  **5 of the 6** latencies and which one it skips rotates with the exchange.
  Exchange x latency is NOT fully crossed and cannot be with 90 clips.
  Unbalanced is survivable, unreported imbalance is not.

## `data/ratings.jsonl`

One JSON object per line. One line per (stimulus, rater) judgement.

| field | type | notes |
|---|---|---|
| `stim_id` | str | must join to `data/stimuli.jsonl` |
| `rater_id` | str | stable per rater; LLM raters get one id per model+config |
| `rater_type` | str | `"llm"` or `"human"` |
| `rater_modality` | str | what the rater received, e.g. `"audio"`, `"transcript"`, `"transcript+timing"` |
| `aliveness_1_7` | int | 1-7, primary DV |
| `broken_1_7` | int | 1-7, "the system seemed broken" |
| `would_wait_again_bool` | bool | binary tolerance DV |
| `ts` | str | ISO-8601 timestamp |

Notes that matter for the analysis:

- `rater_modality` is not cosmetic. An LLM rating a transcript cannot hear a
  breath. LLM-vs-human agreement is only interpretable within comparable
  modality, so record it honestly even when it is embarrassing.
- Repeat ratings of the same `stim_id` by the same `rater_id` are allowed
  (they give within-rater reliability). The analysis does not dedupe them.
- Scale direction: higher `aliveness_1_7` is better, higher `broken_1_7` is
  worse, higher `would_wait_again_bool` is better. Do not flip any of them at
  write time. The analysis does not reverse them either -- every contrast is
  reported in the DV's own direction, with the direction named at the number.

## Simulated fixtures

`SIMULATED_stimuli.jsonl` and `SIMULATED_ratings.jsonl` exist so the analysis
can be built before the real data lands. They follow the same schema.

**Hard rule: nothing computed from a `SIMULATED_*` file may appear in
`README.md` or in `figures/`.** The analysis refuses to write into
`figures/` when its input path matches `SIMULATED_*`; it writes to
`figures/simulated/` with a `SIMULATED` stamp burned into every panel
instead. Numbers in the README come from real ratings or the README says the
number does not exist yet.

## Running the analysis

```
.venv/bin/python analysis/run.py --ratings data/ratings.jsonl --stimuli data/stimuli.jsonl
```

Writes `analysis/out/results.json` (real) or `analysis/out/SIMULATED_results.json`.
