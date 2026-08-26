# CONTRACT

Shared interface between the three agents working this repo. If you change
anything in here, change it here first, then tell the other two.

## The question

How many milliseconds of tolerated latency does one non-verbal cue buy?

Everything downstream serves that one number. `latency_ms` is the x-axis,
`aliveness_1_7` is the y-axis, `cue` is the line family, and the exchange
rate is the horizontal shift between a cue line and the `cue=none` line.

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
| `exchange_id` | str | the dialogue turn this stimulus is a version of; the same `exchange_id` appears once per cell |
| `prompt_text` | str | what the human says |
| `response_text` | str | what the agent says after the gap |
| `wav_path` | str | repo-relative path to the rendered audio |
| `actual_gap_ms` | float | **measured** silence between prompt offset and response onset, not the nominal value |
| `cue_onset_ms` | float or null | measured cue onset relative to prompt offset; `null` when `cue == "none"` |

Notes that matter for the analysis:

- `latency_ms` is the *design* cell. `actual_gap_ms` is what the listener
  actually heard. The analysis fits on `actual_gap_ms` when it is present
  and falls back to `latency_ms` when it is not, because renderer jitter is
  measurement error on the independent variable and pretending otherwise
  biases the exchange rate toward zero.
- Cue duration is derived as `actual_gap_ms - cue_onset_ms` is *not* the
  cue duration. If cue duration is known, add an optional `cue_dur_ms`
  field; without it the cost side of the cost/benefit table is unreported.
- The design is intended to be crossed: every `exchange_id` should appear at
  every `latency_ms` x `cue` cell. Unbalanced is survivable, unreported
  imbalance is not.

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
  worse. Do not flip either at write time; the analysis reverses `broken`
  where needed.

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
