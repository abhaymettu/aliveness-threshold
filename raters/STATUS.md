# STATUS — stimulus + rating arms

Last updated at the end of the first full judge run.

## Collected

- **90 stimuli**, all rendered and on disk (`data/stimuli.jsonl`, wavs in
  `stimuli/`, gitignored). 3 per (latency × cue) cell, all 30 cells covered.
- **540 LLM ratings**, `data/ratings.jsonl`. 6 personas × 90 stimuli, one
  `rater_id` each, no averaging. Every cell covered by every persona.
- **0 uncovered conditions.** Nothing is missing from the LLM arm.
- **0 human ratings**, deliberately. There is no human arm in this study.
  `web/rate.html` ships as the instrument for the follow-up.

## Caveats that must travel with the numbers

- `rater_modality` is `transcript+timing`, never `audio`. The judges read a
  timing description built by `raters/render.py`. They did not listen.
- 36 of 90 clips have `actual_gap_ms > latency_ms`: a cue cannot fit inside a
  nominal 0/200/400 ms gap. All of 800/1200/1600 are exact. Fit on
  `actual_gap_ms`.
- Audio is `stimgen/harness_stub.py` (macOS `say` + stdlib), not the real
  `harness/`, which was still an empty directory at the time of this run.
  Every row carries `synth_backend` so the two can never be pooled silently.
- The judges' aliveness ratings look cue-driven rather than timing-driven —
  see the long note at the top of `raters/judge.py` before quoting an
  exchange rate off these rows.

## Reproduce

```
.venv/bin/python -m stimgen.generate     # wavs + data/stimuli.jsonl
.venv/bin/python -m stimgen.check        # balance + every row has a real wav
.venv/bin/python -m raters.judge         # 6 personas -> data/ratings.jsonl
.venv/bin/python -m raters.judge --fill  # only the pairs still missing
.venv/bin/python -m raters.judge --verify
.venv/bin/python web/build.py            # web/stimuli.js for the rating page
```
