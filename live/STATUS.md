# live loop status

Last verified run: 2026-08-26, Apple M4 Pro / 24 GB / macOS, Python 3.12 venv.
Every number below came from a run that actually happened on this machine, and
the run that produced it is in `live/results/*.json`, one record per turn.

`harness/STATUS.md` said the live loop did not exist and that no TTFT or
time-to-first-audio number should be cited. It exists now. These are those
numbers.

## Headline

| | gap, median [IQR] | n | false endpoints | mean WER vs prompt |
|---|---|---|---|---|
| baseline, serial (`22:27` runs, kept) | 1452 ms [1283-1789] | 40 | not measured | not measured |
| baseline, serial (re-run, same code) | **807 ms** [779-895] | 100 | 0/100 | 0.022 |
| `--fast` | **506 ms** [487-529] | 40 | 0/40 | 0.000 |
| `--fast --final-model tiny.en` | **386 ms** [372-435] | 40 | 0/40 | 0.033 |

Two separate things happened and they must not be added together:

1. **The 1452 ms baseline does not reproduce.** The same code, unchanged,
   measures 807 ms twenty minutes later. See "The baseline moved on its own".
2. **The loop is 2.1x faster than its own re-measured baseline**, from running
   the whole downstream inside the endpointer's hangover instead of after it.

The study's stimulus grid is 200, 400, 800, 1200, 2000 ms.

| turns landing under | 22:27 baseline (n=40) | re-run baseline (n=100) | `--fast --final-model tiny.en` (n=40) |
|---|---|---|---|
| 400 ms | 0 | 0 | **26** |
| 800 ms | 0 | 43 | 40 |
| 1200 ms | 4 | 99 | 40 |

The 400 ms cell was unreachable. It is now reached on 26 of 40 turns, with a
median of 386 ms and no false endpoints.

## Setup

Two things beyond `pyproject.toml`'s base deps, both already declared there as
extras:

    uv pip install sounddevice piper-tts
    mkdir -p models/piper-live && cd models/piper-live && curl -LO \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
      -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

Without the voice the loop falls back to macOS `say` and every gap grows by
about 2.6 s (see below). Without `sounddevice` there is no output stage at all.
`models/` is gitignored; no weights are in the repo.

## What runs

    scripts/run_live.py selfcheck                       # real turns, asserts every stage timer
    scripts/run_live.py batch --n 20                    # the baseline path, unchanged
    scripts/run_live.py batch --n 20 --fast             # downstream runs inside the hangover
    scripts/run_live.py batch --n 20 --fast --final-model tiny.en
    scripts/run_live.py batch --n 20 --fast --hangover 250 --arm 60
    scripts/run_live.py batch --n 20 --tts say
    scripts/run_live.py batch --n 3 --mic
    scripts/run_live.py devices

**Without `--fast` the loop is the original serial path, byte for byte**, so the
baseline stays runnable and no optimisation is ever reported without a before
measured on the same machine in the same hour.

    audio in -> ASR -> LM -> TTS -> audio out

| stage | what | streaming? |
|---|---|---|
| in | 20 ms chunks into one queue. `--wav` paces a rendered utterance at 1x; `--mic` reads the default input device | yes |
| ASR partials | faster-whisper `tiny.en`, int8 CPU, worker thread, whole-buffer re-decode every 500 ms | chunked, not streaming |
| ASR final | faster-whisper `base.en` (`--final-model` to change), int8 CPU, one decode of the complete utterance | no |
| LM | mlx-lm, `mlx-community/Llama-3.2-1B-Instruct-4bit`, 4-bit, MPS, token-streamed, stops at the first sentence | yes |
| TTS | `harness/tts.py` unmodified, piper backend, `en_US-lessac-medium` | no, whole utterance |
| out | one `sounddevice` output stream held open for the session, 128-frame blocks (5.8 ms) at 22050 Hz | yes |

## The gap, defined the same way as the stimulus side

Unchanged. `harness/exchange.py` defines latency as **offset of prompt speech to
onset of response speech, both silence-trimmed**, and the live loop measures the
same thing with the same function and the same parameters:

- User speech offset is re-measured from the captured input with
  `harness.audio.segments(x, merge_gap_ms=30, min_len_ms=20)` and is the end of
  the last speech segment. **Not** the moment the endpointer noticed. The
  endpointer's silence hangover therefore sits inside the gap, which is where a
  listener hears it.
- Agent speech onset is the first sample handed to the output callback.
  `tts.Voice.synth` trims leading silence, so sample 0 is speech onset.

Nothing in this definition was touched, so every number here is directly
comparable to the 1452 ms baseline and to `actual_gap_ms` in a rendered
stimulus. Nothing under `harness/` was modified.

## What was changed: run the downstream inside the hangover

The old conclusion was "no single stage dominates, so getting to 400 ms means
replacing all of them". That was true of the *serial* budget and false as a
statement about the loop, because the stages do not have to be serial.

The endpointer waits 350 ms of silence before calling the turn over. During those
350 ms the old loop did nothing at all. `--fast` starts the final decode, the LM
and the TTS after `--arm` ms of silence (default 150; 80 used for every number
reported here), on the audio captured so far.

**Nothing is guessed.** The only thing being bet on is that the talker has
stopped. The audio between the snapshot and the endpoint is silence by
definition, so a claimed result is the same transcript the baseline would have
produced. If the talker resumes, the snapshot is stale, the work is discarded and
the turn falls back to the serial path. Across 240 turns in `--fast` mode, 236 were served
speculatively and 0 produced a false endpoint.

Per stage, medians. `stage_ms` charges each stage only what it added to the
critical path, so a stage that ran inside the hangover reads 0; `work_ms` (also
in every result JSON) records what it cost on its own clock.

| stage | baseline n=100 | `--fast` n=40 | `--fast` + `tiny.en` n=40 |
|---|---|---|---|
| endpoint hangover | 359.8 | 364.6 | 362.6 |
| wait for in-flight partial | 4.6 | **0.0** | **0.0** |
| ASR final decode | 237.0 | **0.0** | **0.0** |
| LM time to first token | 100.7 | 57.6 | **0.0** |
| LM to end of first sentence | 32.3 | 31.6 | **0.0** |
| TTS synthesis | 42.5 | 41.9 | 17.1 |
| handoff to the audio callback | 2.8 | 2.7 | 2.6 |
| **gap** | **807.4** | **506.3** | **386.2** |

With `base.en` the decode (237 ms of work) fits entirely inside the hangover and
pushes the LM half out. With `tiny.en` (111-131 ms of work) the decode, the LM
and all but a sliver of the TTS fit, and **the gap becomes the hangover plus
17 ms**.

The second ASR model is the only accuracy trade in the whole set. Pooled over
every run on this machine: `base.en` mean WER 0.0238 (n=140, 128/140 exact),
`tiny.en` 0.0333 (n=80, 68/80 exact). Both fail on the same prompt, "Is there
parking near the entrance?", by one word. **Those two numbers are not separable
at this n on a five-prompt set** -- individual `base.en` runs scored 0.0, 0.0, 0.0
and 0.0667. The honest statement is that `tiny.en` halves the decode for a WER
cost this measurement cannot resolve, not that it is free.

## What did not work

Four of the five ideas on the list. They are here because the measurement that
kills them is the result.

### Cutting the endpointer hangover buys nothing

It was a quarter of the old gap and the obvious first win. Once the downstream
runs inside it, it is no longer on the critical path, and shortening it only
uncovers the ASR decode again. Measured twice, at two ASR models:

| hangover | `--fast`, `base.en`, n=20 | hangover charged | ASR spill |
|---|---|---|---|
| 350 ms | 498.2 ms | 356.8 | 0.0 |
| 250 ms | 515.4 ms | 267.2 | 73.8 |
| 150 ms | 510.7 ms | 161.8 | 171.8 |
| 120 ms | 511.7 ms | 126.2 | 193.8 |

Cutting 230 ms off the hangover changed the gap by **+13 ms**. The same test with
`tiny.en`: 350 ms gives 386.2 ms (n=40), 250 ms gives 394.4 ms (n=40).

False endpoints were 0/20 at every setting, including 120 ms, because this prompt
set's longest internal pause is 65 ms. That is a property of a piper-rendered
talker, not of a person, and it is why the false-endpoint column is reported next
to every gap rather than trusted once: a human's pauses are longer and a 120 ms
hangover would cut them off. Mean WER did drift up at 120 ms (0.0667 on that run)
but within the run-to-run spread above, so it is not evidence of truncation on
its own.

**Keep the hangover at 350 ms.** It costs nothing and it is the only thing
protecting a real talker from being interrupted.

### Trusting the partial instead of re-decoding

The partial's work is discarded by the final decode, and that waste was inside
every reported gap. It still is, and it no longer matters, because the decode it
would save is now free.

What it would cost is measured: `wer_partial_last` in every result JSON is the
WER of the newest partial against the known prompt. On the runs where the partial
worker ran to the end of the utterance it was 0.033 to 0.067, against 0.000 to
0.033 for the re-decode. So trusting the partial buys 0 ms and costs accuracy.
**Starting the final decode early is the same idea with none of the downside, and
it is what `--fast` does.**

(In the `--arm 40` runs `wer_partial_last` reads 0.58-0.60. That is an artifact:
arming stops the partial worker, and at 40 ms it armed mid-utterance, so the
"newest partial" recorded there is a truncated one. Do not read those cells as a
property of partials.)

### Speculative LM start on the partial transcript

Not built, and the reason is the row above. Speculating on the partial means
handling a mismatch, and its whole payoff is the LM's 100 ms time-to-first-token,
which the `tiny.en` config already charges 0 for. There is nothing left to win
and a restart path to get wrong.

### Chunked TTS: speaking the first clause before the sentence finishes

Not built. This was the highest-ranked idea against the old budget, where "LM to
sentence end" (124 ms) plus TTS (119 ms) was 243 ms of 1452. Against the measured
budget it is worth nothing:

- TTS work is 42 ms median and LM-to-sentence-end 32 ms. A perfect streaming TTS
  can hide at most 74 ms of a 807 ms baseline.
- In the `--fast --final-model tiny.en` config both already charge 0 and 17 ms.
- Of the 7 distinct replies this prompt set produces, 3 have an internal clause
  boundary at all, and those boundaries are 1-2 words in ("Sorry, ...", "In many
  places, ..."). Splitting there yields a first chunk shorter than the seam it
  introduces.

Chunked TTS would trade an audible seam for single-digit milliseconds here. It is
the right optimisation for a stack whose replies are long, and this stack speaks
one short sentence per turn.

### Arming the speculation earlier than 80 ms

`--arm` is the silence the loop waits through before betting the turn is over,
and it is dead time at the head of the pipeline, so lower looks better. It is not:

| arm | `--fast`, `tiny.en`, n=20 | sd | max | served speculatively | pipelines launched |
|---|---|---|---|---|---|
| 80 ms | 379.5 ms | 57.0 | 531.6 | 20/20 | 20 |
| 40 ms | 376.0 ms | 170.2 | 807.5 | **16/20** | **35** |

At 40 ms the arm fires inside the talker's own pauses. The snapshot goes stale,
the decode is thrown away, and 4 turns in 20 fell back to the serial path having
paid for both. The median moved 3.5 ms and the spread tripled. 80 ms is above
this prompt set's longest internal pause (65 ms) and below the 350 ms hangover,
and that is the whole design rule.

## The baseline moved on its own, and that is the biggest caveat here

The first thing done in this pass was to re-run the unchanged code. It came back
at 883 ms, then 803, 829, 796, 797 -- pooled median **807.4 ms, n=100** -- against
the 1452 ms recorded twenty minutes earlier the same evening.

The stage that moved most is the LM: 346 ms time-to-first-token then, 101 ms now.
Model load went from 4.3-6.8 s to 2.4-3.4 s and warmup from 1.3-3.8 s to
0.4-1.1 s. Load average does not explain it: the 22:30 run recorded 16.79 and
returned 1452 ms; the 22:47 run recorded 16.97 and returned 883 ms.

**Whatever the machine was doing at 22:27 is not captured by any variable this
harness records.** Two consequences, and the second is the one that matters:

1. The 1452 ms figure is not wrong -- it happened, the file is still in
   `live/results/` -- but it is not a property of this stack. Any cross-run
   comparison to it is invalid.
2. **Every optimisation above is reported against a baseline re-measured in the
   same hour**, on the same device, interleaved with the optimised runs. That is
   why there are five baseline runs and why the baseline row has n=100.

## The fallback TTS backend costs 2.6 seconds

Unchanged from the previous pass, and still true. `harness/STATUS.md` recorded the
piper path as UNVERIFIED and macOS `say` as what was actually running. Both were
run, same loop, same prompts, n=20 each:

| TTS backend | TTS stage, median [IQR] | gap, median [IQR] | n |
|---|---|---|---|
| piper `en_US-lessac-medium` | 119 ms [89-181] | 1456 ms [1304-1868] | 20 |
| macOS `say` | 2617 ms [2136-2991] | 4252 ms [3637-5342] | 20 |

`say` costs roughly two seconds regardless of what it is asked to speak, because
each utterance is a cold subprocess. It is fine for rendering stimuli offline. It
cannot be in a live loop. (Both rows are from the 22:27-22:36 machine state, so
compare them to each other and not to anything above.)

The piper path in `harness/tts.py` is verified: it loads, synthesizes, and was
used for every turn reported here. The voice lives in `models/piper-live/`,
deliberately *not* `models/piper/`, so `tts.Voice._autodetect` still resolves to
`say` everywhere else in the repo and no offline stimulus changes voice behind
anyone's back.

## Output device latency is not part of the loop, and it is now bigger than the loop

`gap_ms` ends when the first sample reaches the audio callback. The ear waits the
device's own latency on top. `acoustic_gap_ms` adds it, per run, from what
PortAudio reports at stream open:

| device | reported output latency |
|---|---|
| BlackHole 2ch (virtual) | 6-29 ms |
| MacBook Pro Speakers | 42-88 ms |
| AirPods Pro (Bluetooth) | 354-726 ms |

That figure is PortAudio's estimate and it moves between stream opens, so the
value recorded in each result JSON is the one that applies to that run. It was
not measured with a loopback and is not claimed to be exact. The direction is not
in doubt, and the optimisation has made it worse: **the Bluetooth headphones now
add roughly as much delay as the entire optimised loop.** A 386 ms gap on AirPods
is heard at about 740 ms. Every run above was routed to BlackHole.

If this stack ever has to hit the 400 ms cell for a listener rather than for a
log, the transducer is the next thing to fix, not the software.

## Caveats, in order of how much they should bother you

1. **The baseline is not stable across the evening.** See above. This is now the
   first caveat rather than the fifth, and it is why every comparison here is
   run-adjacent.
2. **The user is a TTS voice, not a person.** All 340 turns in this pass were driven by
   piper-rendered prompts fed through the queue at 1x real time. That makes the
   input reproducible and gives an exact speech-offset landmark, at the cost of
   realistic prosody, disfluency and room noise, all of which make endpointing
   harder. This bears directly on `--arm`: 80 ms clears this talker's longest
   internal pause (65 ms) and would not clear a person's. Treat both the hangover
   and the arm as best cases, and re-measure the false-endpoint column before
   trusting either on a human.
3. **`--fast` bets that a quiet talker has finished.** The bet is checked, not
   assumed: the result is discarded unless no speech arrived after the snapshot.
   When it loses, the turn is *slower* than the baseline, because the wasted
   decode has to finish before the real one starts. At `--arm 80` it lost 0 times
   in 180 turns; at `--arm 40` it lost 4 times in 20.
4. **The microphone path is only partly verified.** `--mic` opens a real input
   stream, delivers samples, endpoints and produces a transcript. The full
   acoustic loop, speakers into the room into the mic, was **not** verified. Mic
   stream `t0` is also approximate: it is stamped at the first callback minus one
   block and ignores ADC latency, so mic-mode gaps are a few tens of ms
   optimistic. No reported number comes from mic mode.
5. **The endpointer is not the harness VAD.** Gap *measurement* uses
   `harness.audio.segments`, untouched. Live *endpointing* is a separate, dumber
   rule: chunk RMS against a running peak, over an absolute floor of
   `LIVE_FLOOR_DBFS = -45`. That constant is the calibration knob for a quieter or
   noisier room; it does not touch any measurement.
6. **One sentence per turn.** Generation stops at the first sentence boundary and
   that sentence is spoken. This is what makes chunked TTS pointless here; a stack
   with longer replies would find it worth building.
7. **False endpoints are only detectable on the `--wav` path**, where the prompt's
   true speech offset is known in advance. On `--mic` the column is absent, not
   zero.

## Self-check

    scripts/run_live.py selfcheck
    scripts/run_live.py selfcheck --fast --final-model tiny.en

Runs real turns and asserts, for each: every stage timer is present, is a number,
and is not negative; the gap and TTFA are positive; the transcript and reply are
non-empty; and the stages sum to the gap within one output block.

That last assertion still holds under overlap, and it is the reason the stage
table is built the way it is. `_critical_path` walks the landmarks in order and
charges each stage only the time it added beyond everything already elapsed, so
the stages remain a partition of the gap whether the work was serial or not. On a
serial run it reduces to the plain difference and reproduces the old numbers
exactly. Last run: PASS in both modes.

## Next person

1. **The gap is now the hangover.** 362 of the 386 ms is one design constant
   protecting the talker from being cut off, and the evidence above says you
   cannot simply lower it: you would have to replace it with an endpointer that is
   *right* faster, not one that gives up sooner. A trained endpointer is the only
   remaining large win, and it is the one that needs a false-endpoint rate on
   human speech to be worth anything.
2. **Re-measure the baseline every time.** Do not compare to a number in this file
   that was not produced in the same session as the thing you are testing. That is
   what the 1452 ms row is doing here: standing as a warning, not a reference.
3. Everything after the hangover now sums to about 24 ms of critical path. There
   is no second stage left to optimise. Faster models buy the margin back only if
   the hangover comes down first.
4. If the study ever needs to claim a real agent can hit a given cell, that claim
   needs a run in `live/results/`, not an argument -- and on a real transducer,
   not into BlackHole.
