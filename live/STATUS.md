# live loop status

Last verified run: 2026-08-26, Apple M4 Pro / 24 GB / macOS, Python 3.12 venv.
Every number below came from a run that actually happened on this machine, and
the run that produced it is in `live/results/*.json`, one record per turn.

`harness/STATUS.md` said the live loop did not exist and that no TTFT or
time-to-first-audio number should be cited. It exists now. These are those
numbers.

## What runs

    scripts/run_live.py selfcheck            # real turns, asserts every stage timer
    scripts/run_live.py batch --n 20 --out live/results/run.json
    scripts/run_live.py batch --n 20 --tts say
    scripts/run_live.py batch --n 3 --mic
    scripts/run_live.py devices

    audio in -> ASR -> LM -> TTS -> audio out

| stage | what | streaming? |
|---|---|---|
| in | 20 ms chunks into one queue. `--wav` paces a rendered utterance at 1x; `--mic` reads the default input device | yes |
| ASR partials | faster-whisper `tiny.en`, int8 CPU, worker thread, whole-buffer re-decode every 500 ms | chunked, not streaming |
| ASR final | faster-whisper `base.en`, int8 CPU, one decode of the complete utterance at endpoint | no |
| LM | mlx-lm, `mlx-community/Llama-3.2-1B-Instruct-4bit`, 4-bit, MPS, token-streamed, stops at the first sentence | yes |
| TTS | `harness/tts.py` unmodified, piper backend, `en_US-lessac-medium` | no, whole utterance |
| out | one `sounddevice` output stream held open for the session, 128-frame blocks (5.8 ms) at 22050 Hz | yes |

The model pull that the previous agent could not finish did finish:
`Llama-3.2-1B-Instruct-4bit` loads and generates. No substitution was needed.

## The gap, defined the same way as the stimulus side

`harness/exchange.py` defines latency as **offset of prompt speech to onset of
response speech, both silence-trimmed**. The live loop measures the same thing:

- User speech offset is re-measured from the captured input with
  `harness.audio.segments(x, merge_gap_ms=30, min_len_ms=20)` -- the same
  function with the same parameters `exchange.measure_exchange` uses -- and is
  the end of the last speech segment. **Not** the moment the endpointer
  noticed. The endpointer's silence hangover therefore sits inside the gap,
  which is where a listener hears it.
- Agent speech onset is the first sample handed to the output callback.
  `tts.Voice.synth` trims leading silence, so sample 0 is speech onset. Time to
  first audio and the gap are the same quantity by construction, and are
  reported as `ttfa_ms` and `gap_ms`.

So `gap_ms` here and `actual_gap_ms` in a rendered stimulus are directly
comparable numbers.

## Measured: 40 turns, piper TTS

Two independent runs of 20 turns, five prompts x four repeats each
(`live-piper-blackhole-n20.json`, `live-piper-blackhole-n20-rep2.json`).
Pooled, **n = 40**:

    user speech offset -> agent speech onset

    median   1452 ms
    IQR      1283 - 1789 ms
    p90      2246 ms
    min      1067 ms      max  2718 ms
    mean     1568 ms      sd    381 ms

The two runs agree closely: medians 1456.4 ms (n=20) and 1452.0 ms (n=20).
Run-to-run reproducibility is good; within-run spread is not small.

Per stage, from `live-piper-blackhole-n20.json`, n = 20, median [IQR]:

| stage | ms | note |
|---|---|---|
| endpoint hangover | 363 [355-367] | design constant `HANGOVER_MS = 350`; a floor, not a measurement |
| wait for in-flight partial | 4 [1-138] | the final decode waits for the partial worker to release the cores |
| ASR final decode | 434 [375-593] | `base.en` int8 CPU, ~2.9 s of audio |
| **LM time to first token** | **346 [305-432]** | 4-bit 1B on MPS |
| LM to end of first sentence | 124 [108-145] | |
| TTS synthesis | 119 [89-181] | piper, whole utterance |
| handoff to the audio callback | 3.5 [1.9-4.0] | |

Time to first ASR partial, measured from the start of the input stream (the
prompt audio has 300 ms of lead silence before speech): median 672 ms
[662-732], n = 20. It lands while the user is still talking, as it should.

Startup, excluded from every turn above: model load 5.8-6.8 s, then one warmup
ASR decode + LM generation + TTS synthesis, 1.3-2.2 s. Turn 0 is warm.

## The finding

The study's stimulus grid is 200, 400, 800, 1200, 2000 ms.

**This stack on this laptop never once entered the bottom three cells.** The
fastest of 40 turns was 1067 ms. 0/40 turns came in under 800 ms; 4/40 under
1200 ms; 35/40 under 2000 ms. The median, 1452 ms, sits between the 1200 and
2000 ms cells.

Nothing in the pipeline is obviously wasteful, and no single stage dominates:
350 ms of it is the endpointer's hangover, ~430 ms is the final ASR decode,
~470 ms is the LM, ~120 ms is TTS. To reach the 400 ms cell every one of those
would have to be replaced, not tuned. It is slower than we hoped, and that is
the result.

The jitter matters as much as the median. sd is 381 ms on a 1452 ms median, and
the same prompt spoken twice can differ by a second. A stimulus set holds gap
error to 10 ms (`harness/STATUS.md`); the live agent it is meant to stand in for
does not hold anything to better than a few hundred ms.

## The fallback TTS backend costs 2.6 seconds

`harness/STATUS.md` recorded the piper path as UNVERIFIED and macOS `say` as
what was actually running. Both were run here, same loop, same prompts, n = 20
each:

| TTS backend | TTS stage, median [IQR] | gap, median [IQR] | n |
|---|---|---|---|
| piper `en_US-lessac-medium` | 119 ms [89-181] | 1456 ms [1304-1868] | 20 |
| macOS `say` | 2617 ms [2136-2991] | 4252 ms [3637-5342] | 20 |

`say` costs roughly two seconds regardless of what it is asked to speak --
0.30 s of audio and 1.77 s of audio both take about 2 s -- because each
utterance is a cold subprocess. It is fine for rendering stimuli offline. It
cannot be in a live loop.

**The piper path in `harness/tts.py` is now verified: it loads, synthesizes,
and was used for 45 of the turns reported here.** The voice lives in
`models/piper-live/`, deliberately *not* `models/piper/`, so
`tts.Voice._autodetect` still resolves to `say` everywhere else in the repo and
no offline stimulus changes voice behind anyone's back. Nothing under
`harness/` was modified.

## Output device latency is not part of the loop, and it is bigger than the loop

`gap_ms` ends when the first sample reaches the audio callback. The ear waits
the device's own latency on top. `acoustic_gap_ms` adds it, per run, from what
PortAudio reports at stream open:

| device | reported output latency |
|---|---|
| BlackHole 2ch (virtual) | 6-29 ms |
| MacBook Pro Speakers | 42-88 ms |
| AirPods Pro (Bluetooth) | 354-726 ms |

That figure is PortAudio's estimate and it moves between stream opens, so the
value recorded in each result JSON is the one that applies to that run. It was
not measured with a loopback, and it is not claimed to be exact. The direction
is not in doubt: **on Bluetooth, the headphones add more delay than the LM and
the TTS combined.** The 20-turn runs were routed to BlackHole so they would not
play aloud on a shared machine; a 5-turn run on the built-in speakers
(`live-piper-speakers-n5.json`) confirms the loop stages are unchanged by the
device, median gap 1810 ms, n = 5.

## Caveats, in order of how much they should bother you

1. **The machine was busy.** Load average was 8-19 throughout (12 cores, other
   jobs running). ASR is CPU-bound and felt it. Each result JSON records
   `loadavg_start` and `loadavg_end`. An idle machine would be faster; how much
   faster was not measured, so no number is offered.
2. **The user is a TTS voice, not a person.** The 40 reported turns were driven
   by piper-rendered prompts fed through the queue at 1x real time, not by a
   human at a microphone. That makes the input reproducible and gives an exact
   speech-offset landmark, at the cost of realistic prosody, disfluency, and
   room noise -- all of which make endpointing harder in reality. Treat the
   endpointing hangover as a best case.
3. **The microphone path is only partly verified.** `--mic` opens a real input
   stream, delivers samples, endpoints, and produces a transcript; that was run
   and observed. The full acoustic loop -- speakers into the room into the mic
   -- was **not** verified, because this machine's audio output is routed to
   Bluetooth headphones and routing around that is not mine to change. Mic
   stream `t0` is also approximate: it is stamped at the first callback minus
   one block and ignores ADC latency, so mic-mode gaps are a few tens of ms
   optimistic. No reported number comes from mic mode.
4. **`asr_final_dispatch_ms` is an artifact of this design**, not a law. The
   final decode waits for the in-flight partial to finish rather than running
   two decoders over the same cores. Median 4 ms, but the IQR reaches 138 ms
   and one turn paid 249 ms.
5. **The final decode throws away the partial's work.** A true streaming ASR
   would not. That waste is inside every gap reported here.
6. **The endpointer is not the harness VAD.** Gap *measurement* uses
   `harness.audio.segments`, untouched. Live *endpointing* is a separate,
   dumber rule: chunk RMS against a running peak, over an absolute floor of
   `LIVE_FLOOR_DBFS = -45`. The harness's -55 dBFS floor is right for a rendered
   file but arms the hangover on room noise in a live stream. That constant is
   the calibration knob for a quieter or noisier room; it does not touch any
   measurement.
7. **One sentence per turn.** Generation stops at the first sentence boundary
   and that sentence is spoken. A longer reply would not change the gap -- only
   the first sentence is inside it -- but it would change the LM stage.

## Self-check

    scripts/run_live.py selfcheck

Runs real turns and asserts, for each: every stage timer is present, is a
number, and is not negative; the gap and TTFA are positive; the transcript and
reply are non-empty; and the stages sum to the gap within one output block
(5.8 ms + 1 ms). A missing or negative timer would make everything above
fiction, so it fails loudly rather than warning. Last run: PASS.

## Next person

1. Re-run on an idle machine. Everything above was measured under load average
   8-19 and the ASR stage is CPU-bound.
2. The endpointer's 350 ms hangover is a quarter of the median gap and is a
   constant, not a measurement. A trained endpointer, or a shorter hangover
   with a false-cutoff rate to go with it, is the single cheapest win.
3. Streaming TTS would move the ~120 ms piper stage inside the gap to near
   zero, but it is already the smallest stage. Streaming ASR that does not
   discard the partial's work is worth more.
4. If the study ever needs to claim a real agent can hit a given cell, that
   claim needs a run in `live/results/`, not an argument.
