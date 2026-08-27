# harness status

Last verified run: 2026-08-26, Apple M4 Pro / macOS, Python 3.12 venv.
Every number below came from a run that actually happened on this machine.

## Built and verified working

- `harness/audio.py` — mono float32 @ 22050 Hz, frame-RMS voice-activity
  segmentation. Every latency measurement in the repo reduces to `segments()`.
- `harness/tts.py` — two backends. **Running backend on this machine: `say`**
  (macOS built-in). Piper is preferred and coded, but `pip install piper-tts`
  failed twice on network timeout, so the piper path is **UNVERIFIED — never
  executed**. Backend used is reported as `tts_backend` in every result dict.
- `harness/cues.py` — `none | filled_pause | breath | backchannel | verbal_stall`.
  Real audio, real sample offsets, no text tokens.
- `harness/exchange.py` — `synthesize_exchange(...)`. The contract the
  stimulus-generation side depends on; read its module docstring first.
- `scripts/run_agent.py` — `selfcheck`, `render`, `cues`.

## Measured tolerance (`python scripts/run_agent.py selfcheck`)

Grid of latency {200, 400, 800, 1200, 2000} ms x 5 cues, each rendered to disk
and re-measured from the written file:

    worst |gap error|        10.0 ms
    worst |cue onset error|   0.0 ms
    tolerance asserted       25.0 ms   -> PASS

The gap error is systematic and positive (+5 ms at short latencies, +10 ms at
long), not jitter. Two causes, both understood:
1. VAD frame quantization — 5 ms frames, segment ends round up and starts round
   down, so a gap is over-measured by up to 2 frames.
2. The relative (-35 dB) threshold is computed over the whole file, so a longer
   file shifts which frames near an utterance edge count as speech.
Halve it by dropping `audio.FRAME_MS` to 2.5 if the design ever needs it.

Honest caveat on what this verifies: the re-measurement confirms that what was
written to disk contains speech ending and starting where intended, and that
assembly and file I/O are sample-exact. It does **not** independently validate
the VAD threshold itself, since trimming and measurement share one function.

Calibration found the hard way: the synthetic breath originally used a
symmetric `sin^1.6` envelope, whose slow attack put the measured cue onset
25.0 ms late — exactly the whole error budget. The attack is now ~6% of the
cue duration and the error is 0.0 ms.

## Design constraints the stimulus side must respect

A cue lives inside the gap and never extends it. At the default 150 ms cue
onset, cues do not fit below these latencies, and `synthesize_exchange` raises
`ValueError` rather than silently rendering something else:

    filled_pause  305 ms   backchannel   510 ms
    breath        320 ms   verbal_stall  585 ms

So the 200 ms and 400 ms latency cells are `cue="none"` only. Call
`cue_duration_ms(cue)` to plan cells.

## NOT built

- **The live streaming voice loop (mic -> ASR -> LLM -> TTS) does not exist.**
  faster-whisper and mlx-lm are installed and pinned in `pyproject.toml`, but
  the Llama-3.2-1B-Instruct-4bit download did not finish inside the session and
  no live loop was written or run. No TTFT, time-to-first-audio, or wall-clock
  jitter number has been measured. Do not cite one.
- Consequence: the repo currently measures the *stimulus*, not a live agent.
  For a listening study that is the part that matters, but the "can a real
  robot hit these targets" question is untouched.

## Next person

1. Finish the model pull, then build the live loop and report its real jitter
   distribution — expect it to be far worse than 10 ms, and say so.
2. Retry `UV_HTTP_TIMEOUT=180 uv pip install piper-tts`, download a voice to
   `models/piper/`, and re-run `selfcheck` to confirm the piper path.
3. Cues are TTS/synthetic, not recorded humans. A TTS "uh" has a clean onset
   and no coarticulation with the surrounding speech. UNVERIFIED whether
   listeners hear it as genuine disfluency. `cues.register_cue_wav()` swaps in
   recordings without touching anything else.
