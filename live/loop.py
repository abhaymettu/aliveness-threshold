"""A real streaming voice loop on this laptop: audio in -> ASR -> LM -> TTS ->
audio out, with every stage on the wall clock.

The point of this module is measurement, not conversation quality. It exists so
the repo can say what a live agent on this hardware actually does, next to what
the offline stimulus harness renders.

The gap is defined exactly as ``harness/exchange.py`` defines it, so the two
numbers are directly comparable:

    gap = onset of agent speech - offset of user speech

Both landmarks are silence-trimmed. Concretely:

- **User speech offset** is re-measured from the captured input with
  ``harness.audio.segments(..., merge_gap_ms=30, min_len_ms=20)`` -- the same
  function and the same parameters ``exchange.measure_exchange`` uses -- and is
  the *end of the last speech segment*, not the moment the endpointer noticed.
  A real system's silence hangover is therefore inside the gap, where a
  listener hears it.
- **Agent speech onset** is the first sample handed to the output device.
  ``tts.Voice.synth`` trims leading silence, so sample 0 is speech onset.

Two clocks, and the difference matters:

- ``gap_ms`` ends when the first sample reaches the audio callback.
- ``acoustic_gap_ms`` adds the output device's own reported latency, which is
  what a listener's ear waits through. On Bluetooth it dominates everything
  this loop does. Both are reported; neither is estimated.

Input is a real-time stream. ``--wav`` paces a rendered utterance at 1x through
the same queue the microphone path uses, which makes the measurement
reproducible; ``--mic`` reads the microphone. Everything downstream is
identical.

Stages, and which are streaming:

- ASR: faster-whisper, **chunked, not streaming**. Two models, as a real
  streaming stack uses: ``tiny.en`` in a worker thread re-decodes the whole
  buffer every ``PARTIAL_EVERY_MS`` for partials, and ``base.en`` decodes the
  complete utterance once at endpoint for the final transcript. The final
  decode throws away the partial's work -- a true streaming ASR would not, and
  that waste is inside the reported gap.
- LM: mlx-lm, token-streamed. Generation stops at the first sentence boundary
  and that sentence is what gets spoken.
- TTS: ``harness.tts``, unmodified. Whole-utterance, **not streaming** --
  the full synthesis of the first sentence sits inside the gap. The piper
  backend is used when a voice is present in ``models/piper-live/``, otherwise
  macOS ``say``. That directory is deliberately *not* ``models/piper/``, so
  ``tts.Voice._autodetect`` still resolves to ``say`` for everything else in
  the repo and no offline stimulus changes voice behind anyone's back.
"""

from __future__ import annotations

import json
import os
import queue
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from harness import audio, tts

CHUNK_MS = 20.0  # input chunk granularity
HANGOVER_MS = 350.0  # trailing silence before the endpointer calls the turn over
PARTIAL_EVERY_MS = 500.0  # new audio needed before another partial decode
BLOCK = 128  # output callback block, 5.8ms at 22050 Hz
MAX_TOKENS = 48
ASR_MODEL = "base.en"        # final transcript
PARTIAL_MODEL = "tiny.en"    # partials, in the background thread
LM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SYSTEM = "You are a voice assistant. Reply in one short spoken sentence."

# same params exchange.measure_exchange uses, so both sides measure alike
SEG_KW = {"merge_gap_ms": 30.0, "min_len_ms": 20.0}

PROMPTS = [
    "What time do you close on Sunday?",
    "Is there parking near the entrance?",
    "How much does the annual pass cost?",
    "Can I bring a dog inside?",
    "Where do I pick up my order?",
]


def pick_voice(which: str = "auto"):
    """The loop's TTS voice. ``auto`` prefers piper, falls back to ``say``.

    Backend selection itself lives in ``harness.tts`` and is not reimplemented
    here; this only chooses which of its two backends to ask for.
    """
    found = sorted((Path(__file__).resolve().parent.parent / "models" / "piper-live").glob("*.onnx"))
    if which in ("auto", "piper") and found:
        return tts.Voice("piper", name=str(found[0]))
    if which == "piper":
        raise RuntimeError("piper requested but no .onnx voice in models/piper-live/")
    return tts.default_voice()


def _to_whisper(x: np.ndarray) -> np.ndarray:
    """22050 Hz -> the 16 kHz faster-whisper expects.

    ponytail: linear resample, same as audio.read's safety net. Whisper's own
    front end is a mel filterbank; a sharper anti-alias filter is not worth a
    dependency here.
    """
    n = int(round(len(x) * 16000 / audio.SR))
    return np.interp(
        np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float64)
    ).astype(np.float32)


class Asr:
    def __init__(self, model: str = ASR_MODEL):
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.name = model
        self.m = WhisperModel(model, device="cpu", compute_type="int8")

    def text(self, x: np.ndarray) -> str:
        # temperature=0 pins the decode to a single pass. The default is a
        # temperature ladder that re-decodes up to five times when the
        # logprob/compression thresholds fail, which a partial ending mid-word
        # does constantly -- measured at up to 6.7s for one partial before this
        # was pinned. condition_on_previous_text off for the same reason.
        segs, _ = self.m.transcribe(
            _to_whisper(x), language="en", beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()


class Lm:
    def __init__(self, model: str = LM_MODEL):
        from mlx_lm import load  # noqa: PLC0415

        self.name = model
        self.model, self.tok = load(model)

    def first_sentence(self, user_text: str):
        """Stream until the first sentence boundary.

        Returns ``(sentence, t_first_token, t_sentence, n_tokens)`` where the
        two times are ``perf_counter`` stamps.
        """
        from mlx_lm import stream_generate  # noqa: PLC0415

        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out, t_first, n = "", None, 0
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=MAX_TOKENS):
            if t_first is None:
                t_first = time.perf_counter()
            out += r.text
            n += 1
            if re.search(r"[.!?](\s|$)", out) and len(out.strip()) > 8:
                break
        m = re.search(r"^(.*?[.!?])(\s|$)", out.strip(), re.S)
        sentence = m.group(1) if m else out.strip()
        return sentence, t_first, time.perf_counter(), n


class Player:
    """One output stream held open for the whole session, as a real agent would.

    Opening a stream per turn would charge device setup to the agent's latency.
    ``t_first`` is stamped when the first sample is written into the callback
    buffer; the ear hears it ``latency_ms`` later.
    """

    def __init__(self, device=None):
        import sounddevice as sd  # noqa: PLC0415

        self.buf: np.ndarray | None = None
        self.pos = 0
        self.t_first: float | None = None
        self.stream = sd.OutputStream(
            samplerate=audio.SR, channels=1, dtype="float32", blocksize=BLOCK,
            device=device, latency="low", callback=self._cb,
        )
        self.stream.start()
        self.latency_ms = float(self.stream.latency) * 1000.0
        self.device = sd.query_devices(self.stream.device)["name"]

    def _cb(self, outdata, frames, _t, _status):
        outdata[:] = 0.0
        x = self.buf
        if x is None:
            return
        n = min(frames, len(x) - self.pos)
        if n <= 0:
            self.buf = None
            return
        if self.pos == 0:
            self.t_first = time.perf_counter()
        outdata[:n, 0] = x[self.pos : self.pos + n]
        self.pos += n

    def play(self, x: np.ndarray) -> None:
        self.pos, self.t_first, self.buf = 0, None, x

    def wait(self, timeout: float = 10.0) -> None:
        end = time.perf_counter() + timeout
        while self.buf is not None and time.perf_counter() < end:
            time.sleep(0.005)

    def close(self):
        self.stream.stop()
        self.stream.close()


def _pace_wav(x: np.ndarray, q: queue.Queue, t0: float) -> None:
    """Feed `x` into `q` at 1x real time. Sample k is delivered at t0 + k/SR."""
    n = audio.samples(CHUNK_MS)
    for i in range(0, len(x), n):
        d = (t0 + (i + n) / audio.SR) - time.perf_counter()
        if d > 0:
            time.sleep(d)
        q.put(x[i : i + n].copy())
    q.put(None)


def _mic(q: queue.Queue, stop: threading.Event, box: dict):
    """Microphone into the same queue. t0 is stamped at the first callback minus
    one block, so it is approximate by up to one block plus the ADC latency."""
    import sounddevice as sd  # noqa: PLC0415

    def cb(indata, frames, _t, _s):
        if "t0" not in box:
            box["t0"] = time.perf_counter() - frames / audio.SR
        q.put(indata[:, 0].copy())

    s = sd.InputStream(
        samplerate=audio.SR, channels=1, dtype="float32",
        blocksize=audio.samples(CHUNK_MS), callback=cb, latency="low",
    )
    s.start()
    stop.wait()
    s.stop()
    s.close()


def capture(source, asr: Asr, partial_asr: Asr) -> dict:
    """Run the input stage: stream in, partial-decode in the background, stop on
    the endpointer. Returns the captured audio and the wall-clock landmarks.

    `source` is ``("wav", ndarray)`` or ``("mic", None)``. `partial_asr` runs in
    the worker thread; the final decode waits for it to finish rather than
    running two decoders over the same cores at once, and that wait is timed as
    ``asr_final_dispatch_ms`` because a listener pays for it.
    """
    q: queue.Queue = queue.Queue()
    box: dict = {}
    stop_mic = threading.Event()
    if source[0] == "wav":
        t0 = time.perf_counter()
        box["t0"] = t0
        feeder = threading.Thread(target=_pace_wav, args=(source[1], q, t0), daemon=True)
    else:
        feeder = threading.Thread(target=_mic, args=(q, stop_mic, box), daemon=True)
    feeder.start()

    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    stop_partials = threading.Event()
    partial = {"t": None, "text": None}

    def partial_worker():
        seen = 0
        while not stop_partials.is_set():
            with lock:
                x = np.concatenate(chunks) if chunks else None
            if x is None or len(x) - seen < audio.samples(PARTIAL_EVERY_MS):
                time.sleep(0.005)
                continue
            seen = len(x)
            txt = partial_asr.text(x)
            if txt and partial["t"] is None:
                partial["t"], partial["text"] = time.perf_counter(), txt

    pw = threading.Thread(target=partial_worker, daemon=True)
    pw.start()

    # live endpointer: frame RMS against the loudest chunk so far, same -35dB
    # relative / -55dBFS absolute pair audio.speech_mask uses
    peak, last_speech, t_end, ended = 0.0, None, None, False
    while not ended:
        c = q.get()
        if c is None:
            t_end, ended = time.perf_counter(), True
            break
        with lock:
            chunks.append(c)
        now = time.perf_counter()
        r = float(np.sqrt((c.astype(np.float64) ** 2).mean()))
        peak = max(peak, r)
        if r >= peak * 10 ** (-35.0 / 20.0) and r >= 10 ** (-55.0 / 20.0):
            last_speech = now
        elif last_speech is not None and (now - last_speech) * 1000.0 >= HANGOVER_MS:
            t_end, ended = now, True

    stop_mic.set()
    stop_partials.set()
    pw.join(timeout=10.0)
    x = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    t_stream0 = box["t0"]
    t_final0 = time.perf_counter()
    final = asr.text(x)
    t_final = time.perf_counter()

    segs = audio.segments(x, **SEG_KW)
    if not segs:
        raise RuntimeError("no speech found in captured input; nothing to time against")
    return {
        "audio": x,
        "t_stream0": t_stream0,
        "t_speech_onset": t_stream0 + segs[0][0] / 1000.0,
        "t_speech_offset": t_stream0 + segs[-1][1] / 1000.0,
        "t_endpoint": t_end,
        "t_first_partial": partial["t"],
        "partial_text": partial["text"],
        "t_final0": t_final0,
        "t_final": t_final,
        "transcript": final,
        "input_ms": audio.millis(len(x)),
        "n_segments": len(segs),
    }


def run_turn(source, asr: Asr, partial_asr: Asr, lm: Lm, voice, player: Player,
             label: str = "") -> dict:
    cap = capture(source, asr, partial_asr)
    sentence, t_tok, t_sent, n_tok = lm.first_sentence(cap["transcript"] or "Hello?")
    t_tts0 = time.perf_counter()
    y = voice.synth(sentence)
    t_tts = time.perf_counter()
    if len(y) == 0:
        raise RuntimeError(f"TTS produced no audio for {sentence!r}")
    player.play(y)
    end = time.perf_counter() + 5.0
    while player.t_first is None and time.perf_counter() < end:
        time.sleep(0.001)
    if player.t_first is None:
        raise RuntimeError("output stream never consumed the reply audio")
    t_out = player.t_first
    player.wait()

    off = cap["t_speech_offset"]
    ms = lambda a, b: round((a - b) * 1000.0, 2)  # noqa: E731
    stage = {
        "asr_partial_first_ms": ms(cap["t_first_partial"], cap["t_stream0"])
        if cap["t_first_partial"] else None,
        "endpoint_hangover_ms": ms(cap["t_endpoint"], off),
        "asr_final_ms": ms(cap["t_final"], cap["t_final0"]),
        "asr_final_dispatch_ms": ms(cap["t_final0"], cap["t_endpoint"]),
        "lm_ttft_ms": ms(t_tok, cap["t_final"]),
        "lm_sentence_ms": ms(t_sent, t_tok),
        "tts_ms": ms(t_tts, t_tts0),
        "playback_dispatch_ms": ms(t_out, t_tts),
    }
    return {
        "label": label,
        "transcript": cap["transcript"],
        "partial_text": cap["partial_text"],
        "reply": sentence,
        "lm_tokens": n_tok,
        "input_ms": round(cap["input_ms"], 1),
        "user_speech_ms": ms(off, cap["t_speech_onset"]),
        "reply_audio_ms": round(audio.millis(len(y)), 1),
        # the number that is comparable to harness/exchange.py
        "gap_ms": ms(t_out, off),
        "ttfa_ms": ms(t_out, off),  # identical by construction: synth() trims
        "acoustic_gap_ms": round(ms(t_out, off) + player.latency_ms, 2),
        "stage_ms": stage,
        "n_input_segments": cap["n_segments"],
    }


def _stats(v: list[float]) -> dict:
    v = sorted(v)
    q = statistics.quantiles(v, n=4) if len(v) > 3 else [v[0], statistics.median(v), v[-1]]
    return {
        "n": len(v), "median": round(statistics.median(v), 1),
        "p25": round(q[0], 1), "p75": round(q[2], 1),
        "iqr": round(q[2] - q[0], 1), "min": round(v[0], 1), "max": round(v[-1], 1),
        "mean": round(statistics.fmean(v), 1),
        "sd": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0,
    }


def render_prompts(voice, lead_ms=300.0, tail_ms=900.0) -> list[tuple[str, np.ndarray]]:
    """Prompt utterances as real audio, with the trailing silence a real speaker
    leaves. Without that tail there is nothing for the endpointer to detect."""
    out = []
    for text in PROMPTS:
        p = voice.synth(text)
        x = np.concatenate(
            [np.zeros(audio.samples(lead_ms), np.float32), p,
             np.zeros(audio.samples(tail_ms), np.float32)]
        )
        out.append((text, x))
    return out


def run(n_turns: int = 20, out_path=None, device=None, mic: bool = False,
        tts_backend: str = "auto") -> dict:
    load0 = os.getloadavg()
    voice = pick_voice(tts_backend)
    # prompts always get the same voice regardless of what the agent speaks
    # with, so ASR-side timings stay comparable across TTS backends
    prompt_voice = pick_voice("auto")
    t0 = time.perf_counter()
    asr, partial_asr, lm = Asr(), Asr(PARTIAL_MODEL), Lm()
    load_ms = (time.perf_counter() - t0) * 1000.0
    player = Player(device)
    prompts = render_prompts(prompt_voice)
    t0 = time.perf_counter()
    asr.text(prompts[0][1])
    partial_asr.text(prompts[0][1])
    lm.first_sentence("Hello.")
    voice.synth("Ready.")
    warm_ms = (time.perf_counter() - t0) * 1000.0
    turns = []
    try:
        for i in range(n_turns):
            text, x = prompts[i % len(prompts)]
            src = ("mic", None) if mic else ("wav", x)
            t = run_turn(src, asr, partial_asr, lm, voice, player, label=text)
            t["turn"] = i
            turns.append(t)
            print(f"  turn {i:2d}  gap {t['gap_ms']:7.1f}ms  "
                  f"ttft {t['stage_ms']['lm_ttft_ms']:6.1f}  "
                  f"tts {t['stage_ms']['tts_ms']:6.1f}  {t['reply'][:44]!r}", flush=True)
    finally:
        player.close()

    keys = ["gap_ms", "acoustic_gap_ms"] + [f"stage_ms.{k}" for k in turns[0]["stage_ms"]]
    summary = {}
    for k in keys:
        vals = [
            (t["stage_ms"][k.split(".", 1)[1]] if k.startswith("stage_ms.") else t[k])
            for t in turns
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[k] = _stats(vals)

    res = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_turns": len(turns),
        "input": "microphone (live)" if mic else "TTS-rendered prompt, paced at 1x",
        "asr": {"final_model": asr.name, "partial_model": partial_asr.name,
                "mode": "chunked, whole-buffer re-decode",
                "partial_every_ms": PARTIAL_EVERY_MS},
        "lm": {"model": lm.name, "max_tokens": MAX_TOKENS, "stop": "first sentence"},
        "tts": {"backend": voice.backend, "voice": voice.name, "mode": "whole utterance"},
        "prompt_tts": {"backend": prompt_voice.backend, "voice": prompt_voice.name},
        "output_device": player.device,
        "output_device_latency_ms": round(player.latency_ms, 2),
        "model_load_ms": round(load_ms, 1),
        # this laptop is shared with other jobs; a CPU-bound ASR stage is only
        # as fast as the cores it can get, so the load is part of the result
        "loadavg_start": [round(v, 2) for v in load0],
        "warmup_ms": round(warm_ms, 1),
        "warmup": "one ASR decode, one LM generation and one TTS synthesis are run "
                  "before turn 0, so no measured turn pays a cold graph",
        "hangover_ms": HANGOVER_MS,
        "gap_definition": "agent speech onset - user speech offset, both silence-trimmed, "
                          "measured with harness.audio.segments (merge_gap_ms=30, min_len_ms=20) "
                          "-- the definition in harness/exchange.py",
        "loadavg_end": [round(v, 2) for v in os.getloadavg()],
        "summary_ms": summary,
        "turns": turns,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {out_path}")
    return res


def demo(n_turns: int = 2, **kw) -> dict:
    """Self-check. Runs real turns and asserts every stage timer exists, is a
    number, and is not negative. A missing or negative timer means the clock
    plumbing is wrong, which would make every number in live/STATUS.md fiction,
    so this fails loudly rather than warning."""
    res = run(n_turns=n_turns, **kw)
    assert res["turns"], "no turns ran"
    for t in res["turns"]:
        for k in ("gap_ms", "ttfa_ms", "acoustic_gap_ms"):
            v = t[k]
            assert isinstance(v, (int, float)), f"turn {t['turn']}: {k} missing"
            assert v > 0, f"turn {t['turn']}: {k} = {v}, must be positive"
        for k, v in t["stage_ms"].items():
            assert v is not None, f"turn {t['turn']}: stage {k} never timed"
            assert v >= 0, f"turn {t['turn']}: stage {k} = {v}ms, negative"
        assert t["transcript"], f"turn {t['turn']}: empty transcript"
        assert t["reply"], f"turn {t['turn']}: empty reply"
        # the parts must add up to the whole, within one output block
        parts = sum(
            t["stage_ms"][k] for k in
            ("endpoint_hangover_ms", "asr_final_dispatch_ms", "asr_final_ms",
             "lm_ttft_ms", "lm_sentence_ms", "tts_ms", "playback_dispatch_ms")
        )
        assert abs(parts - t["gap_ms"]) <= audio.millis(BLOCK) + 1.0, (
            f"turn {t['turn']}: stages sum to {parts:.1f}ms but gap is {t['gap_ms']:.1f}ms"
        )
    print(f"\nselfcheck PASS -- {len(res['turns'])} turns, every stage timer present, "
          f"positive, and summing to the gap")
    return res


if __name__ == "__main__":
    demo()
