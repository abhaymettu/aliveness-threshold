"""Offline stimulus rendering: one prompt/response exchange with a controlled
gap and a controlled non-verbal cue inside that gap.

This is the API the stimulus-generation side depends on. Treat the signature
and the returned keys as a contract.

    from harness import synthesize_exchange

    info = synthesize_exchange(
        prompt_text="What time do you close on Sunday?",
        response_text="We close at six on Sundays.",
        latency_ms=800,
        cue="filled_pause",          # none|filled_pause|breath|backchannel|verbal_stall
        out_wav_path="stimuli/001.wav",
        cue_onset_ms=150,            # optional, default 150
    )

Timeline of the rendered wav (mono, 16-bit PCM, 22050 Hz)::

    |<-lead->|<--- prompt --->|<------- gap: latency_ms ------->|<-- response -->|<-tail->|
                              ^ prompt offset                   ^ response onset
                              |<-cue_onset_ms->|<-- cue -->|

Definitions, because the whole study hangs on them:

- ``latency_ms`` is measured from the **offset of prompt speech** to the
  **onset of response speech**. Both utterances are trimmed of TTS
  leading/trailing silence first; untrimmed clips inflate the gap by a few
  hundred ms without telling you.
- The cue lives **inside** the gap. It never extends it. A cue that cannot fit
  (``cue_onset_ms + cue_duration > latency_ms``) raises ``ValueError`` rather
  than quietly producing a different stimulus than you asked for. Call
  ``cue_duration_ms(cue)`` to plan your design cells.

Returned dict (keys marked * are the guaranteed contract)::

    {
      "wav_path": str,             *  path written
      "actual_gap_ms": float,      *  measured from the written file
      "cue_onset_ms": float,       *  measured from the written file (None if cue="none")
      "component_timings": {...},  *  wall-clock ms per stage of *this render*
      "requested": {...},             what you asked for
      "nominal": {...},               what was assembled, sample-exact
      "verification": {...},          measurement method, per-landmark error, verified bool
      "cue": str, "cue_source": str,  "tts_backend": str,
      "sample_rate": int, "duration_ms": float,
    }

``actual_gap_ms`` and ``cue_onset_ms`` are re-measured from the file on disk by
voice-activity segmentation, not copied from the request. When segmentation
cannot resolve a landmark, ``verification["verified"]`` is False, the reason is
given, and the nominal (sample-exact) value is returned in its place -- flagged,
never silently substituted.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import audio, cues

LEAD_MS = 100.0  # silence before the prompt so nothing clips at file start
TAIL_MS = 200.0  # silence after the response
DEFAULT_CUE_ONSET_MS = 150.0

# verification search radius around each nominal landmark
SEARCH_MS = 40.0
# segments closer together than this are treated as one utterance during
# verification; also the minimum cue separation we can resolve
VERIFY_MERGE_MS = 30.0


def cue_duration_ms(cue: str, voice=None) -> float:
    """Duration of a cue's audio in ms. 0.0 for ``none``. Use this to work out
    which (latency_ms, cue) design cells are physically renderable."""
    return audio.millis(len(cues.cue_audio(cue, voice=voice)))


def _find_landmark(segs, nominal_ms, which, search_ms=SEARCH_MS):
    """Last segment end / first segment start within +-search_ms of nominal."""
    lo, hi = nominal_ms - search_ms, nominal_ms + search_ms
    if which == "end":
        hits = [e for _, e in segs if lo <= e <= hi]
        return float(hits[-1]) if hits else None
    hits = [s for s, _ in segs if lo <= s <= hi]
    return float(hits[0]) if hits else None


def _cue_onset_in_gap(x, gap_start_ms, gap_end_ms):
    """Onset of the cue, measured inside the gap with a threshold relative to
    the cue's own level.

    A cue is quiet -- a filled pause sits 20-30 dB under the surrounding speech.
    Judged against the whole file's peak it appears to start ~30ms late, because
    its attack is below a file-global threshold. What we actually want is where
    the cue rises out of the silence around it, so the gap is segmented on its
    own.
    """
    a, b = audio.samples(gap_start_ms), audio.samples(gap_end_ms)
    seg = audio.segments(x[a:b], merge_gap_ms=VERIFY_MERGE_MS, min_len_ms=20.0)
    return gap_start_ms + float(seg[0][0]) if seg else None


def measure_exchange(wav_path, nominal: dict, has_cue: bool) -> dict:
    """Re-measure a rendered exchange from disk. Returns measured landmarks in
    ms plus the error against `nominal`. Used by synthesize_exchange, and
    usable standalone to audit any stimulus file this repo produced."""
    x = audio.read(wav_path)
    segs = audio.segments(x, merge_gap_ms=VERIFY_MERGE_MS, min_len_ms=20.0)

    p_off = _find_landmark(segs, nominal["prompt_offset_ms"], "end")
    r_on = _find_landmark(segs, nominal["response_onset_ms"], "start")
    c_on = None
    if has_cue and p_off is not None and r_on is not None:
        c_on = _cue_onset_in_gap(x, p_off, r_on)

    out = {
        "method": (
            f"frame-RMS VAD, {audio.FRAME_MS}ms frames, -35dB rel / -55dBFS abs, "
            f"segments merged below {VERIFY_MERGE_MS}ms, landmark searched "
            f"+-{SEARCH_MS}ms around nominal"
        ),
        "n_segments": len(segs),
        "segments_ms": [(round(float(a), 1), round(float(b), 1)) for a, b in segs],
        "measured_prompt_offset_ms": p_off,
        "measured_response_onset_ms": r_on,
        "measured_cue_onset_abs_ms": c_on,
        "reasons": [],
    }
    if p_off is None:
        out["reasons"].append("prompt offset not found within search window")
    if r_on is None:
        out["reasons"].append("response onset not found within search window")
    if has_cue and c_on is None:
        out["reasons"].append(
            "cue onset not found within search window "
            f"(cue must be >={VERIFY_MERGE_MS}ms clear of prompt and response to resolve)"
        )

    if p_off is not None and r_on is not None:
        out["gap_ms"] = round(r_on - p_off, 3)
        out["gap_err_ms"] = round(out["gap_ms"] - nominal["latency_ms"], 3)
    if p_off is not None and c_on is not None:
        out["cue_onset_ms"] = round(c_on - p_off, 3)
        out["cue_onset_err_ms"] = round(out["cue_onset_ms"] - nominal["cue_onset_ms"], 3)

    out["verified"] = not out["reasons"]
    return out


def synthesize_exchange(
    prompt_text: str,
    response_text: str,
    latency_ms: float,
    cue: str,
    out_wav_path,
    *,
    cue_onset_ms: float = DEFAULT_CUE_ONSET_MS,
    voice=None,
    prompt_voice=None,
) -> dict:
    """Render one exchange to a wav. See the module docstring for the contract."""
    if cue not in cues.CUES:
        raise ValueError(f"unknown cue {cue!r}; expected one of {cues.CUES}")
    if latency_ms < 0:
        raise ValueError("latency_ms must be >= 0")
    if cue_onset_ms < 0:
        raise ValueError("cue_onset_ms must be >= 0")

    from . import tts  # noqa: PLC0415

    t_all = time.perf_counter()
    voice = voice or tts.default_voice()
    prompt_voice = prompt_voice or voice

    p, t_prompt = prompt_voice.synth_timed(prompt_text)
    r, t_resp = voice.synth_timed(response_text)

    t0 = time.perf_counter()
    c = cues.cue_audio(cue, voice=voice)
    t_cue = (time.perf_counter() - t0) * 1000.0

    cue_dur = audio.millis(len(c))
    if cue != "none" and cue_onset_ms + cue_dur > latency_ms:
        raise ValueError(
            f"cue {cue!r} is {cue_dur:.0f}ms and cannot fit in a {latency_ms:.0f}ms gap "
            f"at onset {cue_onset_ms:.0f}ms. Raise latency_ms, lower cue_onset_ms, or "
            f"drop this design cell. (cue_duration_ms('{cue}') = {cue_dur:.0f})"
        )
    if len(p) == 0 or len(r) == 0:
        raise ValueError("prompt_text and response_text must each produce audible speech")

    t0 = time.perf_counter()
    lead, gap, tail = audio.samples(LEAD_MS), audio.samples(latency_ms), audio.samples(TAIL_MS)
    out = np.zeros(lead + len(p) + gap + len(r) + tail, dtype=np.float32)
    out[lead : lead + len(p)] = p
    resp_start = lead + len(p) + gap
    out[resp_start : resp_start + len(r)] = r
    cue_start = None
    if cue != "none":
        cue_start = lead + len(p) + audio.samples(cue_onset_ms)
        out[cue_start : cue_start + len(c)] += c
    t_assemble = (time.perf_counter() - t0) * 1000.0

    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    audio.write(out_wav_path, out)
    t_write = (time.perf_counter() - t0) * 1000.0

    nominal = {
        "prompt_offset_ms": audio.millis(lead + len(p)),
        "response_onset_ms": audio.millis(resp_start),
        "cue_onset_abs_ms": audio.millis(cue_start) if cue_start is not None else None,
        # sample quantization means the assembled gap can differ from the
        # request by up to one sample (0.045ms at 22050 Hz)
        "latency_ms": audio.millis(gap),
        "cue_onset_ms": audio.millis(audio.samples(cue_onset_ms)) if cue != "none" else None,
        "cue_duration_ms": cue_dur,
    }
    ver = measure_exchange(out_wav_path, nominal, has_cue=(cue != "none"))

    return {
        "wav_path": str(out_wav_path),
        "actual_gap_ms": ver.get("gap_ms", nominal["latency_ms"]),
        "cue_onset_ms": ver.get("cue_onset_ms", nominal["cue_onset_ms"]),
        "component_timings": {
            "tts_prompt_ms": round(t_prompt, 2),
            "tts_response_ms": round(t_resp, 2),
            "cue_audio_ms": round(t_cue, 2),
            "assemble_ms": round(t_assemble, 2),
            "write_wav_ms": round(t_write, 2),
            "total_ms": round((time.perf_counter() - t_all) * 1000.0, 2),
        },
        "requested": {
            "prompt_text": prompt_text,
            "response_text": response_text,
            "latency_ms": float(latency_ms),
            "cue": cue,
            "cue_onset_ms": float(cue_onset_ms) if cue != "none" else None,
        },
        "nominal": nominal,
        "verification": ver,
        "cue": cue,
        "cue_source": cues.cue_source(cue),
        "cue_duration_ms": cue_dur,
        "tts_backend": voice.backend,
        "sample_rate": audio.SR,
        "duration_ms": audio.millis(len(out)),
    }


TOLERANCE_MS = 25.0


def demo(out_dir="/tmp/aliveness-demo") -> dict:
    """Self-check. Renders a grid of (latency, cue) cells, re-measures each from
    disk, and asserts every landmark lands within TOLERANCE_MS of the request.

    This is a measurement instrument. If it cannot hit its own targets the
    study is void, so this fails loudly rather than warning.
    """
    from pathlib import Path as _P

    prompt = "What time do you close on Sunday?"
    response = "We close at six on Sundays."
    rows, worst_gap, worst_cue = [], 0.0, 0.0
    _P(out_dir).mkdir(parents=True, exist_ok=True)

    for lat in (200.0, 400.0, 800.0, 1200.0, 2000.0):
        for cue in cues.CUES:
            if cue != "none" and DEFAULT_CUE_ONSET_MS + cue_duration_ms(cue) > lat:
                rows.append((lat, cue, None, None, "skipped: cue does not fit in gap"))
                continue
            info = synthesize_exchange(
                prompt, response, lat, cue, f"{out_dir}/{int(lat)}-{cue}.wav"
            )
            v = info["verification"]
            ge = v.get("gap_err_ms")
            ce = v.get("cue_onset_err_ms")
            rows.append((lat, cue, ge, ce, "" if v["verified"] else "; ".join(v["reasons"])))
            assert v["verified"], f"{lat}/{cue}: unverifiable -- {v['reasons']}"
            assert abs(ge) <= TOLERANCE_MS, f"{lat}/{cue}: gap off by {ge:.1f}ms"
            worst_gap = max(worst_gap, abs(ge))
            if ce is not None:
                assert abs(ce) <= TOLERANCE_MS, f"{lat}/{cue}: cue onset off by {ce:.1f}ms"
                worst_cue = max(worst_cue, abs(ce))

    print(f"{'latency':>8} {'cue':<14} {'gap_err':>8} {'cue_err':>8}  note")
    for lat, cue, ge, ce, note in rows:
        g = f"{ge:+.1f}" if ge is not None else "-"
        c = f"{ce:+.1f}" if ce is not None else "-"
        print(f"{lat:8.0f} {cue:<14} {g:>8} {c:>8}  {note}")
    print(f"\nworst |gap error| {worst_gap:.1f}ms, worst |cue onset error| {worst_cue:.1f}ms, "
          f"tolerance {TOLERANCE_MS:.0f}ms -- PASS")
    return {"worst_gap_err_ms": worst_gap, "worst_cue_onset_err_ms": worst_cue,
            "tolerance_ms": TOLERANCE_MS, "rows": rows}


if __name__ == "__main__":
    demo()
