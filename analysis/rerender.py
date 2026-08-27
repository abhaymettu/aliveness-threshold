"""Re-render the 90 design cells through the REAL harness, not the stub.

    .venv/bin/python analysis/rerender.py

Writes analysis/rerender.jsonl (one row per design cell) and the wavs into
stimuli/. It does NOT touch data/stimuli.jsonl, which belongs to the stimulus
agent and, more importantly, records the stub timings that the LLM judges were
actually shown. Overwriting it would silently re-key 540 existing ratings to
timings nobody described to a judge.

The one thing this script exists to establish: the stub CLAMPS a gap open when
the cue does not fit; the real harness REFUSES. Those refusals are the design's
structurally impossible cells and they are recorded as rows with
`renderable: false` and the harness's own error string, never worked around.

Floors are measured here rather than copied from harness/STATUS.md, because
STATUS.md lists the cue *durations* (305/320/510/585 ms) under a heading that
calls them minimum latencies. The real floor is cue_onset_ms + duration.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # noqa: E402
from stimgen.design import CUE_ORDER, LATENCIES, condition_matrix  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "rerender.jsonl")
WAV_DIR = os.path.join(ROOT, "stimuli")
CUE_ONSET_MS = 150.0  # harness default; the floor is this + cue duration


def floors():
    return {c: (0.0 if c == "none" else CUE_ONSET_MS + harness.cue_duration_ms(c))
            for c in CUE_ORDER}


def main():
    os.makedirs(WAV_DIR, exist_ok=True)
    f = floors()
    print("measured floors (ms): " + "  ".join(f"{k}={v:.0f}" for k, v in f.items()),
          file=sys.stderr)

    rows, ok, impossible = [], 0, 0
    for i, r in enumerate(condition_matrix(), 1):
        wav = os.path.join(WAV_DIR, r["stim_id"] + ".wav")
        base = {"stim_id": r["stim_id"], "exchange_id": r["exchange_id"],
                "cue": r["cue"], "latency_ms": r["latency_ms"],
                "synth_backend": "harness"}
        try:
            res = harness.synthesize_exchange(
                r["prompt_text"], r["response_text"], r["latency_ms"], r["cue"], wav)
        except ValueError as e:
            impossible += 1
            # a stub-clamped wav for this cell may still be on disk. Leaving it
            # there would let a stub clip masquerade as real-harness audio.
            if os.path.exists(wav):
                os.remove(wav)
            rows.append({**base, "renderable": False, "reason": str(e),
                         "wav_path": None, "actual_gap_ms": None,
                         "cue_onset_ms": None, "cue_dur_ms": None})
            continue
        ok += 1
        rows.append({**base, "renderable": True, "reason": None,
                     "wav_path": os.path.relpath(wav, ROOT),
                     "actual_gap_ms": res["actual_gap_ms"],
                     "cue_onset_ms": res["cue_onset_ms"],
                     "cue_dur_ms": res["cue_duration_ms"],
                     "tts_backend": res["tts_backend"],
                     "verified": res["verification"]["verified"],
                     "reasons": res["verification"]["reasons"],
                     "gap_err_ms": res["verification"].get("gap_err_ms"),
                     "cue_onset_err_ms": res["verification"].get("cue_onset_err_ms")})
        if i % 15 == 0:
            print(f"  {i}/90", file=sys.stderr)

    with open(OUT, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    errs = [abs(r["gap_err_ms"]) for r in rows if r.get("gap_err_ms") is not None]
    print(f"rendered {ok}, structurally impossible {impossible}, "
          f"worst |gap err| {max(errs):.1f}ms" if errs else f"rendered {ok}",
          file=sys.stderr)
    print(f"wrote {len(rows)} rows -> analysis/rerender.jsonl", file=sys.stderr)

    # every impossible cell must be a cue whose floor genuinely exceeds the cell
    for r in rows:
        if not r["renderable"]:
            assert f[r["cue"]] > r["latency_ms"], r
    for lat in LATENCIES:
        n = sum(1 for r in rows if r["latency_ms"] == lat and r["renderable"])
        print(f"  {lat:>5}ms renderable {n}/15", file=sys.stderr)


if __name__ == "__main__":
    main()
