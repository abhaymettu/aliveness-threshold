"""Render the condition matrix to wavs + data/stimuli.jsonl.

    .venv/bin/python -m stimgen.generate            # skip clips already rendered
    .venv/bin/python -m stimgen.generate --force    # re-render everything

Prefers the real harness/ if it has landed; falls back to stimgen.harness_stub
otherwise, and writes which one it used into every row as `synth_backend` so a
mixed-backend dataset can never be silently analysed as one thing.
"""

import argparse
import json
import os
import sys

from .design import condition_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_DIR = os.path.join(ROOT, "stimuli")          # gitignored, wavs are build output
OUT = os.path.join(ROOT, "data", "stimuli.jsonl")


def load_backend():
    """Real harness if importable, stub otherwise. Never blocks on the real one."""
    sys.path.insert(0, ROOT)
    try:
        import harness  # noqa
        fn = harness.synthesize_exchange
        return fn, "harness", getattr(harness, "__version__", "unknown")
    except Exception:
        from . import harness_stub
        harness_stub.warm_up()
        return harness_stub.synthesize_exchange, "harness_stub", "say+stdlib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="dev: first N clips")
    args = ap.parse_args()

    synth, backend, version = load_backend()
    print(f"backend: {backend} ({version})", file=sys.stderr)
    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    rows = condition_matrix()[: args.limit]
    prev = {x["stim_id"]: x for x in read_existing()}
    out = []
    for i, r in enumerate(rows, 1):
        wav = os.path.join(WAV_DIR, r["stim_id"] + ".wav")
        rel = os.path.relpath(wav, ROOT)
        # reuse only if the wav is on disk AND we still have its measured
        # timings; a wav with no recorded timing gets re-rendered rather than
        # guessed at.
        res = prev.get(r["stim_id"]) if os.path.exists(wav) else None
        if args.force or res is None or res.get("synth_backend") != backend:
            res = synth(r["prompt_text"], r["response_text"],
                        r["latency_ms"], r["cue"], wav)
        out.append({
            "stim_id": r["stim_id"],
            "latency_ms": r["latency_ms"],
            "cue": r["cue"],
            "exchange_id": r["exchange_id"],
            "prompt_text": r["prompt_text"],
            "response_text": r["response_text"],
            "wav_path": rel,
            "actual_gap_ms": res["actual_gap_ms"],
            "cue_onset_ms": res["cue_onset_ms"],
            # CONTRACT.md asks for this explicitly: without it the cost side
            # of the cost/benefit table is unreported. 0.0 when cue is none.
            "cue_dur_ms": res["component_timings"]["cue_ms"],
            "component_timings": res["component_timings"],
            "synth_backend": backend,
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", file=sys.stderr)

    with open(OUT, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows -> {os.path.relpath(OUT, ROOT)}", file=sys.stderr)


def read_existing():
    if not os.path.exists(OUT):
        return []
    with open(OUT) as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    main()
