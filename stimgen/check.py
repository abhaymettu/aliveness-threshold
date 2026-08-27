"""Self-check: is the matrix balanced, and is every row backed by a real wav?

    .venv/bin/python -m stimgen.check

Plain asserts, no test framework. Fails loudly on the two ways this dataset
can quietly rot: the counterbalance drifting, and stimuli.jsonl pointing at
wavs that were never rendered (or were rendered by a different backend).
"""

import collections
import json
import os
import wave

from .design import CUE_ORDER, LATENCIES
from .exchanges import EXCHANGE_IDS
from .generate import OUT, ROOT

TOL_MS = 60.0   # wav length vs recorded component_timings


def load():
    with open(OUT) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    rows = load()
    n_expected = len(EXCHANGE_IDS) * len(CUE_ORDER)
    assert len(rows) == n_expected, f"{len(rows)} rows, expected {n_expected}"
    ids = [r["stim_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate stim_id"

    # --- balance -----------------------------------------------------------
    cell = collections.Counter((r["latency_ms"], r["cue"]) for r in rows)
    per_cell = n_expected // (len(LATENCIES) * len(CUE_ORDER))
    for lat in LATENCIES:
        for cue in CUE_ORDER:
            assert cell[(lat, cue)] == per_cell, \
                f"cell ({lat}, {cue}) has {cell[(lat, cue)]}, expected {per_cell}"

    by_cue = collections.Counter(r["cue"] for r in rows)
    assert len(set(by_cue.values())) == 1, f"cue counts unbalanced: {dict(by_cue)}"
    by_lat = collections.Counter(r["latency_ms"] for r in rows)
    assert len(set(by_lat.values())) == 1, f"latency counts unbalanced: {dict(by_lat)}"

    # cue must not be confounded with content: each exchange heard once per cue
    ex_cue = collections.Counter((r["exchange_id"], r["cue"]) for r in rows)
    for ex in EXCHANGE_IDS:
        for cue in CUE_ORDER:
            assert ex_cue[(ex, cue)] == 1, f"{ex} x {cue} appears {ex_cue[(ex, cue)]}x"

    # latency must be rotated against content, not nested in it
    for ex in EXCHANGE_IDS:
        lats = [r["latency_ms"] for r in rows if r["exchange_id"] == ex]
        assert len(set(lats)) == len(lats) == len(CUE_ORDER), \
            f"{ex} repeats a latency: {sorted(lats)}"

    # --- every row is backed by real audio ---------------------------------
    backends = {r["synth_backend"] for r in rows}
    assert len(backends) == 1, f"mixed synth backends in one dataset: {backends}"

    for r in rows:
        path = os.path.join(ROOT, r["wav_path"])
        assert os.path.exists(path), f"missing wav: {r['wav_path']}"
        with wave.open(path, "rb") as w:
            dur_ms = 1000.0 * w.getnframes() / w.getframerate()
        assert dur_ms > 500, f"{r['stim_id']} is only {dur_ms:.0f}ms"
        claimed = r["component_timings"]["total_ms"]
        assert abs(dur_ms - claimed) < TOL_MS, \
            f"{r['stim_id']}: wav is {dur_ms:.0f}ms, jsonl claims {claimed}ms"

    # --- timing claims are internally consistent ---------------------------
    for r in rows:
        assert r["actual_gap_ms"] >= r["latency_ms"], \
            f"{r['stim_id']} gap shrank below nominal"
        if r["cue"] == "none":
            assert r["cue_onset_ms"] is None
            assert r["cue_dur_ms"] == 0.0
            assert r["actual_gap_ms"] == float(r["latency_ms"]), \
                f"{r['stim_id']} has no cue but a padded gap"
        else:
            assert r["cue_onset_ms"] is not None
            t = r["component_timings"]
            assert r["cue_dur_ms"] == t["cue_ms"], f"{r['stim_id']} cue_dur_ms drift"
            assert r["cue_dur_ms"] > 0, f"{r['stim_id']} claims a cue but no audio"
            assert r["cue_onset_ms"] + t["cue_ms"] <= r["actual_gap_ms"], \
                f"{r['stim_id']} cue spills past the gap into the response"

    # how much of the design got clamped -- reported, not hidden
    clamped = [r for r in rows if r["actual_gap_ms"] > r["latency_ms"]]
    print(f"ok: {len(rows)} stimuli, {len(set(ids))} unique, "
          f"{per_cell}/cell across {len(LATENCIES)}x{len(CUE_ORDER)} cells, "
          f"backend={backends.pop()}")
    print(f"    {len(clamped)} clips have actual_gap > nominal latency "
          f"(cue did not fit); analysis must use actual_gap_ms")
    by = collections.Counter(r["latency_ms"] for r in clamped)
    print("    clamped by nominal latency:",
          {k: by[k] for k in LATENCIES if by[k]})


if __name__ == "__main__":
    main()
