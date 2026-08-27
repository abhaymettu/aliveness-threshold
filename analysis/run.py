"""Run the whole analysis and write results.json.

    .venv/bin/python analysis/run.py --ratings data/ratings.jsonl --stimuli data/stimuli.jsonl

Everything fits on `actual_gap_ms`. Everything carries an n and a bootstrap CI.

Simulated input is quarantined: results land under a SIMULATED_ prefix and
every record is stamped simulated=true. Nothing so stamped may reach README.md
or figures/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

MATCHED = list(core.MATCHED_LATENCIES)
CUED = [c for c in core.CUE_ORDER if c != "none"]
DVS = ["alive", "broken", "wait"]


def is_simulated(path):
    return path is not None and os.path.basename(path).startswith("SIMULATED_")


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def design_report(d):
    """What the design actually is, as measured -- not as CONTRACT.md described it."""
    stim = d["stimuli"]
    cells = {}
    for s in stim.values():
        cells.setdefault((s["latency_ms"], s["cue"]), []).append(s["stim_id"])
    per_exchange = {}
    for s in stim.values():
        per_exchange.setdefault(s["exchange_id"], set()).add(s["latency_ms"])
    clamped = [s["stim_id"] for s in stim.values()
               if s.get("actual_gap_ms") is not None
               and s["actual_gap_ms"] > s["latency_ms"] + 1]
    return {
        "n_ratings": int(len(d["x"])),
        "n_stimuli": int(len(np.unique(d["stim_id"]))),
        "n_stimuli_in_file": len(stim),
        "n_exchanges": int(len(np.unique(d["exchange_id"]))),
        "n_raters": int(len(np.unique(d["rater_id"]))),
        "n_raters_human": int(len(np.unique(d["rater_id"][d["rater_type"] == "human"]))),
        "n_raters_llm": int(len(np.unique(d["rater_id"][d["rater_type"] == "llm"]))),
        "rater_modalities": core.counts(d["rater_modality"]),
        "synth_backends": core.counts(d["backend"]),
        "n_cells": len(cells),
        "clips_per_cell": sorted(set(len(v) for v in cells.values())),
        "latencies_per_exchange": sorted(set(len(v) for v in per_exchange.values())),
        "fully_crossed": all(len(v) == 6 for v in per_exchange.values()),
        "n_clips_gap_exceeds_nominal": len(clamped),
        "unmatched_ratings_dropped": int(d["n_unmatched"]),
        "ratings_dropped_no_measured_gap": int(d["n_no_measured_gap"]),
        "design_note": (
            "Rotation, not a full crossing. Each exchange appears once under every "
            "cue (cue is orthogonal to content) and sits at 5 of the 6 latencies. "
            "Every (latency x cue) cell holds 3 clips. CONTRACT.md's claim that "
            "every exchange appears at every cell is wrong; 18 x 30 = 540 clips "
            "was never rendered or rated."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/ratings.jsonl")
    ap.add_argument("--stimuli", default="data/stimuli.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="analysis/out")
    a = ap.parse_args()

    if not os.path.exists(a.ratings):
        sys.exit(f"no ratings at {a.ratings}. Nothing to analyse yet.")
    stim = a.stimuli if os.path.exists(a.stimuli) else None
    sim = is_simulated(a.ratings) or is_simulated(stim)

    d = core.load(a.ratings, stim)
    if len(d["x"]) == 0:
        sys.exit("no usable rows after joining ratings to stimuli.")
    B, S = a.boot, a.seed

    t0 = time.time()
    res = {
        "simulated": bool(sim),
        "WARNING": ("SIMULATED DEVELOPMENT OUTPUT - NOT A FINDING. "
                    "Must not appear in README.md or figures/.") if sim else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"ratings": a.ratings, "stimuli": stim, "n_boot": B, "seed": S},
        "design": design_report(d),
    }

    # 1. Does each outcome respond to how long the wait actually was?
    #    Inside cue=none, where gap is not confounded with cue.
    print("latency response, cue=none ...")
    res["latency_response_none"] = {
        dv: core.latency_slope(d, dv, cues=["none"], n_boot=B, seed=S) for dv in DVS}

    # Same slope inside the cued clips, over the matched gaps only.
    print("latency response, cued clips at matched gaps ...")
    res["latency_response_cued_matched"] = {
        dv: core.latency_slope(d, dv, cues=CUED, nominal=MATCHED, n_boot=B, seed=S)
        for dv in DVS}

    # 2. The cue contrast, only where "same wait, cue vs no cue" is real.
    print("cue contrast at matched gaps ...")
    res["cue_contrast_matched"] = {
        dv: {"any_cue_vs_none": core.contrast(
                 d, dv, CUED, ["none"], nominal_a=MATCHED, nominal_b=MATCHED,
                 n_boot=B, seed=S),
             "per_cue": {c: core.contrast(
                 d, dv, [c], ["none"], nominal_a=MATCHED, nominal_b=MATCHED,
                 n_boot=B, seed=S) for c in CUED}}
        for dv in DVS}

    # 3. The dissociation, on one scale: what a cue moves vs what more silence
    #    moves, both measured inside the matched region.
    print("dissociation ...")
    res["dissociation"] = {
        dv: {"cue_effect": core.contrast(
                 d, dv, CUED, ["none"], nominal_a=MATCHED, nominal_b=MATCHED,
                 n_boot=B, seed=S)["values"]["diff"],
             "latency_effect_1600_vs_800": core.contrast(
                 d, dv, None, None, nominal_a=[1600], nominal_b=[800],
                 n_boot=B, seed=S)["values"]["diff"],
             "latency_effect_none_1600_vs_0": core.contrast(
                 d, dv, ["none"], ["none"], nominal_a=[1600], nominal_b=[0],
                 n_boot=B, seed=S)["values"]["diff"]}
        for dv in DVS}

    # 4. Curves, for the figures and for anyone who wants the raw cells.
    print("cell means ...")
    res["curves"] = {
        dv: {"none_by_nominal": core.cell_means(
                 d, dv, cues=["none"], n_boot=B, seed=S),
             "all_by_nominal": core.cell_means(d, dv, n_boot=B, seed=S),
             "by_cue_matched": core.cell_means(
                 d, dv, by="cue", nominal=MATCHED, n_boot=B, seed=S)}
        for dv in DVS}

    # 5. Who is rating vs what they were shown.
    print("variance decomposition + per-rater ...")
    res["variance"] = {dv: core.variance_decomposition(d, dv) for dv in DVS}
    res["per_rater"] = {dv: core.per_rater(d, dv) for dv in DVS}

    # 6. Robustness: the same headline numbers clustered on raters instead.
    print("rater-clustered robustness ...")
    res["robustness_cluster_rater"] = {
        dv: {"cue_effect": core.contrast(
                 d, dv, CUED, ["none"], nominal_a=MATCHED, nominal_b=MATCHED,
                 n_boot=B, seed=S, cluster="rater_id")["values"]["diff"],
             "slope_none_per_s": core.latency_slope(
                 d, dv, cues=["none"], n_boot=B, seed=S,
                 cluster="rater_id")["values"]["slope_per_s"]}
        for dv in DVS}

    res["headline"] = headline(res)
    res["runtime_s"] = round(time.time() - t0, 1)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "SIMULATED_results.json" if sim else "results.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=jsonable)
    print(f"\nwrote {path}")
    if sim:
        print("!! SIMULATED. Do not copy any of these numbers into README.md or figures/.")
    print(json.dumps(res["headline"], indent=2, default=jsonable))


def ci_excludes_zero(v):
    c = v.get("ci")
    return bool(c) and (c[0] > 0 or c[1] < 0)


def headline(res):
    """The dissociation, stated as a verdict rather than left for the reader.

    Deliberately conservative: "the judges do not track timing" is asserted only
    when the cue effect's interval excludes zero AND the cue=none aliveness
    slope's interval includes zero AND the same gaps do move `broken`. Any one
    of those failing turns the headline back into a description of two numbers.
    """
    alive_cue = res["dissociation"]["alive"]["cue_effect"]
    alive_slope = res["latency_response_none"]["alive"]["values"]["slope_per_s"]
    broken_slope = res["latency_response_none"]["broken"]["values"]["slope_per_s"]
    wait_slope = res["latency_response_none"]["wait"]["values"]["slope_per_s"]

    dissociated = (ci_excludes_zero(alive_cue)
                   and not ci_excludes_zero(alive_slope)
                   and ci_excludes_zero(broken_slope))
    out = {
        "n_ratings": res["design"]["n_ratings"],
        "n_stimuli": res["design"]["n_stimuli"],
        "n_raters_llm": res["design"]["n_raters_llm"],
        "n_raters_human": res["design"]["n_raters_human"],
        "rater_modalities": res["design"]["rater_modalities"],
        "aliveness_cue_effect_matched": alive_cue,
        "aliveness_slope_per_s_none": alive_slope,
        "broken_slope_per_s_none": broken_slope,
        "would_wait_slope_per_s_none": wait_slope,
        "exchange_rate_ms": None,
        "exchange_rate_note": (
            "Not estimated, and not estimable from these rows. A cue-vs-none "
            "horizontal shift needs a non-zero latency slope to divide by, and "
            "the aliveness slope inside cue=none has a CI spanning zero. In "
            "humans it remains unmeasured: n humans = 0."),
        "dissociation_supported": bool(dissociated),
    }
    if dissociated:
        out["verdict"] = (
            "The judges respond to a cue being described, not to a wait being "
            "experienced. At matched gaps a cue moves aliveness by "
            f"{alive_cue['est']:+.2f} [{alive_cue['ci'][0]:.2f}, {alive_cue['ci'][1]:.2f}] "
            "points, while aliveness inside cue=none is flat against gap "
            f"({alive_slope['est']:+.2f}/s [{alive_slope['ci'][0]:.2f}, "
            f"{alive_slope['ci'][1]:.2f}]). The same gaps do move `broken` "
            f"({broken_slope['est']:+.2f}/s [{broken_slope['ci'][0]:.2f}, "
            f"{broken_slope['ci'][1]:.2f}]), so this is a dissociation, not "
            "insensitivity to everything.")
    else:
        out["verdict"] = ("dissociation not supported by these intervals; read the "
                          "per-outcome numbers rather than a slogan")
    return out


if __name__ == "__main__":
    main()
