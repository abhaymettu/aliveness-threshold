"""Run the whole analysis and write results.json.

    .venv/bin/python analysis/run.py --ratings data/ratings.jsonl --stimuli data/stimuli.jsonl

Simulated input is quarantined: results land under a SIMULATED_ prefix and every
record is stamped simulated=true. Nothing so stamped may reach README.md or figures/.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def is_simulated(path):
    return os.path.basename(path).startswith("SIMULATED_")


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def design_report(d):
    cue = d["cue"]
    cells = {}
    for c, s in zip(cue, d["stim_id"]):
        cells.setdefault(c, set()).add(s)
    per_rater = core._counts(d["rater_id"])
    counts = np.asarray(list(per_rater.values()))
    return {
        "n_ratings": int(len(cue)),
        "n_stimuli": int(len(np.unique(d["stim_id"]))),
        "n_exchanges": int(len(np.unique(d["exchange_id"]))),
        "n_raters": int(len(np.unique(d["rater_id"]))),
        "n_raters_human": int(len(np.unique(d["rater_id"][d["rater_type"] == "human"]))),
        "n_raters_llm": int(len(np.unique(d["rater_id"][d["rater_type"] == "llm"]))),
        "stimuli_per_cue": {c: len(v) for c, v in sorted(cells.items())},
        "ratings_per_cue": core._counts(cue),
        "latency_values_ms": sorted(set(np.round(d["x"]).astype(int).tolist()))[:40],
        "ratings_per_rater_min": int(counts.min()) if counts.size else 0,
        "ratings_per_rater_median": float(np.median(counts)) if counts.size else 0,
        "unmatched_ratings_dropped": int(d.get("n_unmatched", 0)),
        "balanced": len(set(core._counts(cue).values())) == 1,
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
    sim = is_simulated(a.ratings) or (stim and is_simulated(stim))

    d = core.load(a.ratings, stim)
    if len(d["x"]) == 0:
        sys.exit("no usable rows after joining ratings to stimuli.")

    t0 = time.time()
    res = {
        "simulated": bool(sim),
        "WARNING": ("SIMULATED DEVELOPMENT OUTPUT - NOT A FINDING. "
                    "Must not appear in README.md or figures/.") if sim else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"ratings": a.ratings, "stimuli": stim, "n_boot": a.boot, "seed": a.seed},
        "design": design_report(d),
    }

    print("fitting exchange rate (aliveness)...")
    res["exchange_rate_aliveness"] = core.bootstrap_shifts(d, "alive", a.boot, a.seed)
    print("fitting exchange rate (would_wait_again)...")
    if np.isfinite(d["wait"]).sum() > 20:
        res["exchange_rate_would_wait"] = core.bootstrap_shifts(d, "wait", a.boot, a.seed)
    else:
        res["exchange_rate_would_wait"] = {"available": False,
            "note": "too few would_wait_again_bool values"}
    print("fitting exchange rate (broken, reversed)...")
    if np.isfinite(d["broken"]).sum() > 20:
        res["exchange_rate_broken_rev"] = core.bootstrap_shifts(d, "broken_rev", a.boot, a.seed)
    else:
        res["exchange_rate_broken_rev"] = {"available": False, "note": "no broken_1_7 values"}

    print("shape-heterogeneity check (free per-cue fits)...")
    res["free_per_cue_aliveness"] = core.fit_free_per_cue(d["x"], d["cue"], d["alive"])

    print("cost vs benefit...")
    res["cost_benefit"] = core.cost_benefit(d, res["exchange_rate_aliveness"])

    print("variance decomposition...")
    res["variance_aliveness"] = core.variance_decomposition(d, "alive")

    print("llm vs human agreement...")
    res["llm_vs_human_aliveness"] = core.llm_human_agreement(d, "alive", a.boot, a.seed)
    if np.isfinite(d["wait"]).sum() > 20:
        res["llm_vs_human_would_wait"] = core.llm_human_agreement(d, "wait", a.boot, a.seed)

    res["headline"] = headline(res)
    res["runtime_s"] = round(time.time() - t0, 1)

    os.makedirs(a.outdir, exist_ok=True)
    name = ("SIMULATED_results.json" if sim else "results.json")
    path = os.path.join(a.outdir, name)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=jsonable)
    print(f"\nwrote {path}")
    if sim:
        print("!! SIMULATED. Do not copy any of these numbers into README.md or figures/.")
    print(json.dumps(res["headline"], indent=2, default=jsonable))


def headline(res):
    """The one paragraph a robotics team needs, plus an honest can-we-say-this flag."""
    er = res.get("exchange_rate_aliveness", {})
    shifts = er.get("shifts", {})
    ref = er.get("reference")
    ranked = sorted(((c, v) for c, v in shifts.items() if c != ref),
                    key=lambda kv: kv[1]["shift_ms"], reverse=True)
    usable = [(c, v) for c, v in ranked
              if v.get("identified") and v.get("excludes_zero")]
    out = {
        "n_ratings": res["design"]["n_ratings"],
        "n_raters": res["design"]["n_raters"],
        "n_stimuli": res["design"]["n_stimuli"],
        "reference_cue": ref,
        "reference_x50_ms": er.get("x50_ms"),
        "any_cue_buys_time": bool(usable),
        "best_cue": usable[0][0] if usable else None,
        "best_cue_ms": usable[0][1]["shift_ms"] if usable else None,
        "best_cue_ci": [usable[0][1].get("ci_lo"), usable[0][1].get("ci_hi")] if usable else None,
        "ranking_ms": [{"cue": c, "shift_ms": v["shift_ms"],
                        "ci": [v.get("ci_lo"), v.get("ci_hi")],
                        "identified": v.get("identified"),
                        "excludes_zero": v.get("excludes_zero")} for c, v in ranked],
    }
    if not ranked:
        out["verdict"] = "no non-reference cue in the data; no exchange rate to estimate"
    elif not any(v.get("identified") for _, v in ranked):
        out["verdict"] = ("design cannot locate the curve: every interval is wider than "
                          "the latency range tested. Report the interval, not a number.")
    elif not usable:
        out["verdict"] = ("no cue buys a detectable amount of latency: every interval "
                          "includes zero. Negative result.")
    else:
        out["verdict"] = (f"{usable[0][0]} buys {usable[0][1]['shift_ms']:.0f} ms "
                          f"[{usable[0][1]['ci_lo']:.0f}, {usable[0][1]['ci_hi']:.0f}]")
    return out


if __name__ == "__main__":
    main()
