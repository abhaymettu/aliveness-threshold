"""One runnable check: does the estimator recover a known exchange rate?

    .venv/bin/python analysis/test_analysis.py

The fixture is generated with TRUE_SHIFTS_MS baked in. If the shift model is
wrong, the recovery assertions fail. That is the whole point of simulating.
"""
import os, subprocess, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from simulate import TRUE_SHIFTS_MS, TRUE_X50, CUES, LATENCIES


def build(tmp):
    subprocess.check_call([sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulate.py"),
        "--outdir", tmp, "--humans", "30", "--llms", "6", "--exchanges", "8",
        "--seed", "3"], stdout=subprocess.DEVNULL)
    return core.load(f"{tmp}/SIMULATED_ratings.jsonl", f"{tmp}/SIMULATED_stimuli.jsonl")


def main():
    tmp = tempfile.mkdtemp()
    d = build(tmp)

    # --- contract: loader shape
    assert len(d["x"]) == len(d["cue"]) == len(d["alive"]) == len(d["rater_id"])
    assert set(np.unique(d["cue"])) == set(CUES)
    assert d["n_unmatched"] == 0, "every rating must join to a stimulus"

    # --- loader must prefer measured gap over the nominal cell
    assert not np.all(np.isin(d["x"], LATENCIES)), \
        "x should carry renderer jitter from actual_gap_ms, not the nominal latency"

    # --- reference cue is pinned at zero shift
    hum = core.subset(d, d["rater_type"] == "human")
    f = core.fit_shift_continuous(hum["x"], hum["cue"], hum["alive"])
    assert f.cues[0] == "none", "the none cue must be the reference"
    assert f.shifts_ms["none"] == 0.0

    # --- RECOVERY: human-only fit must land near the ground truth
    errs = {}
    for c, true in TRUE_SHIFTS_MS.items():
        est = f.shifts_ms[c]
        errs[c] = est - true
        assert abs(est - true) < 80, f"{c}: recovered {est:.0f} ms, truth {true:.0f} ms"
    assert abs(f.x50_ms - TRUE_X50) < 120, f"x50 {f.x50_ms:.0f} vs truth {TRUE_X50}"

    # --- ordering must be preserved
    order = [c for c, _ in sorted(f.shifts_ms.items(), key=lambda kv: -kv[1])]
    truth_order = [c for c, _ in sorted(TRUE_SHIFTS_MS.items(), key=lambda kv: -kv[1])]
    assert order == truth_order, f"ranking {order} != truth {truth_order}"

    # --- a null fixture must return a null result (negative findings stay negative)
    null = core.subset(d, np.ones(len(d["x"]), bool))
    rng = np.random.default_rng(0)
    null["cue"] = rng.permutation(null["cue"])  # break the cue/latency pairing
    nb = core.bootstrap_shifts(null, "alive", n_boot=200, seed=1)
    for c, r in nb["shifts"].items():
        if c == nb["reference"]:
            continue
        assert not r.get("excludes_zero"), \
            f"shuffled cues produced a 'real' shift for {c}: {r}"

    # --- bootstrap CI must bracket the point estimate and the truth
    bs = core.bootstrap_shifts(hum, "alive", n_boot=300, seed=5)
    for c, r in bs["shifts"].items():
        if c == bs["reference"] or r["ci_lo"] is None:
            continue
        assert r["ci_lo"] <= r["shift_ms"] <= r["ci_hi"], f"{c}: point outside its own CI"
        assert r["ci_lo"] <= TRUE_SHIFTS_MS[c] <= r["ci_hi"], \
            f"{c}: 95% CI [{r['ci_lo']:.0f},{r['ci_hi']:.0f}] misses truth {TRUE_SHIFTS_MS[c]}"

    # --- binary DV fits and orders the same way
    fb = core.fit_shift_binary(hum["x"], hum["cue"], hum["wait"])
    ob = [c for c, _ in sorted(fb.shifts_ms.items(), key=lambda kv: -kv[1])]
    assert ob == truth_order, f"would_wait ranking {ob} != truth {truth_order}"

    # --- variance decomposition components are a partition of 1
    v = core.variance_decomposition(d, "alive")
    tot = v["unique_condition"] + v["unique_rater"] + v["shared"] + v["residual"]
    assert abs(tot - 1.0) < 1e-6, f"variance components sum to {tot}"

    # --- agreement must detect the attenuation the fixture built in
    ag = core.llm_human_agreement(d, "alive", n_boot=200, seed=2)
    assert ag["available"]
    div = ag["exchange_rate_divergence"]
    assert all(v["ratio_llm_over_human"] < 0.7 for v in div.values()), \
        "fixture attenuates llm shifts to 0.25x; agreement check failed to see it"
    assert ag["pearson_r"] > 0.8, "fixture llm raters should still correlate on stimulus means"

    # --- the simulated-output quarantine
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run import is_simulated
    assert is_simulated("SIMULATED_ratings.jsonl")
    assert is_simulated("/a/b/SIMULATED_ratings.jsonl")
    assert not is_simulated("data/ratings.jsonl")

    print("recovery errors (est - truth, ms):",
          {c: round(v) for c, v in errs.items()})
    print("all checks passed")


if __name__ == "__main__":
    main()
