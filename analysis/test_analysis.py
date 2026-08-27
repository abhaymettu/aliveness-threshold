"""The runnable check behind the headline.

    .venv/bin/python analysis/test_analysis.py

The claim in README.md is that these judges do not track conversational
timing. That claim is only worth anything if the analysis WOULD have seen
timing sensitivity had it been there. So the fixture is the positive control:
SIMULATED_* is generated with a real latency response and a real cue effect
baked in, and these assertions require the estimators to find both. If they
cannot recover an effect that is present by construction, the flat result on
the real ratings means nothing and this test fails loudly.

Also checked: the shuffled-cue null returns null, the variance components
partition to 1, and the SIMULATED_ quarantine holds.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
from run import MATCHED, CUED, is_simulated  # noqa: E402
from simulate import CUES, LATENCIES  # noqa: E402


def build(tmp):
    subprocess.check_call(
        [sys.executable, os.path.join(HERE, "simulate.py"), "--outdir", tmp,
         "--humans", "30", "--llms", "6", "--exchanges", "8", "--seed", "3"],
        stdout=subprocess.DEVNULL)
    return core.load(f"{tmp}/SIMULATED_ratings.jsonl",
                     f"{tmp}/SIMULATED_stimuli.jsonl")


def ci(v):
    return v["values"] if "values" in v else v


def excludes_zero(v):
    c = v["ci"]
    return c is not None and (c[0] > 0 or c[1] < 0)


def brackets(v):
    return v["ci"] is not None and v["ci"][0] <= v["est"] <= v["ci"][1]


def main():
    tmp = tempfile.mkdtemp()
    d = build(tmp)
    hum = core.subset(d, d["rater_type"] == "human")

    # --- loader contract
    assert len(d["x"]) == len(d["cue"]) == len(d["alive"]) == len(d["rater_id"])
    assert set(np.unique(d["cue"])) == set(CUES)
    assert d["n_unmatched"] == 0, "every rating must join to a stimulus"
    assert not np.all(np.isin(d["x"], LATENCIES)), \
        "x must carry renderer jitter from actual_gap_ms, not the nominal cell"

    # --- POSITIVE CONTROL 1: the fixture's latency response must be detected.
    # The simulated humans sit on a falling sigmoid, so aliveness inside
    # cue=none must slope down and the interval must clear zero.
    s = ci(core.latency_slope(hum, "alive", cues=["none"], n_boot=400, seed=5))
    slope = s["slope_per_s"]
    assert slope["est"] < 0, f"fixture slope should be negative, got {slope['est']:.2f}"
    assert excludes_zero(slope), \
        f"analysis missed a latency response that is there by construction: {slope}"
    assert brackets(slope), "point estimate outside its own CI"

    # --- POSITIVE CONTROL 2: the matched-gap cue contrast must be detected.
    c = ci(core.contrast(hum, "alive", CUED, ["none"], nominal_a=MATCHED,
                         nominal_b=MATCHED, n_boot=400, seed=5))["diff"]
    assert c["est"] > 0 and excludes_zero(c), f"fixture cue effect missed: {c}"

    # --- NULL: shuffling the cue label must kill the cue contrast.
    # Negative findings have to stay negative, including in the test.
    null = core.subset(hum, np.ones(len(hum["x"]), bool))
    null["cue"] = np.random.default_rng(0).permutation(null["cue"])
    nc = ci(core.contrast(null, "alive", CUED, ["none"], nominal_a=MATCHED,
                          nominal_b=MATCHED, n_boot=400, seed=1))["diff"]
    assert not excludes_zero(nc), f"shuffled cues produced a 'real' effect: {nc}"

    # --- the contrast must respect the matched-gap restriction it advertises
    r = core.contrast(hum, "alive", CUED, ["none"], nominal_a=MATCHED,
                      nominal_b=MATCHED, n_boot=50, seed=1)
    assert r["a"]["nominal"] == MATCHED and r["b"]["nominal"] == MATCHED
    assert r["n_ratings"] == r["a"]["n_ratings"] + r["b"]["n_ratings"]

    # --- variance components are a partition of 1
    v = core.variance_decomposition(d, "alive")
    tot = (v["unique_condition"] + v["unique_rater"] + v["shared"] + v["residual"])
    assert abs(tot - 1.0) < 1e-6, f"variance components sum to {tot}"

    # --- per_rater reports one row per rater and no fabricated CIs
    pr = core.per_rater(hum, "alive")
    assert pr["n_raters"] == len(np.unique(hum["rater_id"])) == len(pr["raters"])
    assert all("ci" not in row for row in pr["raters"].values())

    # --- every estimator carries its own n
    assert core.latency_slope(hum, "alive", cues=["none"], n_boot=50)["n_ratings"] > 0
    assert core.cell_means(hum, "alive", n_boot=50)["n_per_level"]

    # --- the simulated-output quarantine
    assert is_simulated("SIMULATED_ratings.jsonl")
    assert is_simulated("/a/b/SIMULATED_ratings.jsonl")
    assert not is_simulated("data/ratings.jsonl")
    assert not is_simulated(None)

    print(f"positive control: slope {slope['est']:+.2f}/s "
          f"[{slope['ci'][0]:.2f}, {slope['ci'][1]:.2f}], "
          f"cue effect {c['est']:+.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]")
    print(f"null control: shuffled cue effect {nc['est']:+.2f} "
          f"[{nc['ci'][0]:.2f}, {nc['ci'][1]:.2f}] -- includes zero, as required")
    print("all checks passed")


if __name__ == "__main__":
    main()
