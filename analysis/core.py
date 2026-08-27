"""Estimators for the aliveness-threshold rating data.

WHAT THIS FILE NO LONGER DOES, AND WHY

It used to fit an "exchange rate": a horizontal shift between each cue's
aliveness curve and the cue=none curve, in milliseconds. That estimator is
gone. Not because it failed to converge -- because the quantity it names does
not exist in this data.

A horizontal shift is only meaningful if the y-axis responds to x. In these
540 ratings, aliveness within cue=none is flat across the whole latency range
(see `latency_slope` on the none subset, and its CI). Dividing a large vertical
cue effect by a slope indistinguishable from zero produces a number with the
units of milliseconds and the content of a divide-by-zero. Reporting it would
have been the single most dishonest thing this repo could do.

So the estimators here are deliberately blunt and assumption-light:

  - `latency_slope`   OLS of a DV on actual_gap_ms, inside one cue subset
  - `contrast`        difference of two subset means
  - `cell_means`      the raw cell means everything else has to be consistent with
  - `variance_decomposition`  commonality analysis, condition vs rater
  - `boot`            one cluster bootstrap that all of the above run through

Two rules the whole file obeys:

1. x is `actual_gap_ms`, never `latency_ms`. 36 of the 90 stub-rendered clips
   have a gap longer than their nominal cell, because the stub clamped the gap
   open to make room for a cue that did not fit. Fitting on the nominal cell
   would assign those clips a wait they never had.

2. Every returned dict carries its own n. A number without its n is not a
   result, and downstream code should not have to go looking.
"""
from __future__ import annotations

import json

import numpy as np

CUE_ORDER = ["none", "filled_pause", "breath", "backchannel", "verbal_stall"]

# The gaps where a cue and cue=none coexist at the same wait. Below 800 ms the
# stub had to clamp cued clips open, so "same wait, cue vs no cue" does not
# exist there and the contrast would be comparing 460 ms against 200 ms.
MATCHED_LATENCIES = (800, 1200, 1600)


# ---------------------------------------------------------------- data loading

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load(ratings_path, stimuli_path=None):
    """Join ratings to stimuli. Returns a dict of parallel numpy arrays.

    `x` is actual_gap_ms. A row whose stimulus has no measured gap is dropped
    rather than backfilled from the nominal cell, and the count is returned as
    `n_no_measured_gap` so the drop is visible.
    """
    ratings = read_jsonl(ratings_path)
    stim = {s["stim_id"]: s for s in read_jsonl(stimuli_path)} if stimuli_path else {}

    names = ("x", "nominal", "cue", "alive", "broken", "wait", "rater_id",
             "rater_type", "rater_modality", "stim_id", "exchange_id", "backend")
    cols = {k: [] for k in names}
    unmatched = no_gap = 0
    for r in ratings:
        s = stim.get(r["stim_id"])
        if s is None:
            unmatched += 1
            continue
        if s.get("actual_gap_ms") is None:
            no_gap += 1
            continue
        cols["x"].append(float(s["actual_gap_ms"]))
        cols["nominal"].append(float(s["latency_ms"]))
        cols["cue"].append(s["cue"])
        cols["exchange_id"].append(s.get("exchange_id", s["stim_id"]))
        cols["backend"].append(s.get("synth_backend", "unknown"))
        cols["stim_id"].append(r["stim_id"])
        cols["alive"].append(float(r["aliveness_1_7"]))
        cols["broken"].append(float(r.get("broken_1_7", np.nan)))
        w = r.get("would_wait_again_bool")
        cols["wait"].append(np.nan if w is None else float(bool(w)))
        cols["rater_id"].append(r["rater_id"])
        cols["rater_type"].append(r.get("rater_type", "unknown"))
        cols["rater_modality"].append(r.get("rater_modality", "unknown"))

    numeric = {"x", "nominal", "alive", "broken", "wait"}
    d = {k: np.asarray(v, float) if k in numeric else np.asarray(v)
         for k, v in cols.items()}
    d["n_unmatched"] = unmatched
    d["n_no_measured_gap"] = no_gap
    d["stimuli"] = stim
    return d


def subset(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def dv(d, name):
    """Return the DV array. `broken` and `wait` are NOT reversed -- higher
    broken is worse and higher wait is better, and flipping either one here
    would quietly change the sign of every contrast downstream."""
    return {"alive": d["alive"], "broken": d["broken"], "wait": d["wait"]}[name]


# ------------------------------------------------------------ the one bootstrap

def boot(d, stat, cluster="exchange_id", n_boot=2000, seed=7):
    """Cluster bootstrap. `stat(d) -> dict of float` (NaN allowed for undefined).

    Clustering on `exchange_id` by default: the 18 dialogues are the sampled
    thing here, and the CI then answers "would this hold on different content".

    Clustering on `rater_id` is available and reported as a robustness column,
    but with exactly 6 personas -- written to disagree, not drawn from a
    population -- that interval is coarse and does not generalise to "a new
    listener". Rater spread is reported by `variance_decomposition` instead.
    """
    point = stat(d)
    keys = list(point)
    units = d[cluster]
    uu = np.unique(units)
    where = {u: np.flatnonzero(units == u) for u in uu}
    rng = np.random.default_rng(seed)

    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uu, size=len(uu), replace=True)
        ix = np.concatenate([where[u] for u in pick])
        try:
            s = stat(subset(d, ix))
        except (ValueError, np.linalg.LinAlgError):
            continue
        for k in keys:
            draws[k].append(s.get(k, np.nan))

    out = {}
    for k in keys:
        a = np.asarray(draws[k], float)
        a = a[np.isfinite(a)]
        ci = ([float(v) for v in np.percentile(a, [2.5, 97.5])]
              if a.size >= 50 else None)
        p = float(point[k]) if np.isfinite(point[k]) else None
        out[k] = {"est": p, "ci": ci, "n_boot_ok": int(a.size)}
    return {"values": out, "cluster": cluster, "n_clusters": int(len(uu)),
            "n_boot_requested": int(n_boot)}


# --------------------------------------------------------------- the estimators

def _mask(d, cues=None, nominal=None):
    m = np.ones(len(d["x"]), bool)
    if cues is not None:
        m &= np.isin(d["cue"], list(cues))
    if nominal is not None:
        m &= np.isin(d["nominal"], list(nominal))
    return m


def latency_slope(d, name, cues=None, nominal=None, n_boot=2000, seed=7,
                  cluster="exchange_id"):
    """OLS slope of a DV on actual_gap_ms, per 1000 ms of gap, within a subset.

    Reported per second because the design spans 1.6 s and a per-ms slope is
    four leading zeros of nothing.
    """
    dd = subset(d, _mask(d, cues, nominal))
    dd = subset(dd, np.isfinite(dv(dd, name)))
    y = dv(dd, name)

    def stat(s):
        yy = dv(s, name)
        x = s["x"] / 1000.0
        if len(yy) < 3 or np.ptp(x) == 0:
            return {"slope_per_s": np.nan, "intercept": np.nan}
        A = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(A, yy, rcond=None)[0]
        return {"slope_per_s": float(b[1]), "intercept": float(b[0])}

    r = boot(dd, stat, cluster=cluster, n_boot=n_boot, seed=seed)
    r.update(dv_name=name, cues=sorted(set(dd["cue"].tolist())),
             nominal_cells=sorted(set(dd["nominal"].astype(int).tolist())),
             n_ratings=int(len(y)), n_stimuli=int(len(np.unique(dd["stim_id"]))),
             n_raters=int(len(np.unique(dd["rater_id"]))),
             gap_range_ms=[float(dd["x"].min()), float(dd["x"].max())])
    return r


def contrast(d, name, cues_a, cues_b=None, nominal_a=None, nominal_b=None,
             n_boot=2000, seed=7, cluster="exchange_id"):
    """mean(subset A) - mean(subset B) for one DV, with a cluster-bootstrap CI.

    Both subsets are resampled inside the same bootstrap draw, so the interval
    accounts for the two means moving together across content.
    """
    y = dv(d, name)
    ma, mb = _mask(d, cues_a, nominal_a), _mask(d, cues_b, nominal_b)
    dd = subset(d, np.isfinite(y) & (ma | mb))
    ma, mb = _mask(dd, cues_a, nominal_a), _mask(dd, cues_b, nominal_b)
    dd = dict(dd, _a=ma, _b=mb)

    def stat(s):
        yy = dv(s, name)
        a, b = yy[s["_a"]], yy[s["_b"]]
        if a.size == 0 or b.size == 0:
            return {"diff": np.nan, "mean_a": np.nan, "mean_b": np.nan}
        return {"diff": float(a.mean() - b.mean()),
                "mean_a": float(a.mean()), "mean_b": float(b.mean())}

    r = boot(dd, stat, cluster=cluster, n_boot=n_boot, seed=seed)
    r.update(dv_name=name,
             a={"cues": cues_a, "nominal": nominal_a, "n_ratings": int(ma.sum()),
                "n_stimuli": int(len(np.unique(dd["stim_id"][ma])))},
             b={"cues": cues_b, "nominal": nominal_b, "n_ratings": int(mb.sum()),
                "n_stimuli": int(len(np.unique(dd["stim_id"][mb])))},
             n_ratings=int(ma.sum() + mb.sum()))
    return r


def cell_means(d, name, by="nominal", cues=None, nominal=None,
               n_boot=2000, seed=7, cluster="exchange_id"):
    """Mean of a DV in each level of `by`, each with its own bootstrap CI."""
    dd = subset(d, _mask(d, cues, nominal))
    dd = subset(dd, np.isfinite(dv(dd, name)))
    levels = sorted(set(dd[by].tolist()))

    def stat(s):
        yy = dv(s, name)
        out = {}
        for lv in levels:
            v = yy[s[by] == lv]
            out[str(lv)] = float(v.mean()) if v.size else np.nan
        return out

    r = boot(dd, stat, cluster=cluster, n_boot=n_boot, seed=seed)
    r["n_per_level"] = {str(lv): int((dd[by] == lv).sum()) for lv in levels}
    r["stimuli_per_level"] = {
        str(lv): int(len(np.unique(dd["stim_id"][dd[by] == lv]))) for lv in levels}
    r.update(dv_name=name, by=by, cues=sorted(set(dd["cue"].tolist())),
             n_ratings=int(len(dd["x"])))
    return r


# --------------------------------------------------- rater variance decomposition

def _dummies(labels):
    u = np.unique(labels)
    M = np.zeros((len(labels), len(u)))
    for j, v in enumerate(u):
        M[labels == v, j] = 1.0
    return M[:, 1:] if M.shape[1] > 1 else M[:, :0]


def _r2(y, X):
    A = np.column_stack([np.ones(len(y)), X]) if X.shape[1] else np.ones((len(y), 1))
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0


def variance_decomposition(d, name):
    """How much of the response is the condition, and how much is who is rating?

    Commonality analysis on R^2: fit condition-only, rater-only, and both, then
    split into unique and shared. Order-free. The shared piece is reported
    rather than assigned to whichever term was entered first.

    The design cell here is (cue x measured gap) -- what a rater was actually
    shown -- not (cue x nominal latency).
    """
    dd = subset(d, np.isfinite(dv(d, name)))
    y = dv(dd, name)
    cell = np.char.add(np.char.add(dd["cue"], "@"),
                       np.round(dd["x"]).astype(int).astype(str))
    cond, rater = _dummies(cell), _dummies(dd["rater_id"])
    r2_c, r2_r = _r2(y, cond), _r2(y, rater)
    r2_both = _r2(y, np.column_stack([cond, rater]))
    uniq_c = max(0.0, r2_both - r2_r)
    uniq_r = max(0.0, r2_both - r2_c)
    return {
        "dv_name": name,
        "r2_condition_only": r2_c, "r2_rater_only": r2_r, "r2_both": r2_both,
        "unique_condition": uniq_c, "unique_rater": uniq_r,
        "shared": float(r2_both - uniq_c - uniq_r),
        "residual": float(1.0 - r2_both),
        "n_ratings": int(len(y)), "n_raters": int(len(np.unique(dd["rater_id"]))),
        "n_cells": int(len(np.unique(cell))),
    }


def per_rater(d, name):
    """Each persona's own level, cue=none latency slope, and matched-gap cue effect.

    No per-persona CIs on purpose: each persona rates each stimulus exactly
    once, so an interval here would be a bootstrap over 90 clips dressed up as
    uncertainty about the persona. The spread ACROSS personas is the number
    that means something, and it is returned as `spread`.
    """
    out = {}
    for rid in sorted(set(d["rater_id"].tolist())):
        s = subset(d, d["rater_id"] == rid)
        s = subset(s, np.isfinite(dv(s, name)))
        y = dv(s, name)
        none = s["cue"] == "none"
        x = s["x"][none] / 1000.0
        slope = np.nan
        if none.sum() >= 3 and np.ptp(x) > 0:
            A = np.column_stack([np.ones(len(x)), x])
            slope = float(np.linalg.lstsq(A, y[none], rcond=None)[0][1])
        m = _mask(s, nominal=MATCHED_LATENCIES)
        cue_eff = np.nan
        if (m & none).sum() and (m & ~none).sum():
            cue_eff = float(y[m & ~none].mean() - y[m & none].mean())
        out[rid] = {"mean": float(y.mean()), "n_ratings": int(len(y)),
                    "slope_none_per_s": _f(slope),
                    "cue_effect_matched": _f(cue_eff)}
    vals = [v["mean"] for v in out.values()]
    return {"dv_name": name, "raters": out, "n_raters": len(out),
            "spread": {"min_mean": float(min(vals)), "max_mean": float(max(vals)),
                       "range": float(max(vals) - min(vals)),
                       "sd_of_rater_means": float(np.std(vals, ddof=1))}}


def _f(v):
    return float(v) if np.isfinite(v) else None


def counts(arr):
    u, c = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}
