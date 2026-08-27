"""Exchange-rate estimation: how many ms of tolerated latency does a cue buy?

The headline estimator is a shared-shape horizontal-shift model. Every cue gets
the same curve shape; cues differ only by a horizontal offset d_cue, with
d_none fixed at 0. That offset IS the exchange rate in milliseconds, so it is a
fitted parameter with a confidence interval rather than something read off a plot.

CIs are cluster bootstrap over raters. Raters are the unit that generalises to
"a new listener", and rater idiosyncrasy is the dominant variance source.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import pearsonr, spearmanr

CUE_ORDER = ["none", "filled_pause", "breath", "backchannel", "verbal_stall"]
LATENCY_RANGE_MS = 1600.0  # design span; used to judge whether a shift is identified


# ---------------------------------------------------------------- data loading

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load(ratings_path, stimuli_path=None):
    """Join ratings to stimuli. Returns a dict of parallel numpy arrays.

    Latency uses measured `actual_gap_ms` when available and falls back to the
    nominal `latency_ms`. Renderer jitter is measurement error on the x-axis;
    using the nominal cell when the measurement exists throws it away.
    """
    ratings = read_jsonl(ratings_path)
    stim = {s["stim_id"]: s for s in read_jsonl(stimuli_path)} if stimuli_path else {}

    x, cue, alive, broken, wait, rid, rtype, rmod, sid, exch = ([] for _ in range(10))
    unmatched = 0
    for r in ratings:
        s = stim.get(r["stim_id"])
        if s is None:
            if stim:
                unmatched += 1
                continue
            # No stimuli file: recover the design cell from the stim_id is not safe,
            # so such rows are only usable if the rating itself carries them.
            if "latency_ms" not in r or "cue" not in r:
                unmatched += 1
                continue
            s = r
        gap = s.get("actual_gap_ms")
        lat = float(gap) if gap is not None else float(s["latency_ms"])
        x.append(lat)
        cue.append(s["cue"])
        exch.append(s.get("exchange_id", s.get("stim_id")))
        sid.append(r["stim_id"])
        alive.append(float(r["aliveness_1_7"]))
        broken.append(float(r.get("broken_1_7", np.nan)))
        w = r.get("would_wait_again_bool")
        wait.append(np.nan if w is None else float(bool(w)))
        rid.append(r["rater_id"])
        rtype.append(r.get("rater_type", "unknown"))
        rmod.append(r.get("rater_modality", "unknown"))

    d = dict(x=np.asarray(x, float), cue=np.asarray(cue), alive=np.asarray(alive, float),
             broken=np.asarray(broken, float), wait=np.asarray(wait, float),
             rater_id=np.asarray(rid), rater_type=np.asarray(rtype),
             rater_modality=np.asarray(rmod), stim_id=np.asarray(sid),
             exchange_id=np.asarray(exch), n_unmatched=unmatched,
             stimuli={k: v for k, v in stim.items()})
    return d


def subset(d, mask):
    out = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    return out


# ------------------------------------------------------------------ the models

def _sig_down(x, x50, scale):
    """Monotone decreasing logistic in [0,1]. 0.5 at x == x50."""
    z = np.clip((x - x50) / scale, -600, 600)
    return 1.0 / (1.0 + np.exp(z))


@dataclass
class ShiftFit:
    cues: list                      # cue levels, reference first
    shifts_ms: dict                 # cue -> ms bought vs reference (reference = 0.0)
    x50_ms: float                   # reference-cue midpoint
    scale_ms: float                 # curve steepness
    lo: float = float("nan")        # asymptotes (continuous DV only)
    hi: float = float("nan")
    ok: bool = True
    n_obs: int = 0
    n_raters: int = 0
    dv: str = ""
    note: str = ""


def _pack_bounds(k_shift, continuous, x_lo, x_hi):
    span = max(200.0, x_hi - x_lo)
    if continuous:
        lo = [1.0, 1.0, x_lo - 2 * span, math.log(20.0)] + [-2 * span] * k_shift
        hi = [7.0, 7.0, x_hi + 2 * span, math.log(5 * span)] + [2 * span] * k_shift
    else:
        lo = [x_lo - 2 * span, math.log(20.0)] + [-2 * span] * k_shift
        hi = [x_hi + 2 * span, math.log(5 * span)] + [2 * span] * k_shift
    return np.asarray(lo), np.asarray(hi)


def fit_shift_continuous(x, cue, y, cues=None):
    """Shared-shape shift model for a continuous DV (aliveness / reversed broken).

    mu = lo + (hi - lo) * sigmoid_down(x ; x50 + d_cue, scale)
    """
    cues = cues or _ordered_cues(cue)
    ref, others = cues[0], cues[1:]
    idx = {c: (cue == c) for c in cues}
    x_lo, x_hi = float(np.min(x)), float(np.max(x))

    def mu_of(p):
        lo, hi, x50, logs = p[0], p[1], p[2], p[3]
        shifts = p[4:]
        eff = np.full_like(x, x50)
        for j, c in enumerate(others):
            eff[idx[c]] = x50 + shifts[j]
        return lo + (hi - lo) * _sig_down(x, eff, math.exp(logs))

    p0 = np.asarray([float(np.percentile(y, 10)), float(np.percentile(y, 90)),
                     0.5 * (x_lo + x_hi), math.log(max(60.0, (x_hi - x_lo) / 5))]
                    + [0.0] * len(others))
    blo, bhi = _pack_bounds(len(others), True, x_lo, x_hi)
    p0 = np.minimum(np.maximum(p0, blo + 1e-6), bhi - 1e-6)
    try:
        res = least_squares(lambda p: mu_of(p) - y, p0, bounds=(blo, bhi),
                            max_nfev=8000, x_scale="jac")
        p, ok = res.x, bool(res.success or res.status > 0)
    except Exception:
        p, ok = p0, False
    shifts = {ref: 0.0}
    shifts.update({c: float(p[4 + j]) for j, c in enumerate(others)})
    return ShiftFit(cues=cues, shifts_ms=shifts, x50_ms=float(p[2]),
                    scale_ms=float(math.exp(p[3])), lo=float(p[0]), hi=float(p[1]),
                    ok=ok, n_obs=len(y), n_raters=0, dv="continuous")


def fit_shift_binary(x, cue, y, cues=None):
    """Shared-shape shift model for a 0/1 DV (would_wait_again).

    p = sigmoid_down(x ; x50 + d_cue, scale). x50 is the classic 50% threshold.
    """
    cues = cues or _ordered_cues(cue)
    ref, others = cues[0], cues[1:]
    idx = {c: (cue == c) for c in cues}
    x_lo, x_hi = float(np.min(x)), float(np.max(x))

    def negll(p):
        x50, logs, shifts = p[0], p[1], p[2:]
        eff = np.full_like(x, x50)
        for j, c in enumerate(others):
            eff[idx[c]] = x50 + shifts[j]
        pr = np.clip(_sig_down(x, eff, math.exp(logs)), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    p0 = np.asarray([0.5 * (x_lo + x_hi), math.log(max(60.0, (x_hi - x_lo) / 5))]
                    + [0.0] * len(others))
    blo, bhi = _pack_bounds(len(others), False, x_lo, x_hi)
    p0 = np.minimum(np.maximum(p0, blo + 1e-6), bhi - 1e-6)
    try:
        res = minimize(negll, p0, method="L-BFGS-B",
                       bounds=list(zip(blo, bhi)), options={"maxiter": 4000})
        p, ok = res.x, bool(res.success)
    except Exception:
        p, ok = p0, False
    shifts = {ref: 0.0}
    shifts.update({c: float(p[2 + j]) for j, c in enumerate(others)})
    return ShiftFit(cues=cues, shifts_ms=shifts, x50_ms=float(p[0]),
                    scale_ms=float(math.exp(p[1])), ok=ok, n_obs=len(y),
                    n_raters=0, dv="binary")


def fit_free_per_cue(x, cue, y, cues=None):
    """Robustness check: fit each cue its own curve, no shared shape.

    If the per-cue x50 differences disagree badly with the shift model, the
    shared-shape assumption is wrong and the exchange rate is not a pure
    horizontal shift. That has to be reported, not smoothed over.
    """
    cues = cues or _ordered_cues(cue)
    out = {}
    for c in cues:
        m = cue == c
        if m.sum() < 12 or len(np.unique(x[m])) < 3:
            out[c] = None
            continue
        f = fit_shift_continuous(x[m], cue[m], y[m], cues=[c])
        out[c] = {"x50_ms": f.x50_ms, "scale_ms": f.scale_ms, "lo": f.lo,
                  "hi": f.hi, "ok": f.ok, "n_obs": int(m.sum())}
    ref = cues[0]
    if out.get(ref):
        for c in cues:
            if out.get(c):
                out[c]["delta_x50_vs_ref_ms"] = out[c]["x50_ms"] - out[ref]["x50_ms"]
    return out


def _ordered_cues(cue):
    present = set(np.unique(cue).tolist())
    ordered = [c for c in CUE_ORDER if c in present]
    ordered += sorted(present - set(ordered))
    if "none" in ordered:  # reference must come first
        ordered.remove("none")
        ordered.insert(0, "none")
    return ordered


# --------------------------------------------------------------- the bootstrap

def bootstrap_shifts(d, dv="alive", n_boot=2000, seed=7, cluster="rater_id"):
    """Cluster bootstrap over raters. Returns point estimate + percentile CIs.

    Resampling raters (not rows) is what makes the CI answer "how much would
    this move with a different set of listeners", which is the question a
    latency budget actually depends on.
    """
    rng = np.random.default_rng(seed)
    y_all, binary = _dv(d, dv)
    keep = np.isfinite(y_all) & np.isfinite(d["x"])
    x, cue, y = d["x"][keep], d["cue"][keep], y_all[keep]
    units = d[cluster][keep]
    cues = _ordered_cues(cue)
    fit = (fit_shift_binary if binary else fit_shift_continuous)

    point = fit(x, cue, y, cues=cues)
    point.n_raters = int(len(np.unique(d["rater_id"][keep])))

    uu = np.unique(units)
    where = {u: np.flatnonzero(units == u) for u in uu}
    draws, fails = {c: [] for c in cues}, 0
    x50s, scales = [], []
    for _ in range(n_boot):
        pick = rng.choice(uu, size=len(uu), replace=True)
        ix = np.concatenate([where[u] for u in pick])
        bc = cue[ix]
        if len(np.unique(bc)) < len(cues):
            fails += 1
            continue
        f = fit(x[ix], bc, y[ix], cues=cues)
        if not f.ok or not np.isfinite(f.scale_ms):
            fails += 1
            continue
        for c in cues:
            draws[c].append(f.shifts_ms[c])
        x50s.append(f.x50_ms)
        scales.append(f.scale_ms)

    span = float(np.max(x) - np.min(x)) or LATENCY_RANGE_MS
    res = {}
    for c in cues:
        arr = np.asarray(draws[c], float)
        if arr.size < 50:
            res[c] = {"shift_ms": point.shifts_ms[c], "ci_lo": None, "ci_hi": None,
                      "n_boot_ok": int(arr.size), "identified": False,
                      "note": "too few converged bootstrap fits for an interval"}
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        width = float(hi - lo)
        res[c] = {"shift_ms": point.shifts_ms[c], "ci_lo": float(lo), "ci_hi": float(hi),
                  "boot_median_ms": float(np.median(arr)), "ci_width_ms": width,
                  "n_boot_ok": int(arr.size),
                  # A shift is only usable if its interval is narrower than the design
                  # span. Wider than that means the data cannot locate the curve.
                  "identified": bool(width < span),
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return {
        "dv": dv, "cues": cues, "reference": cues[0],
        "x50_ms": point.x50_ms, "scale_ms": point.scale_ms,
        "x50_ci": _pct_ci(x50s), "scale_ci": _pct_ci(scales),
        "lo": point.lo, "hi": point.hi, "fit_ok": point.ok,
        "n_obs": int(len(y)), "n_raters": point.n_raters,
        "n_stimuli": int(len(np.unique(d["stim_id"][keep]))),
        "n_boot_requested": n_boot, "n_boot_failed": fails,
        "design_span_ms": span,
        "shifts": res,
    }


def _pct_ci(vals):
    a = np.asarray(vals, float)
    if a.size < 50:
        return None
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def _dv(d, dv):
    if dv == "alive":
        return d["alive"], False
    if dv == "broken_rev":
        return 8.0 - d["broken"], False
    if dv == "wait":
        return d["wait"], True
    raise ValueError(f"unknown dv {dv}")


# --------------------------------------------------- rater variance decomposition

def _dummies(labels):
    u = np.unique(labels)
    M = np.zeros((len(labels), len(u)))
    for j, v in enumerate(u):
        M[labels == v, j] = 1.0
    return M[:, 1:] if M.shape[1] > 1 else M[:, :0]


def _r2(y, X):
    A = np.column_stack([np.ones(len(y)), X]) if X.shape[1] else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0


def variance_decomposition(d, dv="alive"):
    """How much of the response is the condition, and how much is who is rating?

    Commonality analysis on R^2: fit condition-only, rater-only, and both, then
    split into unique and shared. Order-free, no mixed-model dependency. The
    "shared" piece is real and is reported rather than assigned to whichever
    term happens to be entered first.
    """
    y_all, _ = _dv(d, dv)
    keep = np.isfinite(y_all)
    y = y_all[keep]
    cell = np.char.add(np.char.add(d["cue"][keep], "@"),
                       d["x"][keep].astype(int).astype(str))
    # condition = the design cell (cue x nominal latency), not the jittered value
    cond = _dummies(cell)
    rater = _dummies(d["rater_id"][keep])
    r2_c, r2_r = _r2(y, cond), _r2(y, rater)
    r2_both = _r2(y, np.column_stack([cond, rater]) if cond.shape[1] and rater.shape[1]
                  else (cond if cond.shape[1] else rater))
    uniq_c = max(0.0, r2_both - r2_r)
    uniq_r = max(0.0, r2_both - r2_c)
    return {
        "dv": dv,
        "r2_condition_only": r2_c, "r2_rater_only": r2_r, "r2_both": r2_both,
        "unique_condition": uniq_c, "unique_rater": uniq_r,
        "shared": float(r2_both - uniq_c - uniq_r),
        "residual": float(1.0 - r2_both),
        "n_obs": int(len(y)), "n_raters": int(len(np.unique(d["rater_id"][keep]))),
        "n_cells": int(len(np.unique(cell))),
        "interpretation_note": ("unique_rater is idiosyncratic rater level/spread; "
                               "unique_condition is the designed manipulation; "
                               "shared reflects unbalanced coverage across raters"),
    }


# ------------------------------------------------------- llm vs human agreement

def llm_human_agreement(d, dv="alive", n_boot=2000, seed=13, match_modality=True):
    """Do LLM judges perceive social timing the way people do?

    Two questions, both reported:
      1. Do they rank the same stimuli the same way? (correlation of per-stimulus means)
      2. Do they produce the same exchange rate? (refit the shift model per group)

    (2) is the one that matters here. A judge can correlate well on average
    aliveness and still be blind to the cue manipulation, which is exactly the
    failure mode that would invalidate an LLM-rated latency budget.
    """
    y_all, binary = _dv(d, dv)
    keep = np.isfinite(y_all)
    h = keep & (d["rater_type"] == "human")
    l = keep & (d["rater_type"] == "llm")
    out = {"dv": dv,
           "n_human_ratings": int(h.sum()), "n_llm_ratings": int(l.sum()),
           "n_human_raters": int(len(np.unique(d["rater_id"][h]))) if h.any() else 0,
           "n_llm_raters": int(len(np.unique(d["rater_id"][l]))) if l.any() else 0,
           "human_modalities": _counts(d["rater_modality"][h]),
           "llm_modalities": _counts(d["rater_modality"][l])}
    if not h.any() or not l.any():
        out["available"] = False
        out["note"] = "need both human and llm ratings; one group is empty"
        return out
    out["available"] = True

    shared_mod = set(out["human_modalities"]) & set(out["llm_modalities"])
    out["shared_modalities"] = sorted(shared_mod)
    out["modality_confounded"] = not bool(shared_mod)
    if out["modality_confounded"]:
        out["note"] = ("no shared rater_modality between groups: any divergence is "
                       "confounded with what the two groups were actually given")
    if match_modality and shared_mod:
        mm = np.isin(d["rater_modality"], list(shared_mod))
        h, l = h & mm, l & mm
        out["matched_on_modality"] = True
    else:
        out["matched_on_modality"] = False

    # (1) per-stimulus agreement
    hm, lm = _stim_means(d, y_all, h), _stim_means(d, y_all, l)
    common = sorted(set(hm) & set(lm))
    out["n_stimuli_compared"] = len(common)
    if len(common) >= 4:
        a = np.asarray([hm[s] for s in common])
        b = np.asarray([lm[s] for s in common])
        out["pearson_r"] = float(pearsonr(a, b)[0])
        out["spearman_rho"] = float(spearmanr(a, b)[0])
        out["mean_abs_diff"] = float(np.mean(np.abs(a - b)))
        out["llm_minus_human_mean"] = float(np.mean(b - a))
        out["human_sd_across_stimuli"] = float(np.std(a, ddof=1))
        out["llm_sd_across_stimuli"] = float(np.std(b, ddof=1))
        rng = np.random.default_rng(seed)
        rs = []
        for _ in range(min(n_boot, 2000)):
            ix = rng.integers(0, len(common), len(common))
            if np.std(a[ix]) == 0 or np.std(b[ix]) == 0:
                continue
            rs.append(pearsonr(a[ix], b[ix])[0])
        out["pearson_ci"] = _pct_ci(rs)
    else:
        out["note_stimuli"] = "fewer than 4 shared stimuli; correlation not computed"

    # (2) same exchange rate?
    out["exchange_rate_by_group"] = {}
    for name, mask in (("human", h), ("llm", l)):
        sub = subset(d, mask)
        if len(np.unique(sub["cue"])) < 2 or len(np.unique(sub["rater_id"])) < 3:
            out["exchange_rate_by_group"][name] = {"available": False,
                "note": "need >=2 cues and >=3 raters in this group"}
            continue
        out["exchange_rate_by_group"][name] = bootstrap_shifts(
            sub, dv=dv, n_boot=min(n_boot, 1000), seed=seed)

    hg = out["exchange_rate_by_group"].get("human", {})
    lg = out["exchange_rate_by_group"].get("llm", {})
    if hg.get("shifts") and lg.get("shifts"):
        diffs = {}
        for c in hg["shifts"]:
            if c in lg["shifts"] and c != hg.get("reference"):
                hv, lv = hg["shifts"][c]["shift_ms"], lg["shifts"][c]["shift_ms"]
                hci, lci = hg["shifts"][c], lg["shifts"][c]
                overlap = None
                if None not in (hci.get("ci_lo"), hci.get("ci_hi"),
                                lci.get("ci_lo"), lci.get("ci_hi")):
                    overlap = not (hci["ci_hi"] < lci["ci_lo"] or lci["ci_hi"] < hci["ci_lo"])
                diffs[c] = {"human_ms": hv, "llm_ms": lv, "llm_minus_human_ms": lv - hv,
                            "ratio_llm_over_human": (lv / hv) if abs(hv) > 1e-6 else None,
                            "ci_overlap": overlap}
        out["exchange_rate_divergence"] = diffs
        out["latency_sensitivity"] = {
            "human_scale_ms": hg.get("scale_ms"), "llm_scale_ms": lg.get("scale_ms"),
            "human_x50_ms": hg.get("x50_ms"), "llm_x50_ms": lg.get("x50_ms"),
            "note": "larger scale_ms = flatter curve = less latency-sensitive",
        }
    return out


def _stim_means(d, y, mask):
    out = {}
    sid = d["stim_id"][mask]
    yy = y[mask]
    for s in np.unique(sid):
        out[s] = float(np.mean(yy[sid == s]))
    return out


def _counts(arr):
    u, c = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}


# ------------------------------------------------------------- cost vs benefit

def cost_benefit(d, shifts):
    """Rank cues by ms bought, against what the cue itself costs in ms.

    `actual_gap_ms` spans prompt offset -> response onset and the cue happens
    inside that window, so the cue's duration is already inside the measured
    latency. net_ms is therefore the benefit over and above the reference curve;
    ms_per_ms is how much tolerance each ms of cue audio returns.
    """
    dur = {}
    for s in d.get("stimuli", {}).values():
        v = s.get("cue_dur_ms")
        if v is not None:
            dur.setdefault(s["cue"], []).append(float(v))
    rows = []
    for c, r in shifts.get("shifts", {}).items():
        if c == shifts.get("reference"):
            continue
        ds = dur.get(c, [])
        mean_dur = float(np.mean(ds)) if ds else None
        rows.append({
            "cue": c, "shift_ms": r["shift_ms"],
            "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
            "identified": r.get("identified"), "excludes_zero": r.get("excludes_zero"),
            "cue_dur_ms": mean_dur, "n_stim_with_duration": len(ds),
            "ms_bought_per_ms_of_cue": (r["shift_ms"] / mean_dur)
                if mean_dur else None,
        })
    rows.sort(key=lambda r: r["shift_ms"], reverse=True)
    if not any(r["cue_dur_ms"] for r in rows):
        return {"rows": rows, "cost_side_available": False,
                "note": "no cue_dur_ms in stimuli.jsonl; benefit reported without cost"}
    return {"rows": rows, "cost_side_available": True}
