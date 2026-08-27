"""Build the figure set from analysis/out/*results.json.

    .venv/bin/python figures/make_figures.py --results analysis/out/results.json \
        --ratings data/ratings.jsonl --stimuli data/stimuli.jsonl

Three figures, each carrying one claim:

  dissociation-*.png   what a cue moves vs what a longer wait moves, per outcome
  curves-*.png         aliveness flat against gap while `broken` climbs
  variance-*.png       six personas written to disagree, behaving as one rater

Every figure is rendered twice, -light and -dark, so a README can serve the
right one with <picture>/prefers-color-scheme instead of hoping one PNG
survives both. Every panel labels its own n. No point estimate is drawn
without its interval.

Simulated input is quarantined into figures/simulated/ and watermarked
SIMULATED on every panel.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "analysis"))
import core  # noqa: E402

# Okabe-Ito, colourblind-safe and mid-luminance so it reads on white and near-black.
CUE_COLOR = {
    "filled_pause": "#E69F00",
    "breath":       "#56B4E9",
    "backchannel":  "#009E73",
    "verbal_stall": "#D55E00",
}
NONE_COLOR = {"light": "#3F3F3F", "dark": "#C8C8C8"}
LABEL = {"none": "no cue (reference)", "filled_pause": "filled pause “uh”",
         "breath": "breath", "backchannel": "backchannel",
         "verbal_stall": "verbal stall"}
DV_LABEL = {"alive": "aliveness (1–7)", "broken": "seemed broken (1–7)",
            "wait": "would wait again (proportion)"}
PERSONA = {"llm-sonnet-patient": "patient", "llm-sonnet-impatient": "impatient",
           "llm-sonnet-linguist": "linguist", "llm-sonnet-naive": "naive",
           "llm-sonnet-skeptic": "skeptic", "llm-sonnet-voice_designer": "voice designer"}

THEME = {
    "light": dict(bg="#FFFFFF", fg="#1A1A1A", muted="#6B6B6B", grid="#E2E2E2",
                  ok="#1B7F4B", warn="#B23A15", cue="#7B4FA8", lat="#0072B2"),
    "dark":  dict(bg="#0D1117", fg="#E8E8E8", muted="#9AA0A6", grid="#2A2F36",
                  ok="#4ED88B", warn="#FF8A5C", cue="#C08CE8", lat="#56B4E9"),
}


def style(theme):
    t = THEME[theme]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["muted"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 200,
    })
    return t


def finish(fig, path, theme, simulated, caption=None):
    t = THEME[theme]
    if caption:
        fig.text(0.0, -0.02, caption, fontsize=7.5, color=t["muted"],
                 va="top", ha="left", wrap=True)
    if simulated:
        for ax in fig.axes:
            ax.text(0.5, 0.5, "SIMULATED", transform=ax.transAxes, fontsize=34,
                    color=t["warn"], alpha=0.16, ha="center", va="center",
                    rotation=28, weight="bold", zorder=10)
        fig.text(0.5, 1.015, "SIMULATED DEVELOPMENT DATA — NOT A FINDING",
                 ha="center", va="bottom", fontsize=9, weight="bold", color=t["warn"])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _lo_hi(v):
    """(est, err_low, err_high) for an errorbar. Never draws a bar without a CI."""
    if v["est"] is None or v["ci"] is None:
        return None
    e = v["est"]
    return e, e - v["ci"][0], v["ci"][1] - e


# --------------------------------------------------------- fig 1: dissociation

def fig_dissociation(res, theme, simulated, outdir):
    """The headline. Cue effect and latency effect on the same axis, per outcome."""
    t = style(theme)
    dvs = ["alive", "broken", "wait"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.1))
    # alive and broken are both 1-7, so they share an x scale and the two effects
    # are directly comparable by eye. `wait` is a proportion and gets its own.
    axes[1].sharex(axes[0])

    for ax, name in zip(axes, dvs):
        D = res["dissociation"][name]
        rows = [("a cue, same wait\n(cue − no cue at 0.8–1.6 s)",
                 D["cue_effect"], t["cue"]),
                ("+800 ms of wait\n(1.6 s − 0.8 s, all cues)",
                 D["latency_effect_1600_vs_800"], t["lat"]),
                ("+1600 ms of silence\n(1.6 s − 0 s, no cue only)",
                 D["latency_effect_none_1600_vs_0"], t["lat"])]
        ys = np.arange(len(rows))[::-1]
        ax.axvline(0, color=t["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        for y, (lab, v, col) in zip(ys, rows):
            b = _lo_hi(v)
            if b is None:
                continue
            est, el, eh = b
            ax.errorbar(est, y, xerr=[[el], [eh]], fmt="o", ms=7, lw=2.2,
                        capsize=4, color=col, zorder=3)
            hits_zero = v["ci"][0] <= 0 <= v["ci"][1]
            ax.annotate(f"{est:+.2f}", (est, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9.5,
                        color=t["muted"] if hits_zero else t["fg"],
                        weight="normal" if hits_zero else "bold")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows] if name == dvs[0] else [],
                           fontsize=8.6)
        ax.set_ylim(-0.6, len(rows) - 0.35)
        ax.set_title(DV_LABEL[name], fontsize=11.5)
        ax.set_xlabel("change in rating (95% CI)", fontsize=9.5)
        ax.grid(axis="x", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)

    d0 = res["design"]
    fig.suptitle("A cue moves aliveness. Time moves everything else.",
                 fontsize=14, weight="bold", y=1.035)
    fig.tight_layout()
    finish(fig, f"{outdir}/dissociation-{theme}.png", theme, simulated,
           f"n = {d0['n_ratings']} ratings, {d0['n_stimuli']} clips, "
           f"{d0['n_raters_llm']} LLM judges, {d0['n_raters_human']} human raters. "
           "Modality: transcript+timing, not audio. Bars are 95% percentile CIs from "
           "a cluster bootstrap over the 18 dialogue exchanges. Grey labels mark "
           "intervals containing zero. The cue contrast is restricted to 0.8/1.2/1.6 s, "
           "the only gaps where a cued and an uncued clip share the same wait.")


# -------------------------------------------------------------- fig 2: curves

def fig_curves(res, theme, simulated, outdir):
    """Aliveness flat against gap under cue=none, while `broken` climbs."""
    t = style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True)

    for ax, name, ylim in ((axes[0], "alive", (1, 7)), (axes[1], "broken", (1, 7))):
        cm = res["curves"][name]["none_by_nominal"]
        xs = sorted(float(k) for k in cm["values"])
        est = [cm["values"][_k(x)]["est"] for x in xs]
        lo = [cm["values"][_k(x)]["ci"][0] for x in xs]
        hi = [cm["values"][_k(x)]["ci"][1] for x in xs]
        col = NONE_COLOR[theme] if name == "alive" else t["warn"]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(xs, est, "-o", color=col, lw=2.4, ms=6)
        ax.set_ylim(*ylim)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}" for x in xs], fontsize=9.5)
        ax.set_xlabel("nominal gap (ms) — measured gap is exact here", fontsize=9.5)
        ax.grid(color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)

        s = res["latency_response_none"][name]["values"]["slope_per_s"]
        ax.set_title(DV_LABEL[name], fontsize=11.5)
        flat = s["ci"][0] <= 0 <= s["ci"][1]
        ax.text(0.03, 0.955 if name == "alive" else 0.955,
                f"slope {s['est']:+.2f}/s  [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"
                + ("   ← includes zero" if flat else ""),
                transform=ax.transAxes, va="top", fontsize=9.5,
                color=t["muted"] if flat else t["fg"],
                weight="normal" if flat else "bold")
        n = cm["n_per_level"]
        ax.text(0.03, 0.875, f"n = {sum(n.values())} ratings over "
                            f"{sum(cm['stimuli_per_level'].values())} clips, "
                            f"{list(n.values())[0]} ratings per point",
                transform=ax.transAxes, fontsize=8.4, color=t["muted"])

    fig.suptitle("Silence gets no deader as it gets longer — but it does get "
                 "more broken", fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    finish(fig, f"{outdir}/curves-{theme}.png", theme, simulated,
           "cue = none only, so gap is not confounded with cue. Bands are 95% "
           "percentile CIs from a cluster bootstrap over the 18 exchanges; slopes "
           "are OLS on actual_gap_ms. Note the left panel sits near the bottom of "
           "the scale (floor 1), so part of its flatness may be compression rather "
           "than indifference — the cued clips, which start higher, do fall with gap.")


def _k(x):
    return str(x) if str(x) in ("0.0",) else str(x)


# ------------------------------------------------------------ fig 3: variance

def fig_variance(res, theme, simulated, outdir):
    """Condition vs rater, and what each of the six personas actually did."""
    t = style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.05, 1.25]})

    # left: commonality bars
    ax = axes[0]
    dvs = ["alive", "broken", "wait"]
    parts = [("what they were shown", "unique_condition", t["lat"]),
             ("who was rating", "unique_rater", t["cue"]),
             ("shared", "shared", t["muted"]),
             ("unexplained", "residual", t["grid"])]
    ys = np.arange(len(dvs))[::-1]
    left = np.zeros(len(dvs))
    for lab, key, col in parts:
        w = np.array([max(0.0, res["variance"][dv][key]) for dv in dvs])
        ax.barh(ys, w, left=left, color=col, label=lab, height=0.55,
                edgecolor=THEME[theme]["bg"], lw=0.8)
        for y, x0, ww in zip(ys, left, w):
            if ww > 0.06:
                ax.text(x0 + ww / 2, y, f"{ww:.0%}", ha="center", va="center",
                        fontsize=9, color=THEME[theme]["bg"] if key != "residual"
                        else t["fg"], weight="bold")
            elif key == "unique_rater":
                # too thin to label in place, and it is the point of the panel
                ax.annotate(f"{ww:.1%}", (x0 + ww, y), textcoords="offset points",
                            xytext=(0, 15), ha="center", fontsize=8.5, color=col,
                            weight="bold")
        left = left + w
    ax.set_yticks(ys)
    ax.set_yticklabels([DV_LABEL[dv].split(" (")[0] for dv in dvs], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of variance (R² commonality)", fontsize=9.5)
    ax.set_title("Rater identity explains almost nothing", fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)

    # right: per persona, cue effect vs cue=none slope
    ax = axes[1]
    pr = res["per_rater"]["alive"]["raters"]
    names = list(pr)
    ax.axvline(0, color=t["muted"], lw=1.0, ls=(0, (4, 3)))
    for i, rid in enumerate(names):
        y = len(names) - 1 - i
        ax.plot(pr[rid]["cue_effect_matched"], y, "o", ms=8, color=t["cue"],
                label="cue effect (matched gaps)" if i == 0 else None)
        ax.plot(pr[rid]["slope_none_per_s"], y, "s", ms=7, color=t["lat"],
                label="latency slope, no cue (per s)" if i == 0 else None)
    ax.set_yticks(np.arange(len(names))[::-1])
    ax.set_yticklabels([PERSONA.get(n, n) for n in names], fontsize=10)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("aliveness points", fontsize=9.5)
    sp = res["per_rater"]["alive"]["spread"]
    ax.set_title(f"All six agree: cue effect large, slope ≈ 0\n"
                 f"(persona means span {sp['range']:.2f} of 7 points)", fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=9)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)

    v = res["variance"]["alive"]
    fig.tight_layout()
    finish(fig, f"{outdir}/variance-{theme}.png", theme, simulated,
           f"n = {v['n_ratings']} ratings, {v['n_raters']} personas, {v['n_cells']} "
           "design cells (cue x measured gap). Commonality analysis on R², so the "
           "shared component is reported rather than assigned to whichever term went "
           "in first. Right panel has no per-persona CIs on purpose: each persona "
           "rates each clip exactly once, so an interval there would be uncertainty "
           "about clips dressed up as uncertainty about the persona. All six are the "
           "same model under different instructions — this is within-model persona "
           "variance, not between-model variance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analysis/out/results.json")
    ap.add_argument("--ratings", default="data/ratings.jsonl")
    ap.add_argument("--stimuli", default="data/stimuli.jsonl")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    res = json.load(open(a.results))
    simulated = bool(res.get("simulated"))
    outdir = a.outdir or ("figures/simulated" if simulated else "figures")
    os.makedirs(outdir, exist_ok=True)
    if simulated:
        print("!! SIMULATED results: writing to", outdir, "and watermarking every panel.")
        print("!! Nothing here may be referenced from README.md.")

    made = []
    for theme in ("light", "dark"):
        fig_dissociation(res, theme, simulated, outdir)
        fig_curves(res, theme, simulated, outdir)
        fig_variance(res, theme, simulated, outdir)
        made += [f"{n}-{theme}.png" for n in ("dissociation", "curves", "variance")]
    print(f"wrote {len(made)} figures to {outdir}/")


if __name__ == "__main__":
    main()
