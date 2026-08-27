"""Build the figure set from analysis/out/*results.json.

    .venv/bin/python figures/make_figures.py --results analysis/out/results.json \
        --ratings data/ratings.jsonl --stimuli data/stimuli.jsonl

Every figure is rendered twice, -light and -dark, so a README can serve the right
one with <picture>/prefers-color-scheme instead of hoping one PNG survives both.

Simulated input is quarantined into figures/simulated/ and watermarked SIMULATED.
"""
import argparse, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import core  # noqa: E402

# Okabe-Ito, colourblind-safe and mid-luminance so it reads on white and on near-black.
CUE_COLOR = {
    "filled_pause": "#E69F00",
    "breath":       "#56B4E9",
    "backchannel":  "#009E73",
    "verbal_stall": "#D55E00",
}
NONE_COLOR = {"light": "#3F3F3F", "dark": "#C8C8C8"}
LABEL = {"none": "no cue (reference)", "filled_pause": "filled pause “uh”",
         "breath": "breath", "backchannel": "backchannel", "verbal_stall": "verbal stall"}

THEME = {
    "light": dict(bg="#FFFFFF", fg="#1A1A1A", muted="#6B6B6B", grid="#E2E2E2",
                  ok="#1B7F4B", warn="#B23A15"),
    "dark":  dict(bg="#0D1117", fg="#E8E8E8", muted="#9AA0A6", grid="#2A2F36",
                  ok="#4ED88B", warn="#FF8A5C"),
}


def color(cue, theme):
    return CUE_COLOR.get(cue, NONE_COLOR[theme])


def style(theme):
    t = THEME[theme]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
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
        fig.text(0.0, -0.035, caption, fontsize=7.5, color=t["muted"],
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


def curve_y(x, x50, scale, lo, hi):
    z = np.clip((np.asarray(x, float) - x50) / scale, -600, 600)
    return lo + (hi - lo) / (1.0 + np.exp(z))


# ------------------------------------------------------------------- fig 1

def fig_curves(res, d, theme, simulated, outdir):
    t = style(theme)
    er = res["exchange_rate_aliveness"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    xs = np.linspace(0, max(1750.0, float(np.max(d["x"])) * 1.05), 400)

    for cue in er["cues"]:
        c = color(cue, theme)
        sh = er["shifts"][cue]["shift_ms"]
        ax.plot(xs, curve_y(xs, er["x50_ms"] + sh, er["scale_ms"], er["lo"], er["hi"]),
                color=c, lw=2.4 if cue == "none" else 1.9,
                ls="--" if cue == "none" else "-", zorder=3,
                label=f"{LABEL.get(cue, cue)}" + ("" if cue == "none" else f"  +{sh:.0f} ms"))
        # observed cell means, SE across raters within cell
        m = d["cue"] == cue
        cells = np.round(d["x"][m] / 50.0) * 50.0
        for u in np.unique(cells):
            sel = cells == u
            v = d["alive"][m][sel]
            if v.size < 2:
                continue
            ax.errorbar(u, v.mean(), yerr=v.std(ddof=1) / math.sqrt(v.size),
                        fmt="o", ms=4.2, color=c, alpha=0.85, lw=1.1,
                        capsize=2.5, zorder=4)

    # the money annotation: the horizontal gap between reference and best cue
    best = max(((c, s["shift_ms"]) for c, s in er["shifts"].items() if c != er["reference"]),
               key=lambda kv: kv[1], default=None)
    if best and best[1] > 20:
        ymid = (er["lo"] + er["hi"]) / 2.0
        ax.annotate("", xy=(er["x50_ms"] + best[1], ymid), xytext=(er["x50_ms"], ymid),
                    arrowprops=dict(arrowstyle="<->", color=t["fg"], lw=1.5), zorder=6)
        ax.text(er["x50_ms"] + best[1] / 2, ymid + 0.16,
                f"{best[1]:.0f} ms bought", ha="center", fontsize=10.5,
                weight="bold", color=t["fg"], zorder=6)
        ax.axvline(er["x50_ms"], color=t["muted"], lw=0.9, ls=":", zorder=1)

    ax.set_xlabel("response latency (ms, measured gap)")
    ax.set_ylabel("aliveness (1–7)")
    ax.set_title("A cue shifts the whole curve to the right")
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9.5)
    n = res["design"]
    finish(fig, f"{outdir}/curves-{theme}.png", theme, simulated,
           f"n = {n['n_ratings']} ratings, {n['n_raters']} raters, {n['n_stimuli']} stimuli. "
           f"Points are cell means ± 1 SE. Curves are the shared-shape shift model.")


# ------------------------------------------------------------------- fig 2

def fig_forest(res, theme, simulated, outdir):
    t = style(theme)
    er = res["exchange_rate_aliveness"]
    rows = [(c, s) for c, s in er["shifts"].items() if c != er["reference"]]
    rows.sort(key=lambda kv: kv[1]["shift_ms"])
    fig, ax = plt.subplots(figsize=(7.4, 0.72 * len(rows) + 2.3))

    for i, (cue, s) in enumerate(rows):
        c = color(cue, theme)
        solid = s.get("identified") and s.get("excludes_zero")
        if s.get("ci_lo") is not None:
            ax.plot([s["ci_lo"], s["ci_hi"]], [i, i], color=c, lw=2.6,
                    alpha=1.0 if solid else 0.42, solid_capstyle="butt", zorder=3)
            for e in (s["ci_lo"], s["ci_hi"]):
                ax.plot([e, e], [i - 0.13, i + 0.13], color=c, lw=2.0,
                        alpha=1.0 if solid else 0.42, zorder=3)
        ax.plot([s["shift_ms"]], [i], "o", ms=9, color=c,
                mec=t["bg"], mew=1.4, zorder=4, alpha=1.0 if solid else 0.5)
        lab = (f"{s['shift_ms']:.0f}" if s.get("ci_lo") is None else
               f"{s['shift_ms']:.0f}  [{s['ci_lo']:.0f}, {s['ci_hi']:.0f}]")
        if not solid:
            lab += "  n.s." if s.get("identified") else "  unidentified"
        ax.text(1.02, i, lab, transform=ax.get_yaxis_transform(), va="center",
                fontsize=9.5, color=t["fg"] if solid else t["muted"], family="monospace")

    ax.axvline(0, color=t["muted"], lw=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(c, c) for c, _ in rows])
    ax.set_xlabel("milliseconds of latency bought, vs no cue  (95% CI)")
    ax.set_title("The exchange rate")
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    n = res["design"]
    finish(fig, f"{outdir}/exchange-rate-{theme}.png", theme, simulated,
           f"n = {n['n_ratings']} ratings from {n['n_raters']} raters. "
           f"Cluster bootstrap over raters, {er['n_boot_requested']} resamples. "
           f"Faded = interval includes zero or is wider than the {er['design_span_ms']:.0f} ms tested range.")


# ------------------------------------------------------------------- fig 3

def fig_variance(res, theme, simulated, outdir):
    t = style(theme)
    v = res["variance_aliveness"]
    parts = [("the manipulation\n(cue × latency)", v["unique_condition"], "#009E73"),
             ("who is rating\n(rater idiosyncrasy)", v["unique_rater"], "#E69F00"),
             ("shared / confounded", max(0.0, v["shared"]), "#56B4E9"),
             ("unexplained", v["residual"], THEME[theme]["muted"])]
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    left = 0.0
    for name, frac, c in parts:
        ax.barh([0], [frac], left=left, color=c, height=0.5, edgecolor=t["bg"], lw=1.5)
        if frac > 0.045:
            ax.text(left + frac / 2, 0, f"{frac*100:.0f}%", ha="center", va="center",
                    fontsize=11, weight="bold",
                    color="#FFFFFF" if c != THEME[theme]["muted"] else t["bg"])
        left += frac
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.set_title("Where the variance in aliveness ratings lives")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    ax.legend(handles, [f"{n.replace(chr(10), ' ')} — {f*100:.0f}%" for n, f, _ in parts],
              loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=9)
    finish(fig, f"{outdir}/variance-{theme}.png", theme, simulated,
           f"Commonality analysis on R². n = {v['n_obs']} ratings, "
           f"{v['n_raters']} raters, {v['n_cells']} design cells.")


# ------------------------------------------------------------------- fig 4

def fig_llm_human(res, theme, simulated, outdir):
    t = style(theme)
    ag = res.get("llm_vs_human_aliveness", {})
    if not ag.get("available"):
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    a = ag.get("_human_means"); b = ag.get("_llm_means")
    if a and b:
        ax1.scatter(a, b, s=16, alpha=0.6, color="#56B4E9", edgecolor="none")
    lim = [1, 7]
    ax1.plot(lim, lim, ls="--", lw=1.2, color=t["muted"], zorder=1)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("human mean aliveness, per stimulus")
    ax1.set_ylabel("LLM mean aliveness, per stimulus")
    ci = ag.get("pearson_ci")
    sub = f"r = {ag['pearson_r']:.2f}" if "pearson_r" in ag else "r n/a"
    if ci:
        sub += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    sub += f", {ag.get('n_stimuli_compared', 0)} stimuli"
    ax1.set_title(f"Do they rank stimuli alike?\n{sub}", fontsize=11.5)
    ax1.grid(color=t["grid"], lw=0.8); ax1.set_axisbelow(True)

    div = ag.get("exchange_rate_divergence", {})
    cues = sorted(div, key=lambda c: div[c]["human_ms"])
    yy = np.arange(len(cues))
    hg = ag["exchange_rate_by_group"]["human"]["shifts"]
    lg = ag["exchange_rate_by_group"]["llm"]["shifts"]
    for i, c in enumerate(cues):
        for grp, src, off, mk, col in (("human", hg, 0.16, "o", "#009E73"),
                                       ("LLM", lg, -0.16, "s", "#CC79A7")):
            s = src[c]
            if s.get("ci_lo") is not None:
                ax2.plot([s["ci_lo"], s["ci_hi"]], [i + off] * 2, color=col, lw=2.4, zorder=3)
            ax2.plot([s["shift_ms"]], [i + off], mk, ms=7.5, color=col,
                     mec=t["bg"], mew=1.2, zorder=4,
                     label=grp if i == 0 else None)
    ax2.axvline(0, color=t["muted"], lw=1.2)
    ax2.set_yticks(yy); ax2.set_yticklabels([LABEL.get(c, c) for c in cues])
    ax2.set_xlabel("milliseconds bought (95% CI)")
    ax2.set_title("Do they price the cues alike?", fontsize=11.5)
    ax2.set_ylim(-0.55, len(cues) - 0.45)
    ax2.grid(axis="x", color=t["grid"], lw=0.8); ax2.set_axisbelow(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9.5)

    finish(fig, f"{outdir}/llm-vs-human-{theme}.png", theme, simulated,
           f"{ag['n_human_raters']} human raters ({ag['n_human_ratings']} ratings) vs "
           f"{ag['n_llm_raters']} LLM raters ({ag['n_llm_ratings']} ratings). "
           + ("Modality-matched." if ag.get("matched_on_modality")
              else "NOT modality-matched — divergence is confounded."))
    return True


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

    d = core.load(a.ratings, a.stimuli if os.path.exists(a.stimuli) else None)

    # attach per-stimulus group means for the agreement scatter
    ag = res.get("llm_vs_human_aliveness", {})
    if ag.get("available"):
        h = d["rater_type"] == "human"
        l = d["rater_type"] == "llm"
        hm, lm = core._stim_means(d, d["alive"], h), core._stim_means(d, d["alive"], l)
        common = sorted(set(hm) & set(lm))
        ag["_human_means"] = [hm[s] for s in common]
        ag["_llm_means"] = [lm[s] for s in common]

    made = []
    for theme in ("light", "dark"):
        fig_curves(res, d, theme, simulated, outdir); made.append(f"curves-{theme}.png")
        fig_forest(res, theme, simulated, outdir); made.append(f"exchange-rate-{theme}.png")
        fig_variance(res, theme, simulated, outdir); made.append(f"variance-{theme}.png")
        if fig_llm_human(res, theme, simulated, outdir):
            made.append(f"llm-vs-human-{theme}.png")
    print(f"wrote {len(made)} figures to {outdir}/")


if __name__ == "__main__":
    main()
