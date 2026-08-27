"""Generate SIMULATED_* fixtures so the analysis can be built before real data lands.

The fixture carries a KNOWN exchange rate (TRUE_SHIFTS_MS). That is the point:
test_analysis.py checks the estimator recovers it. Nothing here is a finding.
Nothing computed from these files may appear in README.md or figures/.
"""
import argparse, json, math, random

LATENCIES = [0, 200, 400, 800, 1200, 1600]
CUES = ["none", "filled_pause", "breath", "backchannel", "verbal_stall"]

# Ground truth for recovery testing. Arbitrary values, not a claim about the world.
TRUE_SHIFTS_MS = {"none": 0.0, "filled_pause": 350.0, "breath": 120.0,
                  "backchannel": 260.0, "verbal_stall": 480.0}
TRUE_CUE_DUR_MS = {"none": 0.0, "filled_pause": 300.0, "breath": 220.0,
                   "backchannel": 380.0, "verbal_stall": 520.0}
TRUE_X50 = 700.0     # human midpoint latency, ms
TRUE_SCALE = 300.0   # curve steepness, ms
TRUE_LO, TRUE_HI = 2.0, 6.2   # aliveness asymptotes on the 1-7 scale


def _curve(x, x50, scale, lo, hi):
    return lo + (hi - lo) / (1.0 + math.exp((x - x50) / scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchanges", type=int, default=8)
    ap.add_argument("--humans", type=int, default=24)
    ap.add_argument("--llms", type=int, default=6)
    ap.add_argument("--llm-shift-attenuation", type=float, default=0.25,
                    help="fraction of the true cue shift LLM raters express (1.0 = same as humans)")
    ap.add_argument("--llm-slope-attenuation", type=float, default=0.45,
                    help="fraction of the human latency sensitivity LLM raters express")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    stimuli = []
    for e in range(a.exchanges):
        for cue in CUES:
            for lat in LATENCIES:
                jitter = rng.gauss(0, 18)  # renderer never hits the nominal value exactly
                actual = max(0.0, lat + jitter)
                dur = TRUE_CUE_DUR_MS[cue]
                stimuli.append({
                    "stim_id": f"sim-e{e:02d}-{cue}-{lat}",
                    "latency_ms": lat,
                    "cue": cue,
                    "exchange_id": f"sim-e{e:02d}",
                    "prompt_text": f"[simulated prompt {e}]",
                    "response_text": f"[simulated response {e}]",
                    "wav_path": f"SIMULATED/none-e{e:02d}-{cue}-{lat}.wav",
                    "actual_gap_ms": round(actual, 1),
                    "cue_onset_ms": None if cue == "none" else round(rng.uniform(40, 140), 1),
                    "cue_dur_ms": None if cue == "none" else round(dur + rng.gauss(0, 25), 1),
                })

    raters = []
    for i in range(a.humans):
        raters.append({"rater_id": f"sim-h{i:02d}", "rater_type": "human",
                       "rater_modality": "audio",
                       "b0": rng.gauss(0, 0.55), "gain": math.exp(rng.gauss(0, 0.22)),
                       "shift_gain": 1.0, "slope_gain": 1.0, "noise": 0.62})
    for i in range(a.llms):
        raters.append({"rater_id": f"sim-l{i:02d}", "rater_type": "llm",
                       "rater_modality": "audio",
                       "b0": rng.gauss(0, 0.30), "gain": math.exp(rng.gauss(0, 0.10)),
                       "shift_gain": a.llm_shift_attenuation,
                       "slope_gain": a.llm_slope_attenuation, "noise": 0.35})

    ratings = []
    for s in stimuli:
        x = s["actual_gap_ms"]
        for r in raters:
            shift = TRUE_SHIFTS_MS[s["cue"]] * r["shift_gain"]
            scale = TRUE_SCALE / max(1e-6, r["slope_gain"])  # flatter = less latency-sensitive
            mu = _curve(x, TRUE_X50 + shift, scale, TRUE_LO, TRUE_HI)
            mu = 4.0 + (mu - 4.0) * r["gain"] + r["b0"]
            alive = min(7, max(1, int(round(mu + rng.gauss(0, r["noise"])))))
            broken_mu = 8.0 - mu
            broken = min(7, max(1, int(round(broken_mu + rng.gauss(0, r["noise"])))))
            p_wait = 1.0 / (1.0 + math.exp((x - (TRUE_X50 + shift) - 150.0) / scale))
            ratings.append({
                "stim_id": s["stim_id"], "rater_id": r["rater_id"],
                "rater_type": r["rater_type"], "rater_modality": r["rater_modality"],
                "aliveness_1_7": alive, "broken_1_7": broken,
                "would_wait_again_bool": rng.random() < p_wait,
                "ts": "2026-08-26T00:00:00Z",
            })

    for name, rows in (("SIMULATED_stimuli.jsonl", stimuli), ("SIMULATED_ratings.jsonl", ratings)):
        with open(f"{a.outdir}/{name}", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"wrote {len(stimuli)} SIMULATED stimuli, {len(ratings)} SIMULATED ratings "
          f"({a.humans} human + {a.llms} llm raters)")
    print("ground-truth shifts (ms):", TRUE_SHIFTS_MS)


if __name__ == "__main__":
    main()
