"""Condition matrix for the aliveness-threshold stimulus set.

Factorial cells: 6 latencies x 5 cues = 30.
We do not run the full 18 x 30 crossing (540 clips is not rateable). We run a
counterbalanced fraction:

    row (e, c)  ->  latency = LATENCIES[(e + c) % 6]

with e over 18 exchanges and c over 5 cues. That gives 90 clips and these
properties, all asserted in stimgen/check.py:

    every (latency, cue) cell               3 clips
    every cue                              18 clips, one per exchange
    every latency                          15 clips
    every exchange x cue pair               exactly 1 clip
    every exchange                          5 clips at 5 DISTINCT latencies

So cue is orthogonal to content by construction (each exchange is heard once
under every cue), and latency is rotated against content rather than nested in
it (each exchange sits at 5 of the 6 latencies, and which one it skips rotates
with e). The thing we cannot have with 90 clips is exchange x latency being
fully crossed; the rotation is the standard fix and it keeps the latency main
effect unconfounded with any single exchange.
"""

from itertools import product

from .exchanges import EXCHANGES

LATENCIES = [0, 200, 400, 800, 1200, 1600]

# The cue is whatever the robot does IN the gap before it answers. `none` is
# the baseline: dead air. Text is fixed per cue rather than varied per
# exchange, so cue identity is not confounded with wording.
CUES = {
    "none":         None,
    "filled_pause": "hm",          # nonlexical filler
    "breath":       "<breath>",      # synthesized, not spoken -- see harness
    "backchannel":  "mm hm",       # acknowledgement, "I heard you"
    "verbal_stall": "hang on",     # lexical hold: "I am on it, wait"
}
CUE_ORDER = ["none", "filled_pause", "breath", "backchannel", "verbal_stall"]

CELLS = list(product(LATENCIES, CUE_ORDER))


def condition_matrix():
    """Return the 90 planned conditions, in a fixed deterministic order."""
    rows = []
    for e, (exchange_id, prompt_text, response_text) in enumerate(EXCHANGES):
        for c, cue in enumerate(CUE_ORDER):
            latency_ms = LATENCIES[(e + c) % len(LATENCIES)]
            rows.append({
                "stim_id": f"{exchange_id}-{cue}-{latency_ms}",
                "exchange_id": exchange_id,
                "cue": cue,
                "latency_ms": latency_ms,
                "prompt_text": prompt_text,
                "response_text": response_text,
            })
    return rows


if __name__ == "__main__":
    rows = condition_matrix()
    print(f"{len(rows)} conditions over {len(EXCHANGES)} exchanges")
    for lat in LATENCIES:
        line = " ".join(
            f"{cue}:{sum(1 for r in rows if r['latency_ms'] == lat and r['cue'] == cue)}"
            for cue in CUE_ORDER)
        print(f"{lat:>5}ms  {line}")
