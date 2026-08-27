#!/usr/bin/env python3
"""CLI over the stimulus harness.

    scripts/run_agent.py selfcheck
    scripts/run_agent.py render --prompt "..." --response "..." \
        --latency-ms 800 --cue filled_pause --out stimuli/001.wav [--json timings.json]
    scripts/run_agent.py cues

The live mic->ASR->LLM->TTS loop is NOT implemented -- see harness/STATUS.md.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import cue_duration_ms, synthesize_exchange  # noqa: E402
from harness.cues import CUES  # noqa: E402
from harness.exchange import demo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="render a latency x cue grid and assert the tolerance")
    sub.add_parser("cues", help="list cues and their measured durations")

    r = sub.add_parser("render", help="render one exchange")
    r.add_argument("--prompt", required=True)
    r.add_argument("--response", required=True)
    r.add_argument("--latency-ms", type=float, required=True)
    r.add_argument("--cue", choices=CUES, default="none")
    r.add_argument("--cue-onset-ms", type=float, default=150.0)
    r.add_argument("--out", required=True)
    r.add_argument("--json", help="write the full timings/verification dict here")

    a = ap.parse_args()

    if a.cmd == "selfcheck":
        demo()
        return 0

    if a.cmd == "cues":
        for c in CUES:
            print(f"{c:<14} {cue_duration_ms(c):7.1f} ms")
        return 0

    info = synthesize_exchange(
        a.prompt, a.response, a.latency_ms, a.cue, a.out, cue_onset_ms=a.cue_onset_ms
    )
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(info, indent=2))
    v = info["verification"]
    print(
        f"{info['wav_path']}  gap {info['actual_gap_ms']:.1f}ms "
        f"(err {v.get('gap_err_ms', float('nan')):+.1f})  "
        f"cue {a.cue} onset {info['cue_onset_ms']}  verified={v['verified']}"
    )
    return 0 if v["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
