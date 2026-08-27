#!/usr/bin/env python3
"""CLI over the live voice loop (audio in -> ASR -> LM -> TTS -> audio out).

    scripts/run_live.py selfcheck                      # 2 turns, assert every timer
    scripts/run_live.py batch --n 20 --out live/results/run.json
    scripts/run_live.py batch --n 5 --device "MacBook Pro Speakers"
    scripts/run_live.py batch --n 20 --tts say         # force the macOS fallback
    scripts/run_live.py batch --n 3 --mic              # speak the prompts yourself
    scripts/run_live.py devices

The gap is measured the way harness/exchange.py defines it. See live/loop.py's
module docstring, and live/STATUS.md for what was actually measured here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.loop import demo, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list audio devices and their reported latency")
    sc = sub.add_parser("selfcheck", help="run real turns, assert every stage timer")
    sc.add_argument("--n", type=int, default=2)
    sc.add_argument("--tts", default="auto", choices=("auto", "piper", "say"))
    b = sub.add_parser("batch", help="run n turns and write the timing JSON")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--out", default=None)
    b.add_argument("--device", default=None, help="output device name or index")
    b.add_argument("--tts", default="auto", choices=("auto", "piper", "say"))
    b.add_argument("--mic", action="store_true", help="read the microphone instead of a rendered prompt")
    a = ap.parse_args()

    if a.cmd == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if a.cmd == "selfcheck":
        demo(n_turns=a.n, tts_backend=a.tts)
        return 0

    res = run(n_turns=a.n, out_path=a.out, device=a.device, mic=a.mic, tts_backend=a.tts)
    s = res["summary_ms"]
    print(f"\nn = {res['n_turns']} turns   tts={res['tts']['backend']}  "
          f"asr={res['asr']['partial_model']}/{res['asr']['final_model']}  "
          f"out={res['output_device']} (+{res['output_device_latency_ms']:.0f}ms device)")
    print(f"loadavg {res['loadavg_start']} -> {res['loadavg_end']}  "
          f"(this laptop is shared; a CPU-bound ASR stage feels it)")
    print(f"{'':<28}{'median':>9}{'p25':>9}{'p75':>9}{'min':>9}{'max':>9}")
    for k, v in s.items():
        print(f"{k:<28}{v['median']:>9.1f}{v['p25']:>9.1f}{v['p75']:>9.1f}"
              f"{v['min']:>9.1f}{v['max']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
