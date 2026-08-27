#!/usr/bin/env python3
"""CLI over the live voice loop (audio in -> ASR -> LM -> TTS -> audio out).

    scripts/run_live.py selfcheck                      # 2 turns, assert every timer
    scripts/run_live.py batch --n 20 --out live/results/run.json
    scripts/run_live.py batch --n 5 --device "MacBook Pro Speakers"
    scripts/run_live.py batch --n 20 --tts say         # force the macOS fallback
    scripts/run_live.py batch --n 20 --fast            # downstream runs in the hangover
    scripts/run_live.py batch --n 20 --fast --hangover 200
    scripts/run_live.py batch --n 20 --final-model tiny.en
    scripts/run_live.py batch --n 3 --mic              # speak the prompts yourself
    scripts/run_live.py devices

Without --fast the loop is the original serial path, unchanged, so the baseline
stays runnable and every optimisation has a before to sit next to.

The gap is measured the way harness/exchange.py defines it. See live/loop.py's
module docstring, and live/STATUS.md for what was actually measured here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.loop import demo, run  # noqa: E402


def _opts(a) -> dict:
    """Only pass what was actually asked for, so loop.py owns the defaults."""
    o = {"fast": a.fast}
    if a.hangover is not None:
        o["hangover"] = a.hangover
    if a.final_model is not None:
        o["final_model"] = a.final_model
    if a.arm is not None:
        o["arm_ms"] = a.arm
    return o


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list audio devices and their reported latency")
    sc = sub.add_parser("selfcheck", help="run real turns, assert every stage timer")
    sc.add_argument("--n", type=int, default=2)
    sc.add_argument("--tts", default="auto", choices=("auto", "piper", "say"))
    sc.add_argument("--fast", action="store_true",
        help="run the final decode, LM and TTS inside the endpointer's hangover")
    sc.add_argument("--hangover", type=float, default=None,
        help="trailing silence before the turn is called over, ms (default 350)")
    sc.add_argument("--final-model", default=None,
        help="faster-whisper model for the final decode (default base.en)")
    sc.add_argument("--arm", type=float, default=None,
        help="--fast: silence before the speculative pipeline starts, ms")
    b = sub.add_parser("batch", help="run n turns and write the timing JSON")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--out", default=None)
    b.add_argument("--device", default=None, help="output device name or index")
    b.add_argument("--tts", default="auto", choices=("auto", "piper", "say"))
    b.add_argument("--fast", action="store_true",
        help="run the final decode, LM and TTS inside the endpointer's hangover")
    b.add_argument("--hangover", type=float, default=None,
        help="trailing silence before the turn is called over, ms (default 350)")
    b.add_argument("--final-model", default=None,
        help="faster-whisper model for the final decode (default base.en)")
    b.add_argument("--arm", type=float, default=None,
        help="--fast: silence before the speculative pipeline starts, ms")
    b.add_argument("--mic", action="store_true", help="read the microphone instead of a rendered prompt")
    a = ap.parse_args()

    if a.cmd == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if a.cmd == "selfcheck":
        demo(n_turns=a.n, tts_backend=a.tts, **_opts(a))
        return 0

    res = run(n_turns=a.n, out_path=a.out, device=a.device, mic=a.mic,
              tts_backend=a.tts, **_opts(a))
    s = res["summary_ms"]
    print(f"\nn = {res['n_turns']} turns   tts={res['tts']['backend']}  "
          f"asr={res['asr']['partial_model']}/{res['asr']['final_model']}  "
          f"out={res['output_device']} (+{res['output_device_latency_ms']:.0f}ms device)")
    print(f"loadavg {res['loadavg_start']} -> {res['loadavg_end']}  "
          f"(this laptop is shared; a CPU-bound ASR stage feels it)")
    e, sp = res["endpointing"], res["speculation"]
    print(f"mode={res['mode']}  hangover={res['hangover_ms']:.0f}ms  "
          f"speculative {sp['turns_served_speculatively']}/{res['n_turns']} turns "
          f"({sp['pipelines_launched']} launched)")
    print(f"false endpoints {e['false_endpoints']}/{e['n']}   "
          f"mean WER vs prompt {e['wer_vs_prompt']}   "
          f"(newest partial would have been {e['wer_newest_partial']})")
    print(f"{'':<28}{'median':>9}{'p25':>9}{'p75':>9}{'min':>9}{'max':>9}")
    for k, v in s.items():
        print(f"{k:<28}{v['median']:>9.1f}{v['p25']:>9.1f}{v['p75']:>9.1f}"
              f"{v['min']:>9.1f}{v['max']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
