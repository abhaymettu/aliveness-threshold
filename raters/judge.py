"""Run the LLM judges over data/stimuli.jsonl -> data/ratings.jsonl.

    .venv/bin/python -m raters.judge --smoke        # 1 persona, 6 clips
    .venv/bin/python -m raters.judge                # all personas, all clips

Backend is the local `claude -p` CLI: no API key, no new dependency. One
process per batch, a few in parallel.

THREE THINGS THIS FILE REFUSES TO DO
  1. Pretend the judge heard audio. It did not. Every row is written with
     rater_modality="transcript+timing" and the judge is told as much in its
     own prompt (see raters/render.py).
  2. Average the judges. Six personas, six rater_ids, one row each. Rater
     variance is a result, not noise to be smoothed before it is recorded.
  3. Invent a rating. A malformed or missing judgement is dropped and counted
     on stderr. Nothing is filled in, defaulted, or retried into existence.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys

from .personas import PERSONAS, build_prompt
from .render import render_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIMULI = os.path.join(ROOT, "data", "stimuli.jsonl")
RATINGS = os.path.join(ROOT, "data", "ratings.jsonl")
RUNS = os.path.join(ROOT, "raters", "runs")

MODALITY = "transcript+timing"   # not "audio". see the module docstring.


def load_stimuli(path=STIMULI):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def call_model(prompt, model, timeout=600):
    p = subprocess.run(["claude", "-p", "--model", model],
                       input=prompt, text=True, capture_output=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:400] or "claude -p failed")
    return p.stdout


def parse_judgements(text, expected_labels):
    """Pull JSON objects out of the reply. Anything malformed is dropped."""
    out, seen = {}, set()
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        label = str(o.get("label", "")).strip()
        if label not in expected_labels or label in seen:
            continue
        try:
            a = int(o["aliveness_1_7"])
            b = int(o["broken_1_7"])
            w = o["would_wait_again"]
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= a <= 7 and 1 <= b <= 7) or not isinstance(w, bool):
            continue
        seen.add(label)
        out[label] = {"aliveness_1_7": a, "broken_1_7": b,
                      "would_wait_again_bool": w}
    return out


def rate_batch(persona, stims, model):
    labels = [f"clip {i + 1}" for i in range(len(stims))]
    clips = [render_clip(s, lab) for s, lab in zip(stims, labels)]
    reply = call_model(build_prompt(persona, clips), model)
    got = parse_judgements(reply, set(labels))
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, missing = [], []
    for s, lab in zip(stims, labels):
        j = got.get(lab)
        if j is None:
            missing.append(s["stim_id"])
            continue
        rows.append({
            "stim_id": s["stim_id"],
            "rater_id": f"llm-{model}-{persona}",
            "rater_type": "llm",
            "rater_modality": MODALITY,
            "aliveness_1_7": j["aliveness_1_7"],
            "broken_1_7": j["broken_1_7"],
            "would_wait_again_bool": j["would_wait_again_bool"],
            "ts": ts,
        })
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--batch", type=int, default=10, help="clips per call")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--personas", default=",".join(PERSONAS))
    ap.add_argument("--limit", type=int, default=None, help="first N stimuli")
    ap.add_argument("--out", default=RATINGS)
    ap.add_argument("--smoke", action="store_true",
                    help="one persona, six clips, prints the rows")
    args = ap.parse_args()

    personas = args.personas.split(",")
    stimuli = load_stimuli()
    if args.smoke:
        personas, stimuli = personas[:1], stimuli[:6]
    elif args.limit:
        stimuli = stimuli[: args.limit]
    for p in personas:
        assert p in PERSONAS, f"unknown persona {p!r}"

    # Each persona sees the set in its own shuffled order, seeded by name, so
    # order effects are not shared across raters.
    jobs = []
    for persona in personas:
        order = list(stimuli)
        random.Random(f"{persona}/{args.model}").shuffle(order)
        for i in range(0, len(order), args.batch):
            jobs.append((persona, order[i:i + args.batch]))

    print(f"{len(stimuli)} stimuli x {len(personas)} personas "
          f"= {len(stimuli) * len(personas)} judgements in {len(jobs)} calls "
          f"({args.model})", file=sys.stderr)

    rows, missing, failed = [], [], []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(rate_batch, p, s, args.model): (p, s)
                for p, s in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            persona, stims = futs[fut]
            try:
                got, miss = fut.result()
            except Exception as e:
                failed.append((persona, len(stims), str(e)[:120]))
                print(f"  [{n}/{len(jobs)}] {persona} FAILED: {e}",
                      file=sys.stderr)
                continue
            rows += got
            missing += [(persona, m) for m in miss]
            print(f"  [{n}/{len(jobs)}] {persona} +{len(got)}"
                  + (f" ({len(miss)} unparsed)" if miss else ""), file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    os.makedirs(RUNS, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run": stamp, "model": args.model, "rater_type": "llm",
        "rater_modality": MODALITY, "personas": personas,
        "batch_size": args.batch, "n_stimuli": len(stimuli),
        "expected": len(stimuli) * len(personas), "written": len(rows),
        "dropped_unparsed": [m[1] for m in missing],
        "failed_calls": failed,
        "note": ("judges read raters/render.py descriptions, not audio; "
                 "no rating was defaulted or retried into existence"),
    }
    with open(os.path.join(RUNS, f"{stamp}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    exp = len(stimuli) * len(personas)
    print(f"wrote {len(rows)}/{exp} ratings -> {os.path.relpath(args.out, ROOT)}",
          file=sys.stderr)
    if len(rows) < exp:
        print(f"  {exp - len(rows)} missing (dropped, never invented): "
              f"{len(missing)} unparsed, {sum(f[1] for f in failed)} in failed calls",
              file=sys.stderr)
    if args.smoke:
        for r in rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
