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

WHAT THE FIRST FULL RUN ACTUALLY LOOKED LIKE (sonnet, 6 personas, n=540)

Reported here rather than left for someone to discover, because it is a
problem for this arm of the study and not a result to lead with.

  mean aliveness by cue    none 2.27 | filled_pause 4.42 | breath 4.89
                           backchannel 5.39 | verbal_stall 5.03
  mean aliveness by gap    0ms 4.59 ... 1600ms 3.63

The cue effect is about +2.6 points on a 7-point scale. The latency effect is
about -1.0. Worse, within cue=none the judges are flat across the whole
latency range (2.27 to 2.56 from 0ms to 1600ms) -- they barely register how
long the silence was when nothing happened in it.

That is not what a listener does, and there are at least two mundane
explanations before anything about social timing:

  - a cued clip's description is three lines longer and names a sound. The
    judge may be responding to "something was described" rather than to when
    it happened. Description length is confounded with cue by construction
    and cannot be unconfounded in text.
  - "nothing is heard during it" may simply read as dead, at any duration.

Latency does move the other two DVs (broken 1.06 -> 2.96, would-wait-again
100% -> 59% from 0 to 1600ms), so the judges are not ignoring time entirely;
it is aliveness specifically that looks cue-driven.

Read this as: the LLM arm measures something, and whether that something is
the same construct a listener reports is exactly the open question. It is
what web/rate.html exists to settle. Do not report an exchange rate estimated
from these rows as if it were a human exchange rate.
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


def call_model(prompt, model, timeout=300, tries=2):
    """Ask the model. A transport failure gets ONE retry -- re-asking is not
    the same as inventing an answer. A reply that arrives but is malformed is
    never retried into existence; it is dropped in parse_judgements."""
    last = None
    for _ in range(tries):
        try:
            p = subprocess.run(["claude", "-p", "--model", model],
                               input=prompt, text=True, capture_output=True,
                               timeout=timeout)
            if p.returncode == 0:
                return p.stdout
            last = RuntimeError(p.stderr.strip()[:300] or "claude -p failed")
        except subprocess.TimeoutExpired as e:
            last = e
    raise last


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


def verify(path=RATINGS):
    """Assert the ratings file is joinable and honestly labelled.

        .venv/bin/python -m raters.judge --verify
    """
    stims = {s["stim_id"] for s in load_stimuli()}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert rows, f"{path} is empty"
    for r in rows:
        assert r["stim_id"] in stims, f"orphan rating: {r['stim_id']}"
        assert r["rater_type"] in ("llm", "human"), r["rater_type"]
        # an llm judge must never be filed as, or silently look like, a human
        if r["rater_type"] == "llm":
            assert r["rater_id"].startswith("llm-"), r["rater_id"]
            assert r["rater_modality"] != "audio", \
                f"{r['rater_id']} claims it heard audio; it cannot"
        assert 1 <= r["aliveness_1_7"] <= 7 and isinstance(r["aliveness_1_7"], int)
        assert 1 <= r["broken_1_7"] <= 7 and isinstance(r["broken_1_7"], int)
        assert isinstance(r["would_wait_again_bool"], bool)
        dt.datetime.fromisoformat(r["ts"])

    types = {}
    for r in rows:
        types[r["rater_type"]] = types.get(r["rater_type"], 0) + 1
    print(f"rater_type counts: {types}")

    by_rater = {}
    for r in rows:
        by_rater.setdefault(r["rater_id"], []).append(r)
    print(f"ok: {len(rows)} ratings over {len(by_rater)} raters, "
          f"{len({r['stim_id'] for r in rows})}/{len(stims)} stimuli covered")
    for rid, rs in sorted(by_rater.items()):
        mean = sum(x["aliveness_1_7"] for x in rs) / len(rs)
        wait = sum(x["would_wait_again_bool"] for x in rs) / len(rs)
        print(f"    {rid:<34} n={len(rs):<4} aliveness={mean:.2f}  "
              f"would-wait={wait:.0%}  ({rs[0]['rater_modality']})")


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
    ap.add_argument("--fill", action="store_true",
                    help="rate only the persona x stimulus pairs missing from --out")
    ap.add_argument("--verify", action="store_true",
                    help="check the existing ratings file and exit")
    args = ap.parse_args()

    if args.verify:
        return verify(args.out)

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
    done = set()
    if args.fill and os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["rater_id"], r["stim_id"]))

    jobs = []
    for persona in personas:
        order = list(stimuli)
        random.Random(f"{persona}/{args.model}").shuffle(order)
        if args.fill:
            rid = f"llm-{args.model}-{persona}"
            order = [s for s in order if (rid, s["stim_id"]) not in done]
        for i in range(0, len(order), args.batch):
            jobs.append((persona, order[i:i + args.batch]))
    n_todo = sum(len(j[1]) for j in jobs)

    print(f"{len(stimuli)} stimuli x {len(personas)} personas -> "
          f"{n_todo} judgements to collect in {len(jobs)} calls ({args.model})"
          + (" [fill]" if args.fill else ""), file=sys.stderr)
    if not jobs:
        print("nothing missing", file=sys.stderr)
        return

    rows, missing, failed = [], [], []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(rate_batch, p, s, args.model): (p, s)
                for p, s in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            persona, stims = futs[fut]
            try:
                got, miss = fut.result()
            except Exception as e:
                failed.append({"persona": persona,
                               "stim_ids": [x["stim_id"] for x in stims],
                               "error": str(e)[:160]})
                print(f"  [{n}/{len(jobs)}] {persona} FAILED ({len(stims)} clips): {e}",
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
        "mode": "fill" if args.fill else "full",
        "note": ("judges read raters/render.py descriptions, not audio; "
                 "no rating was defaulted or retried into existence"),
    }
    with open(os.path.join(RUNS, f"{stamp}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    exp = n_todo
    print(f"wrote {len(rows)}/{exp} ratings -> {os.path.relpath(args.out, ROOT)}",
          file=sys.stderr)
    if len(rows) < exp:
        print(f"  {exp - len(rows)} missing (dropped, never invented): "
              f"{len(missing)} unparsed, {sum(len(f['stim_ids']) for f in failed)} in failed calls",
              file=sys.stderr)
    if args.smoke:
        for r in rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
