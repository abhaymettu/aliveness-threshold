"""Turn a stimulus row into text a judge that cannot hear can actually rate.

This file is the honesty boundary of the whole rating pipeline.

An LLM driven through `claude -p` receives no audio. It cannot hear a breath.
So we do not hand it audio and we do not pretend we did: every row this
pipeline writes carries rater_modality="transcript+timing", and the judge is
told in its own prompt that it is reading a description rather than listening.

Two rules the rendering follows:

1. FAITHFUL. Timings come from the measured fields (actual_gap_ms,
   cue_onset_ms, cue_dur_ms), never from the nominal latency_ms. If a cue
   forced the gap open, the judge sees the gap that actually exists.

2. UNLABELLED. The judge never sees the words "cue", "filled_pause",
   "condition", "latency", or any hint that anything is being manipulated.
   Cues are described phenomenally -- what a listener would report hearing --
   because naming the condition is the fastest way to manufacture the effect
   you were hoping to measure.

Whether a model reading this description responds to social timing the way a
listener does is an open question, not an assumption. It is exactly what
comparing these rows against human rows is for.
"""

# What a listener would say they heard. No condition names, no theory.
CUE_SOUND = {
    "filled_pause": 'a short nasal "hm"',
    "breath":       "an audible breath in",
    "backchannel":  'a two-note "mm-hm", the sound someone makes to show they heard you',
    "verbal_stall": 'the words "hang on"',
}


def _s(ms):
    return f"{ms / 1000.0:.2f} s"


def render_clip(stim, label):
    """One clip as plain prose. `label` is what the judge answers under."""
    gap = float(stim["actual_gap_ms"])
    lines = [f"[{label}]",
             f'PERSON: "{stim["prompt_text"]}"']

    if stim["cue"] == "none" or not stim.get("cue_dur_ms"):
        lines.append(f"Then {_s(gap)} of silence. Nothing is heard during it.")
    else:
        onset = float(stim["cue_onset_ms"])
        dur = float(stim["cue_dur_ms"])
        tail = gap - onset - dur
        lines.append(f"Then a gap of {_s(gap)} before the reply. During the gap:")
        lines.append(f"  - {_s(onset)} of silence")
        lines.append(f"  - then the assistant makes {CUE_SOUND[stim['cue']]}, "
                     f"lasting {_s(dur)}")
        lines.append(f"  - then {_s(tail)} more silence")

    lines.append(f'ASSISTANT: "{stim["response_text"]}"')
    return "\n".join(lines)


if __name__ == "__main__":
    base = {"prompt_text": "Do I need a jacket today?",
            "response_text": "It's fifty-two out and dropping, so probably yes."}

    quiet = render_clip({**base, "cue": "none", "actual_gap_ms": 1200.0,
                         "cue_onset_ms": None, "cue_dur_ms": 0.0}, "clip 1")
    assert "1.20 s of silence" in quiet
    assert "hm" not in quiet

    cued = render_clip({**base, "cue": "filled_pause", "actual_gap_ms": 1200.0,
                        "cue_onset_ms": 100.0, "cue_dur_ms": 280.0}, "clip 2")
    assert "1.20 s" in cued and '"hm"' in cued
    assert "0.82 s more silence" in cued, cued   # 1200 - 100 - 280

    # the condition must never leak through the description
    for text in (quiet, cued):
        low = text.lower()
        for banned in ("cue", "filled_pause", "latency", "condition",
                       "backchannel", "stimulus", "aliveness"):
            assert banned not in low, f"{banned!r} leaked into the judge prompt"

    print(quiet, "\n---\n", cued, sep="")
