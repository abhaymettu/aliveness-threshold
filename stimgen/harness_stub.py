"""Local stand-in for harness.synthesize_exchange.

The real harness/ is being built by someone else. This implements the same
signature so stimgen can run today, and stimgen/generate.py will prefer the
real module the moment it imports.

    synthesize_exchange(prompt_text, response_text, latency_ms, cue,
                        out_wav_path) -> dict

Everything is macOS `say` plus stdlib `wave`. No numpy, no ffmpeg.

TIMING MODEL (this is the part that matters):

    [prompt] [------------- gap -------------] [response]
             ^CUE_ONSET_MS
                          [cue]

The cue lives INSIDE the gap, it is not added on top of it. That is the
ecological claim the project is testing -- "hm" is something the robot does
while you are already waiting, not an extra thing you wait through.

Consequence, stated plainly rather than hidden: when latency_ms is too short
to contain the cue, the gap has to grow. cue onset (100ms) + cue duration +
an 80ms tail is the floor. So at nominal 0/200/400 the cued clips genuinely
have a longer gap than the uncued ones. That is not a bug and it is not
papered over -- actual_gap_ms in stimuli.jsonl is the measured truth and
analysis must regress on it, not on latency_ms.

That clamping is also why the exchange rate is estimable at all: cued and
uncued clips overlap in actual gap at 800/1200/1600ms, so "same wait, cue vs
no cue" is a real within-design contrast rather than an extrapolation.

ponytail: `say` voices are not a real dialogue system and the breath is
synthesized noise, not a recorded human inbreath. Ceiling: cue naturalness.
Upgrade path is a neural TTS with real disfluency tokens and a recorded breath
bank, swapped in behind this same function signature.
"""

import array
import math
import os
import random
import subprocess
import wave

SR = 24000
VOICE_HUMAN = "Samantha"   # the person asking
VOICE_ROBOT = "Daniel"     # the thing answering -- distinct speaker on purpose

CUE_RATE_WPM = 200         # cues only -- `say` drags short non-lexical tokens
CUE_ONSET_MS = 100         # how soon after the prompt ends the cue starts
CUE_TAIL_MS = 80           # minimum silence between cue end and response
TRIM_DB = -45.0            # silence floor for edge trimming

# Kept short on purpose. Every cue costs time, and any cue longer than
# ~620ms would force the 800ms condition open too (see the timing note above).
# At 200 wpm these land at hm 280ms, mm-hm 580ms, hang-on 550ms; the breath is
# 380ms. Floors are therefore 460/760/730/560ms -- so 800, 1200 and 1600 are
# exact for every cue, and only 0/200/400 get clamped.
CUE_TEXT = {
    "filled_pause": "hm",
    "backchannel": "mm hm",
    "verbal_stall": "hang on",
}


# --------------------------------------------------------------------------
# wav plumbing

def _read(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        assert w.getframerate() == SR, f"{path} is {w.getframerate()}Hz"
        return array.array("h", w.readframes(w.getnframes()))


def _write(path, samples):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def _ms(n_samples):
    return round(1000.0 * n_samples / SR, 2)


def _trim(samples, win_ms=10):
    """Strip leading/trailing near-silence so gaps mean what they say.

    `say` pads its output, and unpadded padding would silently add 100-200ms
    to every gap in the study.
    """
    win = int(SR * win_ms / 1000)
    floor = (10 ** (TRIM_DB / 20)) * 32768
    def loud(i):
        chunk = samples[i:i + win]
        if not chunk:
            return False
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        return rms > floor
    starts = range(0, len(samples) - win, win)
    first = next((i for i in starts if loud(i)), 0)
    last = next((i for i in reversed(list(starts)) if loud(i)), len(samples) - win)
    return samples[max(0, first - win):min(len(samples), last + 2 * win)]


# --------------------------------------------------------------------------
# audio sources

def _say(text, voice, path, rate=None):
    cmd = ["say", "-v", voice, "-o", path,
           "--data-format=LEI16@%d" % SR, "--file-format=WAVE"]
    if rate:
        cmd += ["-r", str(rate)]
    subprocess.run(cmd + [text], check=True, capture_output=True)
    return _trim(_read(path))


def warm_up():
    """First use of a `say` voice can render differently than every use after.

    Cheap insurance so clip 1 is not systematically unlike clips 2..90.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for v in (VOICE_HUMAN, VOICE_ROBOT):
            _say("warm up", v, f"{d}/w.wav")


def _breath(dur_ms=380, seed=7):
    """A synthesized audible inbreath: lowpassed noise, slow-in / quick-out.

    Not a recorded breath. Named as a stub for exactly that reason.
    """
    rng = random.Random(seed)
    n = int(SR * dur_ms / 1000)
    out = array.array("h", bytes(2 * n))
    lp = 0.0
    for i in range(n):
        white = rng.uniform(-1, 1)
        lp += 0.06 * (white - lp)                    # one-pole lowpass ~230Hz
        band = white * 0.25 + lp * 3.0               # breathy, not hissy
        t = i / n
        env = (t ** 0.6) * math.exp(-3.2 * t) * 2.4  # inbreath shape
        out[i] = max(-32000, min(32000, int(band * env * 9000)))
    return out


def _cue_audio(cue, tmp_prefix):
    if cue == "none":
        return array.array("h")
    if cue == "breath":
        return _breath()
    return _say(CUE_TEXT[cue], VOICE_ROBOT, f"{tmp_prefix}-cue.wav",
                rate=CUE_RATE_WPM)


# --------------------------------------------------------------------------
# the API the rest of stimgen codes against

def synthesize_exchange(prompt_text, response_text, latency_ms, cue,
                        out_wav_path):
    tmp = os.path.splitext(out_wav_path)[0] + ".tmp"
    prompt = _say(prompt_text, VOICE_HUMAN, f"{tmp}-p.wav")
    response = _say(response_text, VOICE_ROBOT, f"{tmp}-r.wav")
    cue_buf = _cue_audio(cue, tmp)

    onset = int(SR * CUE_ONSET_MS / 1000) if cue != "none" else 0
    floor = onset + len(cue_buf) + int(SR * CUE_TAIL_MS / 1000) if cue != "none" else 0
    gap_n = max(int(SR * latency_ms / 1000), floor)

    gap = array.array("h", bytes(2 * gap_n))
    gap[onset:onset + len(cue_buf)] = cue_buf

    _write(out_wav_path, prompt + gap + response)
    for f in (f"{tmp}-p.wav", f"{tmp}-r.wav", f"{tmp}-cue.wav"):
        if os.path.exists(f):
            os.remove(f)

    return {
        "actual_gap_ms": _ms(gap_n),
        "cue_onset_ms": _ms(onset) if cue != "none" else None,
        "wav_path": out_wav_path,
        "component_timings": {
            "prompt_ms": _ms(len(prompt)),
            "gap_ms": _ms(gap_n),
            "cue_ms": _ms(len(cue_buf)),
            "response_ms": _ms(len(response)),
            "total_ms": _ms(len(prompt) + gap_n + len(response)),
        },
    }


if __name__ == "__main__":
    import tempfile
    warm_up()
    with tempfile.TemporaryDirectory() as d:
        for cue in ["none", "filled_pause", "breath", "backchannel", "verbal_stall"]:
            r = synthesize_exchange("Is there any milk left?",
                                    "There's about half a carton.",
                                    800, cue, f"{d}/{cue}.wav")
            assert os.path.exists(r["wav_path"])
            assert r["actual_gap_ms"] >= 800
            print(cue, r["actual_gap_ms"], r["component_timings"])
        # the honest failure mode: a long cue cannot fit in a 0ms gap
        r = synthesize_exchange("Is there any milk left?", "Half a carton.",
                                0, "verbal_stall", f"{d}/x.wav")
        assert r["actual_gap_ms"] > 0, "cue must force the gap open"
        print("nominal 0ms + verbal_stall really is", r["actual_gap_ms"], "ms")
        # every cue has to fit inside 800ms or the mid conditions collapse
        for cue in CUE_TEXT:
            g = synthesize_exchange("Is there any milk left?", "Half a carton.",
                                    800, cue, f"{d}/f.wav")["actual_gap_ms"]
            assert g == 800.0, f"{cue} does not fit in 800ms (got {g})"
