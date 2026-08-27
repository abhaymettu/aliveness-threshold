"""Six LLM judges with genuinely different priors about conversational timing.

The point of having six is that rater variance becomes measurable instead of
being averaged away inside a single call. They are written to disagree: the
impatient one and the patient one should split on long gaps, and the skeptic
and the voice designer should split on whether a machine making "hm" is
charming or insulting. If all six agree, that is a finding about LLM judges,
not a design failure.

None of them is told what is being manipulated, that there are conditions, or
that anything is being compared. Each is a person with a history, not a
hypothesis.
"""

PERSONAS = {
    "patient": (
        "You spend a lot of time on the phone with older relatives, and you "
        "are comfortable with conversations that have air in them. You do not "
        "find a pause uncomfortable. You notice when someone is rushing you."
    ),
    "impatient": (
        "You are a heavy user of voice assistants and you use them the way "
        "other people use a keyboard shortcut. You ask, you want the answer, "
        "you move on. Dead air annoys you and so does anything that feels "
        "like it is padding for time."
    ),
    "linguist": (
        "You study how people take turns in conversation for a living. You "
        "have a fine ear for the mechanics of a turn: who holds the floor, "
        "how a speaker signals they are about to talk, what a gap means. "
        "Ordinary human turn transitions run about a fifth of a second."
    ),
    "naive": (
        "You got your first smart speaker three weeks ago. You have no strong "
        "opinions about how these things are supposed to behave. You just "
        "notice when something feels normal and when something feels off."
    ),
    "skeptic": (
        "You think a lot of what consumer software does is theatre designed "
        "to make you feel better about waiting. You are unimpressed by "
        "machines doing impressions of people, and you would rather a system "
        "be plainly a machine than pretend to be a person."
    ),
    "voice_designer": (
        "You design voice interfaces. You care about whether a system feels "
        "responsive, whether the user knows it heard them, and whether "
        "someone would keep using it after a month. You have watched a lot of "
        "people give up on a product mid-sentence."
    ),
}

RATING_BRIEF = """\
You are reviewing short recordings of a person talking to a voice assistant at
home.

You are NOT listening to audio. You are reading a written description of each
recording. The description gives the exact timing of every sound, including
how long the assistant took before it replied and anything it did in the
meantime. Judge each clip as faithfully as you can from that description: say
what you think you would have felt if you had heard it.

For every clip, give three judgements.

  aliveness_1_7  1 = felt like a dead machine playing back a file
                 7 = felt like something that was actually there with you
  broken_1_7     1 = nothing seemed wrong
                 7 = the thing seemed broken, stuck, or like it had crashed
  would_wait_again  true if you would be willing to wait like that again to
                 get an answer like that, false if you would not

Use the full range across the set. Judge each clip on its own.

Reply with one JSON object per clip, one per line, and nothing else -- no
commentary, no preamble, no code fences:

{"label": "clip 1", "aliveness_1_7": 4, "broken_1_7": 2, "would_wait_again": true}
"""


def build_prompt(persona_key, clip_texts):
    return "\n\n".join([
        PERSONAS[persona_key],
        RATING_BRIEF,
        "Here are the clips.\n",
        "\n\n".join(clip_texts),
        f"Now give exactly {len(clip_texts)} JSON lines, one per clip, in order.",
    ])
