"""Two-turn exchanges: a person asks a home assistant something, it answers.

Deliberately ordinary. No benchmark questions, no trivia-with-a-known-answer,
nothing that makes the rater think about whether the ANSWER is good. We are
measuring timing, so the content has to be uninteresting enough to get out of
the way.

Rules used when writing these:
  - the ask is one clause, the kind of thing you say without planning it
  - the answer is 8-18 words, hedged the way people hedge
  - no answer is impressive, surprising, or wrong in a noticeable way
  - no exchange rewards a long pause on content grounds (nothing that
    plausibly needs "computation"), otherwise latency stops being the
    manipulation and becomes part of the story
"""

EXCHANGES = [
    ("milk",     "Is there any milk left?",
                 "There's about half a carton, but it goes off tomorrow."),
    ("jacket",   "Do I need a jacket today?",
                 "It's fifty-two out and dropping, so probably yes."),
    ("music",    "Can you put something on that isn't too loud?",
                 "Sure, I'll start that ambient playlist you had on Sunday."),
    ("keys",     "Have you seen my keys anywhere?",
                 "They were on the arm of the couch about an hour ago."),
    ("pasta",    "How much longer on the pasta?",
                 "Four minutes, and I'll say something when it's close."),
    ("thursday", "Am I free Thursday afternoon?",
                 "You've got a dentist thing at two, otherwise it's open."),
    ("salt",     "Do you think this needs more salt?",
                 "I can't taste it, but that recipe usually runs a little bland."),
    ("plant",    "Why is this plant going yellow?",
                 "You've been watering it twice a week, which is more than it wants."),
    ("film",     "What was that film we watched with the boat?",
                 "The one with the storm at the end, I think that was All Is Lost."),
    ("bridge",   "Is the bridge backed up again?",
                 "Yeah, about twenty minutes slower than usual heading east."),
    ("electric", "Did the electric bill go through?",
                 "It cleared Tuesday, ninety-one dollars and change."),
    ("gift",     "What do you get someone who has everything?",
                 "Honestly, usually something they'd never buy for themselves."),
    ("tired",    "Why am I so tired lately?",
                 "You've averaged about five and a half hours this week."),
    ("butter",   "Can I swap butter for oil in this?",
                 "You can, it'll just come out a bit flatter and chewier."),
    ("laundry",  "Is it going to rain before the laundry dries?",
                 "There's a decent chance around four, so I'd bring it in."),
    ("dog",      "Does he look limpy to you?",
                 "A little on the back left, but he's been running all morning."),
    ("landlord", "Did I ever hear back from the landlord?",
                 "Nothing since the message you sent on the ninth."),
    ("window",   "Did you leave the window open last night?",
                 "I did, and it's why the front room is freezing right now."),
]

EXCHANGE_IDS = [e[0] for e in EXCHANGES]
BY_ID = {e[0]: {"exchange_id": e[0], "prompt_text": e[1], "response_text": e[2]}
         for e in EXCHANGES}

assert len(EXCHANGE_IDS) == len(set(EXCHANGE_IDS)), "duplicate exchange_id"
