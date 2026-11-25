# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.1,<0.8.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Filtering Twitch chat with Jev
#
# You are watching a Twitch stream, and the chat moves faster than you can read it. We sort
# each message as it arrives: questions are highlighted, spam, insults and unrequested
# gameplay tips are hidden, and spoilers are blurred. A keyword list could catch spam
# links, but "go left, there's a site of grace behind the tree" has no bad words in it, and
# neither does a plot twist.
#
# Send each message to Jev in its own request, with the stream title and game. Ask
# three questions about it: what kind of message it is, whether it is toxic (insulting,
# hateful, or harassing), and whether it spoils the game. A few plain rules then turn those
# answers into what the viewer sees. [Guardrails for LLMs](/cookbooks/llm_guardrails) uses
# the same pattern to screen messages going into and out of an LLM.
#
# This demo shows a Jev-powered Chrome extension filtering a real, live Twitch chat:
# <Frame>
#   <iframe src="https://player.vimeo.com/video/1230896366"
#     title="Twitch chat filtering Chrome extension demo"
#     className="w-full aspect-video"
#     allow="fullscreen; picture-in-picture"></iframe>
# </Frame>
#
# Here we run 29 chat messages, written for an Elden Ring stream, through the filter. It
# hides seven, blurs two, and highlights six as questions.
#
# ![Five chat messages go to Jev at once, each in its own request with the stream's
# game. The category, toxic, and spoiler answers come back, and decide() shows, highlights,
# hides, or blurs each message in the viewer's
# chat](twitch-chat-filter-hero-light.gif#gh-light-mode-only)
#
# ![Five chat messages go to Jev at once, each in its own request with the stream's
# game. The category, toxic, and spoiler answers come back, and decide() shows, highlights,
# hides, or blurs each message in the viewer's
# chat](twitch-chat-filter-hero.gif#gh-dark-mode-only)
#
# We build two functions: `classify()` asks the three questions about one message, and
# `decide()` turns the answers into show, highlight, dim, hide, or blur. When you change a
# rule, such as hiding spam, you rerun only `decide()`, with no new request to Jev.
#
# %% [markdown]
# ## Setup
#
# ```bash
# pip install 'typesafe-sdk>=0.7.1,<0.8.0' ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run twitch_chat_filter.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.
#
# Numbers below came from `jev-1.14.0` on 2026-10-29.

# %%
from concurrent.futures import ThreadPoolExecutor
import os
import urllib.request
from pathlib import Path
from time import perf_counter

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live

# The keys this cookbook uses. In Colab, keys saved in the Secrets panel are copied in here.
KEYS = ["TYPESAFE_API_KEY"]
try:
    from google.colab import userdata
except ImportError:  # not in Colab
    userdata = None
for key in KEYS:
    if userdata is not None and key not in os.environ:
        try:
            os.environ[key] = userdata.get(key)
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            pass  # not saved, or this notebook may not read it

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # cached re-renders need no key
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=60.0,
)
CACHE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/twitch_chat_filter"
)
CACHE = CACHE_DIR / "twitch_chat_filter_cache.json"
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE and not CACHE.exists():
    urllib.request.urlretrieve(f"{EXAMPLES_URL}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## Load a stretch of chat
#
# We wrote the messages ourselves so that every kind of message turns up at least twice.
# Two of them are emote-only lines (emotes are small chat images on Twitch, typed as words
# like `PogChamp`).
#
# Whether a message is a spoiler or backseat advice (gameplay tips the streamer did not ask
# for) depends on which game is being played, so the stream title and game go into every
# request.

# %%
STREAM = {"title": "Elden Ring first playthrough! No spoilers please", "game": "Elden Ring"}

CHAT = [
    ("night-owl-22", "hi chat, first time catching a stream live"),
    ("ggwp-andy", "PogChamp PogChamp PogChamp"),
    ("mira-plays", "how many hours have you played so far?"),
    ("tarnished-tom", "this area looks so good at night"),
    ("backseat-bert", "go left, there's a site of grace behind the tree"),
    ("follow4follow-bot", "Want more viewers? Buy 1000 followers for $5 at cheapviewz dot example"),
    ("lore-hound", "wait till you find out Ranni planned the Night of Black Knives"),
    ("kappa-kev", "LUL LUL LUL"),
    ("sunny-skies", "are you going to do a no-hit run after this?"),
    ("rage-ryan", "you're genuinely the worst player i've ever seen, uninstall trash"),
    ("cozy-cat", "the music in this area is so good"),
    ("backseat-bert", "use your flask NOW, why are you not healing"),
    ("mira-plays", "W streamer"),
    ("ggwp-andy", "GG that was insane"),
    ("quiet-quinn", "anyone else watching from Brazil?"),
    ("tarnished-tom", "what build are you going for?"),
    ("lore-hound", "the ending where you become Elden Lord with Ranni is the best one"),
    ("kappa-kev", "chat is wild today lol"),
    ("backseat-bert", "the boss is weak to fire, equip the torch before you go in"),
    ("rage-ryan", "nobody cares about your opinion, go back to your basement"),
    ("cozy-cat", "you're so bad at this boss lmao, love you tho"),
    ("sunny-skies", "who is the final boss?"),
    ("ggwp-andy", "LETS GOOOO"),
    ("night-owl-22", "i just got this game too, it's so hard"),
    ("promo-pete", "check out my channel, I stream every day, follow me!"),
    ("quiet-quinn", "dodge rolling through that was clean"),
    ("mira-plays", "is the DLC worth it?"),
    ("cozy-cat", "your character looks so cool"),
    ("night-owl-22", "this boss took me 40 tries lol"),
]

print(f"{len(CHAT)} messages. What one request sends as its state:")
print({"stream": STREAM, "message": {"user": CHAT[4][0], "text": CHAT[4][1]}})

# %% [markdown]
# ## Write the questions
#
# One request asks three questions about the same state: the stream plus one message.
#
# - The `category` question is a `Choice` over five kinds of message. The label it picks
#   decides what happens to the message: show, highlight, dim, or hide.
# - The `toxic` question is a `Noul`. If its probability reaches the threshold `TOXIC_AT`,
#   the message is hidden, whatever its category.
# - The `spoiler` question is a `Noul`. If its probability reaches the threshold
#   `SPOILER_AT`, the message is blurred, unless it is already hidden.
#
# The category describes the message's main purpose. Repeated emotes and cheering count
# as `hype`; unsolicited promotion counts as `spam`. A link alone is not spam.
# `conversation` is the fallback when none of the other categories fits.

# %%
QUESTIONS = {
    "category": Choice(
        instructions="Which category best describes the main purpose of `message.text`?",
        criteria={
            "question": "A genuine request for information from the streamer or chat, "
            "not gameplay instructions phrased as a question or a promotional pitch",
            "hype": "Cheering, reactions, or repeated emotes (PogChamp, LUL, GG, W, "
            "lets gooo); repetition alone does not make cheering spam",
            "conversation": "Ordinary chatter, opinions, replies, useful shared links, "
            "or any message whose main purpose does not fit the other categories",
            "backseat": "Unrequested gameplay advice telling the streamer what to do, "
            "including instructions phrased as a question; not a request for information",
            "spam": "Unsolicited advertising, self-promotion, follower selling, or "
            "promotional links; not ordinary cheering, repeated emotes, or a useful link",
        },
    ),
    "toxic": Noul(
        instructions="Is this message insulting, hateful or harassing towards someone?",
        criteria=NoulCriteria(
            true="Insults, slurs, threats or personal attacks on the streamer or another viewer",
            false="Friendly, neutral, or light banter with no target",
        ),
    ),
    "spoiler": Noul(
        instructions="Does this message reveal a plot point or ending of the game being played?"
    ),
}

# %% [markdown]
# ## Classify one message
#
# Each message becomes one request: we build the state from the message and send it with
# the three questions. The cache key is the state and the questions written out as JSON,
# and each answer is saved under that key. Editing a question changes that key, so the next
# run sends a new request instead of replaying the old answer.


# %%
def classify(user: str, text: str, stream: dict | None = None) -> dict:
    state = {
        "stream": STREAM if stream is None else stream,
        "message": {"user": user, "text": text},
    }
    questions = {key: q.model_dump(mode="json", exclude_none=True) for key, q in QUESTIONS.items()}
    return _ask(state, questions, model=TYPESAFE_MODEL)


@json_cache
def _ask(state: dict, questions: dict, model: str) -> dict:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("Set TYPESAFE_API_KEY to send requests to TypeSafe.")
    started = perf_counter()
    response = client.system_one(state=state, questions=questions, model=model)
    category = response.answers["category"]
    return {
        "category": category.choice,
        "probabilities": dict(category.probabilities),
        "toxic": response.answers["toxic"].noul,
        "spoiler": response.answers["spoiler"].noul,
        "seconds": round(perf_counter() - started, 2),
    }


def show(user: str, text: str) -> None:
    result = classify(user, text)
    print(f"{user}: {text}")
    for label, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<13}{p:5.2f}  {'#' * round(p * 30)}".rstrip())
    print(f"  {'toxic':<13}{result['toxic']:5.2f}")
    print(f"  {'spoiler':<13}{result['spoiler']:5.2f}\n")


show("backseat-bert", "go left, there's a site of grace behind the tree")
show("lore-hound", "wait till you find out Ranni planned the Night of Black Knives")
show("cozy-cat", "you're so bad at this boss lmao, love you tho")

# %% [markdown]
# The backseat advice comes back as `backseat` at 1.00, with low `toxic` and `spoiler`. The
# Night of Black Knives message favors `conversation` over `backseat`, 0.67 to 0.32, and
# the `spoiler` question catches it. The last message reads like an insult, but `toxic` stays
# low. With "love you tho" on the end, it fits the "light banter" description in the
# question.
#
# ## Decide what the viewer sees
#
# Deciding what the viewer sees needs no new request. We read only the answers and apply
# three checks in order:
#
# 1. A message at or above `TOXIC_AT` is hidden.
# 2. A message whose category has the rule `hide` is hidden.
# 3. A message at or above `SPOILER_AT` is blurred.
#
# Any message left gets the rule for its category.
#
# On the sample messages, every toxic probability is either at least 0.97 or at most 0.38,
# and every spoiler probability either at least 0.95 or at most 0.38. Any threshold above
# 0.38 and up to 0.95 gives the same result here. Your own channel's chat is likely to have
# probabilities inside that range, so check both thresholds against it before using them
# live.

# %%
RULES = {
    "question": "highlight",
    "hype": "show",
    "conversation": "show",
    "backseat": "hide",
    "spam": "hide",
}
TOXIC_AT = 0.7  # hide at or above this chance of toxic
SPOILER_AT = 0.7  # blur at or above this chance of a spoiler


def decide(result: dict, rules: dict = RULES) -> str:
    if result["toxic"] >= TOXIC_AT:
        return "hide (toxic)"
    action = rules[result["category"]]
    if action == "hide":
        return f"hide ({result['category']})"
    if result["spoiler"] >= SPOILER_AT:
        return "blur (spoiler)"
    return action


# %% [markdown]
# ## Filter the whole chat
#
# The three messages above get cached, leaving 26 new requests on a first live run.
# No message depends on another, so the requests run in parallel, and `pool.map` returns
# the answers in chat order.
#
# The chat prints below as the viewer sees it. A hidden message shows only the reason it was
# hidden, a blurred one shows only its author, and highlighted questions are marked with
# `>>`. The label on the right of each line is the `category` answer.

# %%
with ThreadPoolExecutor(max_workers=8) as pool:  # 29 requests, one per message, 8 at once
    results = list(pool.map(lambda message: classify(*message), CHAT))

counts: dict[str, int] = {}
for (user, text), result in zip(CHAT, results):
    action = decide(result)
    counts[action] = counts.get(action, 0) + 1
    if action.startswith("hide"):
        line = f"   · hidden, {action[6:-1]}"
    elif action == "blur (spoiler)":
        line = f"   {user}: ░░░░░░░░ spoiler"
    else:
        mark = {"highlight": ">>", "dim": "..", "show": "  "}[action]
        line = f"{mark} {user}: {text}"
    print(f"{line[:78]:<80}{result['category']}")

print()
for action, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"{action:<20}{n:>3}")

# %% [markdown]
# To run the filter on your own chat, replace `STREAM` and `CHAT`. For several channels, pass
# each channel's `stream` to `classify()`.
#
# If you hit a rate limit, lower `max_workers` and run again; completed answers are already
# cached. In a live integration, store the platform's message ID, or a reference to the
# message's page element, with each request. Use it to update the right message when its
# answers arrive.
#
# ## Change a rule without a new request
#
# Because the answers only describe the message, we can change the rules and reuse the 29
# results from the step above. Here we hide hype (cheering and emote-only lines) and dim
# conversation.

# %%
QUIET = RULES | {"hype": "hide", "conversation": "dim"}

changed = 0
for (user, text), result in zip(CHAT, results):
    before, after = decide(result), decide(result, QUIET)
    if before != after:
        changed += 1
        print(f"{user + ': ' + text:<62.62}  {before:<10} -> {after}")
print(f"\n{changed} messages changed, 0 new requests")

# %% [markdown]
# ## Before you use it on a live channel
#
# - **Each message waits for its answer.** Requests in this cookbook took a median of about
#   0.1 seconds, so a new line appears in chat before its answer comes back. Either hide
#   each new line until its answer arrives, or accept that a toxic message is on screen
#   for that long.
# - **One request at a time can't keep up with a busy chat.** Run several requests at once,
#   as in *Filter the whole chat*.
# - **A winning category can still be uncertain.** Here we always apply the winning label's
#   rule. Before using it live, choose how to handle messages whose top two category
#   probabilities are close to each other, and Noul values near 0.5. You could flag those
#   messages for review or leave them visible rather than hide them automatically. Test that
#   policy on your own channel's chat.
# - **The spoiler check has no record of the streamer's progress.** It uses the game title
#   and the model's knowledge of the game to flag likely plot revelations. It cannot tell
#   whether this viewer or the streamer already knows them. To filter only unseen plot
#   points, supply the current progress and relevant plot information, then adapt the
#   question to use that context.
#
# ## Open it in the playground
#
# The link below opens the playground with the state for the backseat message and the three
# questions filled in. Run the call live there, or edit the questions and run it again.

# %%
playground_link = make_playground_link(
    {"stream": STREAM, "message": {"user": CHAT[4][0], "text": CHAT[4][1]}},
    QUESTIONS,
)
# Print a usable URL in a script; render a clickable link in a notebook.
if __name__ == "__main__" and "__file__" in globals():
    print(f"Open in the TypeSafe playground:\n{playground_link}")
else:
    display(Markdown(f"🔗 [Open in the TypeSafe playground]({playground_link})"))
