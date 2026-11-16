# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "typesafe-sdk>=0.7.0", "ipython", "matplotlib",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Infinite cardinality choice
#
# Someone remembers a film but not its title, and describes it instead: "a man and a boy
# running from these sharp metal eyeball things with blades on the side". They saw it in the
# 90s. The film is one of 5,000 in a catalog, and its plot summary tells the same story in
# different words.
#
# One `Choice` question can pick the film from a list, but it takes at most 255 options, and
# only about forty when each option is a long plot summary. So the search runs in three
# steps, and each step reads more about fewer films:
#
# 1. Ask one `Noul` question per film, "is this the film they mean?", with the first hundred
#    words of its plot. The questions go out in chunks, all at once, so the wait stays about
#    the same as the catalog grows, as long as your rate limit allows that many calls at a
#    time.
# 2. Ask again about the eighty films with the highest probabilities, this time with six
#    hundred words of plot.
# 3. Put the top twenty in one `Choice` question. Its top probability tells you whether to
#    show one film or a short list.
#
# The 100 requests here come from a forum where people ask for help naming a film, each
# answered by someone who recognised it. When embedding search picks the twenty films for
# step 3 instead, the right film comes first for 33% of them. With all three steps, it
# comes first for 78%. Each of the 48 times the Choice question gave its top film 0.8 or
# more, that film was right.
#
# The result is a `find()` function that takes a request and returns the films ranked by
# probability, in three rounds of parallel calls.
#
# ```mermaid
# flowchart LR
#     ask["a half-remembered film<br/>in the user's words"]
#
#     subgraph filter["round 1 &mdash; filter"]
#         nouls["one Noul question per film<br/>short plot, in parallel chunks"]
#     end
#
#     subgraph reread["round 2 &mdash; re-read the top 80"]
#         long["one Noul question per film<br/>long plot, in parallel chunks"]
#     end
#
#     %% direction TB inside an LR chart stacks the Choice question over its answer
#     subgraph decide["round 3 &mdash; decide"]
#         direction TB
#         choice["one Choice question<br/>over the top 20"]
#         out["top probability picks<br/>one film or a short list"]
#         choice --> out
#     end
#
#     ask --> filter --> reread --> decide
# ```

# %% [markdown]
# ## Setup
#
# ```bash
# pip install openai ipython matplotlib 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# The code downloads the published answers into `infinite_cardinality_choice_cache.json`, so
# it runs as published without an API key; to run live, set `DOWNLOAD_CACHE = False` and
# delete that file. To change a request, set `TYPESAFE_API_KEY`, and `OPENAI_API_KEY` for the
# embedding search the results are measured against; in Colab, add them in the Secrets panel.
#
# Numbers below came from `jev-1.14.0`, run in November 2026.

# %%
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
CHOICE_LIMIT = 255  # enforced by the API; 256 options is rejected outright
CHUNK = 300  # films per filter call: ~150 tokens each keeps a call well under 64k
LONG_CHUNK = 40  # films per re-read call: ~700 tokens each, ~28k a call
FIRST_CUT = 80  # films read again with their long plot
CANDIDATES = 20  # films the Choice question sees
WORKERS = 4  # searches run side by side, each sending all its chunks at once

# The keys this cookbook uses. In Colab, keys saved in the Secrets panel are copied in here.
KEYS = ["TYPESAFE_API_KEY", "OPENAI_API_KEY"]
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
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless kernels replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=300.0,  # 300 questions is a big request
    # a request this size occasionally draws an overload response, so retries are not optional
    retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=30.0),
)


EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/infinite_cardinality_choice"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", path)
    return path


CACHE = Path("infinite_cardinality_choice_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## Load the catalog and the requests
#
# The catalog is `films.jsonl`: each film's title, year, and two versions of the plot section
# of its Wikipedia article, fetched 2026-11-07. The short one, `plot`, is the first hundred
# or so words, and step 1 reads it. The long one, `long_plot`, is the first six hundred or
# so, and steps 2 and 3 read it. The catalog holds the film behind every request. The other
# films, the ones a search has to rule out, are those whose Wikipedia articles exist in the
# most languages.
#
# The 100 requests in `requests.jsonl` come from Microsoft's tip-of-the-tongue dataset,
# collected from the forum iRememberThisMovie.com, and each comes with the film that answered
# it. They are a random sample of the 709 whose film has a plot section on Wikipedia.

# %%
FILMS = {f["id"]: f for f in map(json.loads, download("films.jsonl").read_text().splitlines())}
REQUESTS = [json.loads(line) for line in download("requests.jsonl").read_text().splitlines()]


def label(film_id: str) -> str:
    film = FILMS[film_id]
    return f"{film['title']} ({film['year']})" if film["year"] else film["title"]


example = next(r for r in REQUESTS if r["id"] == "388")
print(f"{len(FILMS):,} films, {len(REQUESTS)} requests\n")
print(f"request {example['id']}:\n  {example['text']}\n")
print(f"the film: {label(example['film'])}\n  {FILMS[example['film']]['plot']}")

# %% [markdown]
# ## Step 1: filter with one Noul question per film
#
# Each Noul question names one film as the answer and includes its plot, so Jev compares the
# request with the plot instead of recalling the film from its title.
#
# A call holds at most 64k tokens, so split the catalog into chunks that fit. No chunk
# depends on another, so send them all at once.

# %%
def proposition(film_id: str, plot: str = "plot") -> str:
    return (
        f"The film this person is trying to remember is {label(film_id)}. "
        f"Its plot: {FILMS[film_id][plot]}"
    )


def as_json(questions: dict) -> str:
    """The questions as API JSON, so the cache key changes when their wording does."""
    return json.dumps(
        {key: q.model_dump(mode="json", exclude_none=True) for key, q in questions.items()}
    )


IDS = sorted(FILMS)
CHUNKS = [
    as_json({i: Noul(instructions=proposition(i)) for i in IDS[start : start + CHUNK]})
    for start in range(0, len(IDS), CHUNK)
]


@json_cache
def filter_chunk(text: str, questions_json: str, model: str) -> dict:
    started = perf_counter()
    response = client.system_one(
        state=text, questions=json.loads(questions_json), model=model
    )
    return {
        "p": {key: answer.noul for key, answer in response.answers.items()},
        "seconds": round(perf_counter() - started, 2),
    }


def filter_films(text: str) -> dict:
    """Every chunk at once; one probability per film."""
    with ThreadPoolExecutor(max_workers=len(CHUNKS)) as pool:
        parts = list(
            pool.map(lambda chunk: filter_chunk(text, chunk, model=TYPESAFE_MODEL), CHUNKS)
        )
    return {key: p for part in parts for key, p in part["p"].items()}


def by_probability(request: dict) -> list[str]:
    p = filter_films(request["text"])
    return sorted(p, key=lambda key: (-p[key], key))


filtered = filter_films(example["text"])
seconds = [
    filter_chunk(example["text"], chunk, model=TYPESAFE_MODEL)["seconds"] for chunk in CHUNKS
]
print(f"{len(filtered):,} answers from {len(CHUNKS)} calls sent together; the slowest "
      f"took {max(seconds)}s\n")
print("the filter's own top five:")
for key in by_probability(example)[:5]:
    mark = "  <- the film" if key == example["film"] else ""
    print(f"  {filtered[key]:.2f}  {label(key)}{mark}")
print(f"\nthe film is at position {by_probability(example).index(example['film']) + 1}")
held = sum(1 for r in REQUESTS if r["film"] in by_probability(r)[:FIRST_CUT])
print(f"the top {FIRST_CUT} hold the right film for {held}/{len(REQUESTS)} requests")

# %% [markdown]
# ## Step 2: read the top eighty again, with the long plot
#
# For this request, the filter ranks *Ghostbusters: Afterlife* first at 0.64. It matches on
# a mother, two kids, a haunted house and money trouble. *The Diabolical* is sixth. The
# request's most specific detail is that the ghosts "turn out to be time travelers". In the
# plot of *The Diabolical*, that twist comes after the first hundred words, so the filter
# never sees it. People remember scenes from anywhere in a film.
#
# So ask again about the filter's top eighty, with six hundred words of each plot. Longer
# questions mean fewer films per chunk, and the chunks still go out together. The filter's
# top eighty hold the right film for 93 of the 100 requests, and this step has to move it
# into the top twenty.

# %%
def reread(request: dict) -> dict:
    """The filter's top films, asked again with the long plot; one probability per film."""
    first = by_probability(request)[:FIRST_CUT]
    chunks = [
        as_json({i: Noul(instructions=proposition(i, "long_plot"))
                 for i in first[start : start + LONG_CHUNK]})
        for start in range(0, FIRST_CUT, LONG_CHUNK)
    ]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        parts = list(
            pool.map(
                lambda chunk: filter_chunk(request["text"], chunk, model=TYPESAFE_MODEL), chunks
            )
        )
    return {key: p for part in parts for key, p in part["p"].items()}


def candidates(request: dict) -> tuple[str, ...]:
    p = reread(request)
    return tuple(sorted(p, key=lambda key: (-p[key], key))[:CANDIDATES])


reread_p = reread(example)
print("the long plots' top five:")
for key in candidates(example)[:5]:
    mark = "  <- the film" if key == example["film"] else ""
    print(f"  {reread_p[key]:.2f}  {label(key)}{mark}")
held = sum(1 for r in REQUESTS if r["film"] in candidates(r))
print(f"\nthe top {CANDIDATES} hold the right film for {held}/{len(REQUESTS)} requests")

# %% [markdown]
# The long plot includes the twist: the apparition is the son, come back from the future.
# *The Diabolical* moves to first at 0.60. Across all requests, the top twenty now hold the
# right film for 88, against 82 for the filter's top twenty alone.
#
# ## Step 3: decide with one Choice question
#
# Put the top twenty in one `Choice` question, with their long plots as the options. Each
# Noul question judges its film alone, so two films that both fit the request can both get
# 0.4 or more, here 0.60 and 0.40, and the top five alone add up to more than 1. A Choice
# question weighs the twenty films against each other, so its top probability says how far
# ahead of the rest the top film is.

# %%
DECISION = "Which film is this person trying to remember?"


def decision_question(options: tuple[str, ...]) -> dict:
    return {
        "film": Choice(
            instructions=DECISION,
            criteria={key: f"{label(key)}: {FILMS[key]['long_plot']}" for key in options},
        )
    }


@json_cache
def rank(text: str, questions_json: str, model: str) -> dict:
    response = client.system_one(
        state=text, questions=json.loads(questions_json), model=model
    )
    answer = response.answers["film"]
    return {"film": answer.choice, "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities)}


def find(request: dict, options: tuple[str, ...] | None = None) -> dict:
    """Rank a shortlist with one Choice; by default, the top twenty on the long plots."""
    options = candidates(request) if options is None else options
    ranked = rank(request["text"], as_json(decision_question(options)), model=TYPESAFE_MODEL)
    p = ranked["probabilities"]
    return {"film": ranked["film"], "order": sorted(p, key=lambda key: (-p[key], key)),
            "confidence": ranked["confidence"], "probabilities": p}


result = find(example)
print("the Choice's own top five, out of the twenty it was given:")
for key in result["order"][:5]:
    mark = "  <- the film" if key == example["film"] else ""
    print(f"  {result['probabilities'][key]:.2f}  {label(key)}{mark}")

# %% [markdown]
# The Choice question puts 1.00 on *The Diabolical*.
#
# To search your own catalog, give each item a short and a long description and rewrite
# `FILMS`, `proposition()` and `decision_question()`. Nothing else mentions films.
#
# ## Measure what each step is worth
#
# The `embeddings` column uses embedding search, the standard first try for matching text
# that says the same thing in other words. It embeds each film's short plot and the request
# with OpenAI's `text-embedding-3-small`, and the twenty closest films go to the same Choice
# question. The `no re-read` column sends the filter's top twenty straight to
# the Choice question, skipping step 2.

# %%
import numpy as np  # noqa: E402
from openai import OpenAI  # noqa: E402

EMBED_MODEL = "text-embedding-3-small"
embedder = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "cache-only"))


def embed(texts: list[str], embed_model: str) -> np.ndarray:
    """Unit-length embeddings, 256 texts per call."""
    vectors = []
    for start in range(0, len(texts), 256):
        response = embedder.embeddings.create(model=embed_model, input=texts[start : start + 256])
        vectors += [item.embedding for item in response.data]
    vectors = np.array(vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


catalog_vectors = None  # the catalog is embedded only when a shortlist isn't cached yet


@json_cache
def embedding_shortlist(text: str, embed_model: str) -> list[str]:
    """The twenty films whose short plots sit closest to the request."""
    global catalog_vectors
    if catalog_vectors is None:
        catalog_vectors = embed([f"{label(i)}: {FILMS[i]['plot']}" for i in IDS], embed_model)
    similarity = catalog_vectors @ embed([text], embed_model)[0]
    order = sorted(range(len(IDS)), key=lambda n: (-similarity[n], IDS[n]))
    return [IDS[n] for n in order[:CANDIDATES]]


shortlists = {
    "embeddings": {
        r["id"]: tuple(embedding_shortlist(r["text"], embed_model=EMBED_MODEL)) for r in REQUESTS
    },
    "no re-read": {r["id"]: tuple(by_probability(r)[:CANDIDATES]) for r in REQUESTS},
    "all 3 steps": {r["id"]: candidates(r) for r in REQUESTS},
}

def position(film: str, order: list[str]) -> int:
    return order.index(film) + 1 if film in order else 10_000


found = {}
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for name, lists in shortlists.items():
        found[name] = list(pool.map(lambda r: find(r, lists[r["id"]]), REQUESTS))
via_long = found["all 3 steps"]


def share(values) -> str:
    return f"{sum(values) / len(REQUESTS):.0%}"


print(f"  {'':<20}" + "".join(f"{name:>14}" for name in shortlists))
print(f"  {'reaches the Choice':<20}" + "".join(
    f"{share([r['film'] in lists[r['id']] for r in REQUESTS]):>14}"
    for lists in shortlists.values()))
for k in (1, 3):
    print(f"  {'film in the top ' + str(k):<20}" + "".join(
        f"{share([position(r['film'], f['order']) <= k for r, f in zip(REQUESTS, picks)]):>14}"
        for picks in found.values()))

# %% [markdown]
# Embedding search holds the right film in its top twenty for 35% of requests, and the
# Choice question puts it first for 33%: the Choice question can only rank the films it is
# given. Reading short plots against the request gets the film into the top twenty for 82%
# of requests, and re-reading the top eighty with long plots raises that to 88%. The right
# film comes first for 73% without step 2 and 78% with it.
#
# ## Decide when to show one film
#
# To decide between showing one film and showing a short list, you need a probability you
# can set a threshold on. Both the re-read and the Choice question give their top film a
# probability, and the table counts how often that film was right at each level.

# %%
BANDS = ((0.0, 0.5, "under 0.5"), (0.5, 0.8, "0.5 to 0.8"), (0.8, 0.95, "0.8 to 0.95"),
         (0.95, 1.01, "0.95 and up"))
SOURCES = {
    "long-plot Nouls": [reread(r) for r in REQUESTS],
    "Choice question": [f["probabilities"] for f in via_long],
}


def top_answer(p: dict) -> tuple[str, float]:
    key = min(p, key=lambda k: (-p[k], k))
    return key, p[key]


bands = {}
for name, dists in SOURCES.items():
    bands[name] = []
    for low, high, _ in BANDS:
        hits_ = [key == r["film"] for r, p in zip(REQUESTS, dists)
                 for key, top in [top_answer(p)] if low <= top < high]
        bands[name].append((len(hits_), sum(hits_)))
print(f"  {'top answer':<14}" + "".join(f"{name:>24}" for name in SOURCES))
for i, (_, _, band) in enumerate(BANDS):
    cells = "".join(
        f"{f'{right}/{n} right':>24}" for n, right in (bands[name][i] for name in SOURCES)
    )
    print(f"  {band:<14}{cells}")

THRESHOLD = 0.8
sure = [(r, f) for r, f in zip(REQUESTS, via_long) if f["confidence"] >= THRESHOLD]
rest = [(r, f) for r, f in zip(REQUESTS, via_long) if f["confidence"] < THRESHOLD]
print(f"\nChoice at {THRESHOLD} or more: {len(sure)} requests, "
      f"right for {sum(f['film'] == r['film'] for r, f in sure)}")
print(f"below {THRESHOLD}: {len(rest)} requests, the film in the top 3 for "
      f"{sum(position(r['film'], f['order']) <= 3 for r, f in rest)}")

# %%
BAR_PALETTES = {
    "light": {"surface": "#ffffff", "ink": "#000000", "note": "#5f6670", "grid": "#eef1f4",
              "bars": {"long-plot Nouls": "#8fb4d9", "Choice question": "#1b4a7a"}},
    "dark": {"surface": "#0d0a0f", "ink": "#fafafa", "note": "#9aa0a8", "grid": "#1f1b22",
             "bars": {"long-plot Nouls": "#3f6d9e", "Choice question": "#b4d2f2"}},
}


def draw_bands(p: dict) -> plt.Figure:
    ink = {key: p["ink"] for key in ("text.color", "axes.labelcolor", "axes.edgecolor",
                                     "xtick.color", "ytick.color")}
    with matplotlib.rc_context(ink):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=p["surface"])
        ax.set_facecolor(p["surface"])
        width = 0.36
        for j, name in enumerate(SOURCES):
            xs = [i + (j - 0.5) * (width + 0.03) for i in range(len(BANDS))]
            rates = [right / n if n else 0 for n, right in bands[name]]
            ax.bar(xs, rates, width=width, color=p["bars"][name], label=name, zorder=2)
            for x, rate, (n, _) in zip(xs, rates, bands[name]):
                ax.text(x, rate + 0.02, f"{n}", ha="center", fontsize=8.5, color=p["note"])
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([band for _, _, band in BANDS])
        ax.set_xlabel("probability of the top answer (number of requests above each bar)")
        ax.set_ylabel("share of top answers that are right")
        ax.set_ylim(0, 1.1)
        ax.set_title(CHART_TITLE)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax.grid(axis="y", color=p["grid"], linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig


CHART_TITLE = "Every Choice answer at 0.8 or more was right"
# The docs build sets COOKSAFE_RENDER and also draws a dark copy.
for theme in ("light", "dark") if os.environ.get("COOKSAFE_RENDER") else ("light",):
    fig = draw_bands(BAR_PALETTES[theme])
    display(fig, metadata={"theme": theme})
    plt.close(fig)

# %% [markdown]
# The Choice question's top answer is at 0.8 or more for 48 requests, and right for all 48.
# Under 0.5, it is right 9 times in 30. The re-read's probability can't sort requests into
# sure and unsure: none of its answers reaches 0.95, only 11 reach 0.8, and its answers under
# 0.5 are still right 30 times in 49.
#
# So decide with the Choice question's top probability. At 0.8 or more, show the one film.
# Below 0.8, show the top three, which hold the right film for 34 of the other 52 requests.
#
# ## Open it in the playground
#
# The playground can't comfortably show thousands of questions, so this link holds step 3
# only: the request, and the Choice question over the twenty films the re-read kept.

# %%
deeplink = make_playground_link(example["text"], decision_question(candidates(example)))
display(Markdown(f"🔗 [Open in the TypeSafe playground]({deeplink})"))
