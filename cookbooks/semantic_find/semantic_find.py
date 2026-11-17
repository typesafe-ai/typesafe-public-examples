# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "cooksafe>=0.3.0,<0.4.0"]
# ///

# ruff: noqa: E402

# %% [markdown]
# # Line-by-line Search
#
# You have GitHub's Terms of Service and a plain-language question about it. You need the
# lines that answer the question and a way to detect when the document has no answer. You
# end up with `find()`, which returns the `exists` probability and one relevance score per
# line.
#
# ![A query scans a document and reveals an answer attached to the matching
# line](semantic-find-hero.gif)
#
# The search has three parts:
#
# 1. Tag each line with an ID so Jev can point to it.
# 2. Use a `Choice` question to rank those line IDs by how well they answer the query.
# 3. In the same request, use a `Noul` question to check whether the document contains an
#    answer at all.

# %% [markdown]
# ## Setup
#
# ### Install the dependencies
#
# ```bash
# pip install ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, [create a TypeSafe API key](https://console.typesafe.ai/keys) and set
# `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.
#
# ### Run the script
#
# [`semantic_find.py`](https://github.com/typesafe-ai/typesafe-public-examples/blob/main/cookbooks/semantic_find/semantic_find.py)
# holds every code block on this page, in order. Run it from the file's directory:
#
# ```bash
# python semantic_find.py
# ```
#
# The first block imports `typesafe_sdk`, reads `TYPESAFE_API_KEY` and creates the client:

# %%
import os
import urllib.request
from pathlib import Path

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
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"), timeout=120.0
)

# %%
EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/semantic_find"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("semantic_find_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## Step 1: tag every line with an ID
#
# The test document is GitHub's Terms of Service, split into 218 clauses, so every search
# result points to one quotable line.

# %%
DOCUMENT_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/cookbooks/semantic_find/github-tos.txt"
)


@json_cache
def fetch_document(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "typesafe-cookbook/1.0"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode()


LINES = fetch_document(DOCUMENT_URL).splitlines()

# %% [markdown]
# The cache prevents repeated downloads.
#
# Now prefix each line with a short ID and join the lines back into one document. The model
# uses these IDs to point to its answer.


# %%
def line_id(i: int) -> str:
    return f"L{i:03d}"


DOCUMENT = "\n".join(f"{line_id(i)}| {line}" for i, line in enumerate(LINES))

# %% [markdown]
# `DOCUMENT` now looks like this:
#
# ```
# L052| You own Your Content. If you post Content you did not create, you are responsible for...
# L053| You grant us and other Users the licenses in Sections D.4–D.8. These licenses apply...
# L054| 4. License Grant to Us
# ```
#
# ## Step 2: ask where the answer is
#
# A `Choice` question returns a probability for every option. Use the line IDs as the
# options.


# %%
def where_question(query: str) -> Choice:
    return Choice(
        instructions=f'Which line of the document contains the answer to: "{query}"?',
        criteria={line_id(i): None for i in range(len(LINES))},
    )


# %% [markdown]
# The option descriptions are `None` because the document already contains the text for each
# ID. The query goes in `instructions`; the state stays unchanged between searches.
#
# <Info> A `Choice` question accepts up to 255 options, so this recipe searches documents of
# up to 255 lines in one request. Past that, search in two passes: one Choice question picks
# a window
# of lines, and a second ranks the lines inside it. </Info>
#
# ## Step 3: check whether an answer exists
#
# Choice probabilities always add up to 1, so some line ranks first even when the document
# doesn't answer the question.


# %%
def exists_question(query: str) -> Noul:
    return Noul(
        instructions=f'Does any line of the document address or answer: "{query}"?',
        criteria=NoulCriteria(
            true="At least one line of the document states or directly implies the answer",
            false="No line of the document addresses this",
        ),
    )


# %% [markdown]
# Unlike the Choice probabilities, the Noul probability doesn't depend on the other options,
# so it can fall near zero when the document has no answer.
#
# ## Step 4: send both questions in one request
#
# Both questions go in one `system_one` call, so the document is sent once.
#
# ```mermaid
# flowchart LR
#     doc["tagged document<br/><i>lines with IDs</i>"]
#     query["user question"]
#
#     subgraph request["one TypeSafe request · both questions see the full document"]
#         direction TB
#         shared["shared document<br/>and user question"]
#         choice["<b>one Choice question</b><br/>which line answers<br/>the question?"]
#         noul["<b>one Noul question</b><br/>does the document contain<br/>an answer at all?"]
#         relevance["probability<br/>for every line ID"]
#         exists["exists noul<br/>0 to 1"]
#         shared --> choice
#         shared --> noul
#         choice --> relevance
#         noul --> exists
#     end
#
#     rank["sort lines by<br/>Choice probability"]
#     verdict["apply verdict<br/>answered · partial · absent"]
#     out["ranked evidence<br/>and document verdict"]
#
#     doc --> shared
#     query --> shared
#     relevance --> rank
#     exists --> verdict
#     rank --> out
#     verdict --> out
# ```


# %%
@json_cache
def _find(
    state: str,
    where: Choice,
    exists: Noul,
    model: str,
) -> dict:
    response = client.system_one(
        state=state,
        questions={"where": where, "exists": exists},
        model=model,
    )
    probabilities = response.answers["where"].probabilities
    return {
        "exists": response.answers["exists"].noul,
        "relevance": [probabilities.get(line_id(i), 0.0) for i in range(len(LINES))],
    }


def find(query: str) -> dict:
    return _find(
        DOCUMENT,
        where_question(query),
        exists_question(query),
        model=TYPESAFE_MODEL,
    )


# %% [markdown]
# The `relevance` list keeps one score per line, in document order.
#
# ## Step 5: read the result
#
# The `exists` probability becomes one of three verdicts, the middle one for partial
# answers. The ranking prints as a bar chart.


# %%
FOUND, ABSENT = 0.7, 0.35  # `exists` cutoffs for answered and not in this document


def verdict(exists: float) -> str:
    if exists >= FOUND:
        return "answered in this document"
    return "not in this document" if exists < ABSENT else "partially addressed"


def show(query: str, top: int = 4) -> dict:
    result = find(query)
    print(f'"{query}"')
    print(f"  exists {result['exists']:.2f} -> {verdict(result['exists'])}")
    ranked = sorted(
        range(len(LINES)), key=lambda i: result["relevance"][i], reverse=True
    )
    for i in ranked[:top]:
        bar = "#" * max(1, round(result["relevance"][i] * 12))
        preview = LINES[i][:58].rstrip()
        print(f"  {line_id(i)}  {result['relevance'][i]:.2f}  {bar:<12}  {preview}")
    return result


# %% [markdown]
# Tune these thresholds on your own documents before using them in production.
#
# ## Step 6: run the search
#
# Ask two questions that have direct answers, one that has a partial answer, and one that has
# no answer.

# %%
print(f"{len(LINES)} lines, {len(DOCUMENT):,} characters\n")
show("who owns the code I upload?")
print()
show("can GitHub kick me off the platform without warning?")
print()
show("do I have to take disputes to arbitration?", top=2)
print()
show("can I join a class action lawsuit against GitHub?", top=2)

# %% [markdown]
# ## What the scores mean
#
# The first two queries return direct answers and the source lines needed to verify them.
#
# - **Arbitration:** The Terms never mention arbitration. The top line, at 0.80, is the
#   governing-law clause, which sends disputes to the courts in San Francisco. `exists` is
#   0.65, so the result is **partially addressed**.
# - **Class action:** The Terms never mention class actions. The ranking still puts the
#   governing-law clause first, at 0.23, because a ranking always has a top line. `exists`
#   is 0.05, so the result is **not in this document**.
#
# The ranking tells you where to look; the `exists` score tells you whether the result
# answers the question.
#
# To search your own document, swap `DOCUMENT_URL`; every other line of the script works off
# `LINES`.
#
# ## Open it in the playground
#
# The link holds the tagged contract and both questions for "who owns the code I upload?".

# %%
QUERY = "who owns the code I upload?"
playground_link = make_playground_link(
    DOCUMENT, {"where": where_question(QUERY), "exists": exists_question(QUERY)}
)
display(Markdown(f"🔗 [Open in the TypeSafe playground]({playground_link})"))
