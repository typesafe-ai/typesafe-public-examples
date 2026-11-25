# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "matplotlib",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Classification using confidence
#
# Every company that files an annual report with the SEC describes its own business in it. We
# classify those descriptions under the Standard Industrial Classification: 75 industry
# groups, one `Choice` question per document.
#
# Many filings are easy. A regional bank is a regional bank. Some are not: a company that
# just sold one of its two segments, or a startup describing a business it plans to enter
# rather than one it runs. The model has to pick a group regardless, and the answer for a
# hard case looks no different from the answer for an easy one. Telling hard cases from easy
# ones is normally where the cost goes: a second model, extra calls, human review.
#
# A Choice already tells you. Alongside the winning option it returns `confidence`, high when
# nearly all the probability landed on one option and low when it spread across several. That
# one number separates the answers you can trust from the ones you can't.
#
# What to do with an untrusted answer depends on your labels. SIC labels form a hierarchy:
# industry groups roll up into broader divisions. When the model is unsure of the group,
# report the division it belongs to. The division follows from the group, so there is no
# second call.
#
# Across 60 filings, a confidence cutoff of 0.9 marks 28 as confident and 32 as not. The
# confident 28 are right 93% of the time; the other 32, 50%. Reported one level up, that 50%
# becomes 78%.
#
# ```mermaid
# flowchart LR
#     doc["Item 1 'Business'<br/>from one 10-K"]
#
#     subgraph request["one request"]
#         q["Choice<br/>75 industry groups"]
#     end
#
#     sure{"confidence<br/>&ge; 0.9?"}
#     grp["report the industry group<br/><i>e.g. 28</i>"]
#     div["report its division<br/><i>e.g. manufacturing</i>"]
#
#     doc --> request --> sure
#     %% both branches leave the test, so they share a rank and stack on their own
#     sure -- "yes" --> grp
#     sure -- "no" --> div
# ```

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython matplotlib 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run classification_using_confidence.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.
#
# Numbers below came from `jev-1.14.0` on 2026-11-03.

# %%
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
CONFIDENT = 0.9  # above this the group is reported; below it, the division

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
    # A run that replays the cache needs no key.
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)


EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/classification_using_confidence"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", path)
    return path


CACHE = Path("classification_using_confidence_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## Build the two levels of the taxonomy
#
# `sic_codes.tsv` is the industry list the SEC publishes for filers to pick their own code
# from, fetched 2026-08-10: 444 four-digit codes, each with an industry title. The digits are
# a hierarchy. The first two are the **major group** (75 of them here, from `01` agricultural
# production to `99` non-classifiable), and fixed ranges of major groups make up the ten
# **divisions**, the broadest split SIC has.
#
# Both levels come out of that one file with no model involved: group the codes by their
# first two digits, then map those digits to a division.

# %%
DIVISIONS = [
    (1, 9, "agriculture, forestry and fishing"),
    (10, 14, "mining"),
    (15, 17, "construction"),
    (20, 39, "manufacturing"),
    (40, 49, "transportation, communications and utilities"),
    (50, 51, "wholesale trade"),
    (52, 59, "retail trade"),
    (60, 67, "finance, insurance and real estate"),
    (70, 89, "services"),
    (91, 99, "public administration"),
]

INDUSTRIES: dict[str, str] = {}
for line in download("sic_codes.tsv").read_text().splitlines()[1:]:
    code, _office, title = line.split("\t")
    INDUSTRIES[code] = title.lower()

GROUPS: dict[str, list[str]] = defaultdict(list)
for code in sorted(INDUSTRIES):
    GROUPS[code[:2]].append(code)


def division(group: str) -> str:
    number = int(group)
    return next(name for low, high, name in DIVISIONS if low <= number <= high)


print(
    f"{len(INDUSTRIES)} industries -> {len(GROUPS)} major groups -> {len(DIVISIONS)} divisions"
)
print(
    f"  group 35 = {division('35')} / {', '.join(INDUSTRIES[c] for c in GROUPS['35'][:3])} ..."
)

# %% [markdown]
# A Choice question needs something to describe each option, and a group's own name is not
# always
# there: 42 of the 75 carry an umbrella title in the SEC's list, and the rest carry none. So
# each group is described by the industries inside it, which is what someone reading the
# filing would match against anyway.

# %%
MAX_NAMED = (
    8  # industries listed per group; enough to characterise it without a wall of text
)


def describe(group: str) -> str:
    umbrella = INDUSTRIES.get(f"{group}00")
    inside = [INDUSTRIES[c] for c in GROUPS[group] if c != f"{group}00"][:MAX_NAMED]
    listed = "; ".join(inside)
    return (
        f"{umbrella} — includes: {listed}"
        if umbrella and listed
        else (umbrella or listed)
    )


print(f"group 20: {describe('20')[:150]}")
print(f"\ngroup 65: {describe('65')[:150]}")

# %% [markdown]
# ## The filings
#
# `filings.jsonl` holds 60 annual reports (10-K), each trimmed to Item 1 "Business", the
# section where a company describes what it does, which is the only part an industry code is
# about. They span 1993–2024 and run from 700 to 2,200 words. Each one carries the SIC code
# its filer chose, plus the accession number to look it up on EDGAR.
#
# The label is self-reported: whoever prepared the filing picked it once, and it goes stale
# when a company sells the business the code names and keeps the code. These 60 were
# filtered down to filings whose own text supports the code they carry, so the numbers here
# measure the recipe rather than the state of EDGAR's metadata.

# %%
FILINGS = [json.loads(line) for line in download("filings.jsonl").read_text().splitlines()]
example = FILINGS[7]
print(
    f"{len(FILINGS)} filings, {sum(f['words'] for f in FILINGS) // len(FILINGS)} words on average"
)
print(f"\n{example['id']} (filed {example['year']}, accession {example['accession']}):")
print(f"  {example['text'][:230]}...")
print(f"  filer's code: {example['sic']} {INDUSTRIES[example['sic']]}")

# %% [markdown]
# ## Ask one Choice question, and read the confidence
#
# The whole taxonomy fits in one request: a Choice works reliably up to roughly 240 options,
# and 75 is well inside that.
#
# The answer comes back with `choice`, the winning group; `probabilities`, the weight on each
# of the 75; and `confidence`, which says how concentrated that spread was. The recipe reads
# `confidence` rather than the winner's own probability. A winner at 0.45 with a runner-up at
# 0.44, and a winner at 0.45 with the rest of the weight scattered thinly, are different
# situations, and `confidence` is what separates them.

# %%
QUESTION = (
    "Which broad industry does this company operate in? Judge the company's own operations "
    "as this filing describes them."
)


def questions() -> dict:
    return {
        "group": Choice(
            instructions=QUESTION,
            criteria={group: describe(group) for group in sorted(GROUPS)},
        )
    }


@json_cache
def ask(text: str, group: Choice, model: str) -> dict:
    response = client.system_one(
        state=text, questions={"group": group}, model=model
    )
    answer = response.answers["group"]
    return {
        "group": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
    }


# %% [markdown]
# ## Return the group when sure, its division when not
#
# At 0.9 confidence or above, the answer is reported as an industry group; below that, the
# same answer is reported as the division that group sits in.
#
# Every filing still comes back with a usable label. One the model could not classify
# confidently comes back one level up instead of being dropped or sent on. If a division is
# too coarse for your application to act on, this branch is where you hand it to a person.


# %%
def classify(filing: dict) -> dict:
    answer = ask(filing["text"], questions()["group"], model=TYPESAFE_MODEL)
    sure = answer["confidence"] >= CONFIDENT
    return {
        "level": "group" if sure else "division",
        "label": answer["group"] if sure else division(answer["group"]),
        "confidence": answer["confidence"],
        "group": answer["group"],
    }


def show(filing: dict) -> None:
    result = classify(filing)
    named = describe(result["group"]).split(" — ")[0][:46]
    print(
        f"  {filing['id']:>13}  conf {result['confidence']:.2f}  -> {result['level']:<8} "
        f"{result['label']:<14} (group {result['group']}: {named})"
    )


print("three filings the model was sure about:")
for f in sorted(FILINGS, key=lambda f: -classify(f)["confidence"])[:3]:
    show(f)
print("\nthree it was not:")
for f in sorted(FILINGS, key=lambda f: classify(f)["confidence"])[:3]:
    show(f)

# %% [markdown]
# The confidences line up with how hard each filing is to classify. The top three, all at
# 1.00, are a marketing company, a pharmaceutical maker and a carpet manufacturer; all three
# have sold or left earlier businesses, but each filing names the one business it runs now
# outright. The three at the bottom are harder for reasons you can read in the text. Two are
# startups describing a business they have not begun (Nevaeh "intends to operate as a
# software developer", and Hotel Management Systems has a sole officer and writes of its
# "planned operations"), and the third had two segments and sold one of them weeks before
# filing. Those three come back as a division rather than a group.

# %% [markdown]
# ## What the broader answer buys
#
# We score all 60 filings against the code each filer chose, under both policies: name a
# group every time, or report the division whenever confidence lands under 0.9.


# %%
def correct(filing: dict, result: dict) -> bool:
    gold_group = filing["sic"][:2]
    if result["level"] == "group":
        return result["label"] == gold_group
    return result["label"] == division(gold_group)


results = [(f, classify(f)) for f in FILINGS]
sure = [(f, r) for f, r in results if r["level"] == "group"]
unsure = [(f, r) for f, r in results if r["level"] == "division"]

forced = sum(r["group"] == f["sic"][:2] for f, r in results)
broadened = sum(correct(f, r) for f, r in results)

print(f"forced to name a group every time      {forced}/{len(results)} right")
print(
    f"  of those, the {len(sure)} it was sure about  "
    f"{sum(r['group'] == f['sic'][:2] for f, r in sure)}/{len(sure)} right"
)
print(
    f"  and the {len(unsure)} it was not           "
    f"{sum(r['group'] == f['sic'][:2] for f, r in unsure)}/{len(unsure)} right"
)
print(
    f"\nletting it answer coarsely when unsure  {broadened}/{len(results)} useful answers"
)

# %% [markdown]
# Where the model was sure, the group it named is right nine times in ten. Where it was not,
# the group is right only half the time. Reporting those same answers as a division takes
# them to 78%.

# %%
labels = ["sure\n(group reported)", "unsure\n(division reported)"]
forced_split = [
    sum(r["group"] == f["sic"][:2] for f, r in sure) / len(sure),
    sum(r["group"] == f["sic"][:2] for f, r in unsure) / len(unsure),
]
broad_split = [
    sum(correct(f, r) for f, r in sure) / len(sure),
    sum(correct(f, r) for f, r in unsure) / len(unsure),
]

SPLIT_PALETTES = {
    "light": {"surface": "#ffffff", "ink": "#000000", "forced": "#c8ccd4", "broad": "#3b6ea5"},
    "dark": {"surface": "#0d0a0f", "ink": "#fafafa", "forced": "#5b5f66", "broad": "#7aa7e0"},
}


def draw_split(p: dict) -> plt.Figure:
    ink = {key: p["ink"] for key in ("text.color", "axes.labelcolor", "axes.edgecolor", "xtick.color", "ytick.color")}
    with matplotlib.rc_context(ink):
        fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=p["surface"])
        ax.set_facecolor(p["surface"])
        x = range(len(labels))
        ax.bar(
            [i - 0.19 for i in x],
            forced_split,
            0.38,
            label="always name a group",
            color=p["forced"],
        )
        ax.bar(
            [i + 0.19 for i in x],
            broad_split,
            0.38,
            label="answer broadly when unsure",
            color=p["broad"],
        )
        for i, (a, b) in enumerate(zip(forced_split, broad_split)):
            ax.text(i - 0.19, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
            ax.text(i + 0.19, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{lab}\nn={n}" for lab, n in zip(labels, [len(sure), len(unsure)])]
        )
        ax.set_ylabel("labels that are right")
        ax.set_ylim(0, 1.12)
        ax.set_title("Where the broader answer helps: the filings it was unsure about")
        ax.legend(frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig


# The docs build sets COOKSAFE_RENDER and also draws a dark copy.
for theme in ("light", "dark") if os.environ.get("COOKSAFE_RENDER") else ("light",):
    fig = draw_split(SPLIT_PALETTES[theme])
    display(fig, metadata={"theme": theme})
    plt.close(fig)

# %% [markdown]
# ## Open it in the playground
#
# This share link holds one filing and the 75-option question, so you can see the
# distribution and the confidence it produces without writing any code.

# %%
playground_link = make_playground_link(example["text"], questions())
display(
    Markdown(
        f"🔗 [Open in the TypeSafe playground]({playground_link})"
    )
)
