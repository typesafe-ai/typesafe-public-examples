# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Double-checking citations
#
# An LLM answers a question and attaches citations: for each claim, a section of a source
# document and the quote it rests on. Some of those citations are wrong or hallucinated:
# the quote can be missing from the document altogether, or sit in it word for word while
# its context says the opposite of the claim.
#
# Checking one by hand is slow: find the document, find the quote inside it, then read
# enough of its context to tell whether it backs the claim up.
#
# To automate that check, we first look for missing quotes with an ordinary string match,
# and then we use a `Choice` question to read each surviving quote's context and decide
# whether it supports the claim.
#
# ```mermaid
#   %%{init: {"flowchart": {"wrappingWidth": 330}}}%%
# flowchart LR
#     cite["source document + citation"]
#
#     match{"is the quote<br/>in the source?"}
#     fab["mark <b>fabricated</b>"]
#
#     subgraph request[" "]
#         q["Choice &mdash; how does the<br/>section relate to the claim?<br/>supports &rarr; mark <b>verified</b><br/>contradicts &rarr; mark <b>contradicted</b><br/>says nothing &rarr; mark <b>unsupported</b>"]
#     end
#
#     check{"confidence<br/>&ge; 0.8?"}
#     stand["let the verdict stand"]
#     review["a human confirms it"]
#
#     cite --> match
#     %% the two edges that reach the call come first, so they stay adjacent; the
#     %% string match's own verdict is declared last and lands below them
#     match -- "found" --> request
#     match -- "no quote" --> request
#     match -- "not found" --> fab
#     request --> check
#     check --> stand
#     check --> review
# ```
#
# Below, eight citations from an LLM's answer about RFC 7519 (JSON Web Token) go through the
# check. The four accurate ones came back `verified` at confidence 0.98 or higher. All four
# planted failures were caught: a fabricated quote, a contradicted claim, and two unsupported
# citations, one of them sent to a human.
#
# `check_citation()`, the function you build here, takes a source document and one citation
# and returns one of four verdicts: `verified`, `unsupported`, `contradicted`, or
# `fabricated`. It also returns a confidence that flags the ones a human should look at.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.
#
# Numbers below came from `jev-1.14.0` on 2026-11-03.

# %%
import json
import os
import re
import urllib.request
from pathlib import Path
from time import perf_counter

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
AUTO_ACCEPT = 0.8  # start high for more human review as you build trust in the model

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
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)


EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/citation_check"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", path)
    return path


CACHE = Path("citation_check_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## Load the source and the citations
#
# The source is [RFC 7519](https://www.rfc-editor.org/rfc/rfc7519.html) (JSON Web Token),
# fetched from rfc-editor.org and committed next to this cookbook as `rfc7519.txt`. The code
# below strips the page headers and footers, then splits the text into numbered sections.
#
# The eight citations in `citations.json` were written by an LLM against the RFC. Four are
# accurate; we edited the other four to fail the check.


# %%
def load_source() -> str:
    """RFC 7519 verbatim, minus the page headers and footers that interrupt its paragraphs."""
    lines = []
    for line in download("rfc7519.txt").read_text().splitlines():
        bare = line.lstrip("\f")
        if re.match(r"Jones, et al\.\s.*\[Page \d+\]$", bare):
            continue
        if re.match(r"RFC 7519\s+JSON Web Token \(JWT\)\s+May 2015$", bare):
            continue
        lines.append(bare)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def split_sections(source: str) -> dict[str, str]:
    """Map each numbered section ("4.1.3") to its text, split on the RFC's header lines."""
    boundary = re.compile(r"(?m)^(?:(\d+(?:\.\d+)*)\.  .+|Appendix [A-Z]\..*)$")
    marks = list(boundary.finditer(source))
    sections = {}
    for mark, nxt in zip(marks, marks[1:] + [None]):
        if mark.group(1) is None:  # an appendix header only terminates the section before it
            continue
        sections[mark.group(1)] = source[mark.start() : nxt.start() if nxt else len(source)].strip()
    return sections


SOURCE = load_source()
SECTIONS = split_sections(SOURCE)
CITATIONS = json.loads(download("citations.json").read_text())

print(f"{len(SOURCE):,} characters, {len(SECTIONS)} numbered sections, {len(CITATIONS)} citations")
print("\nA citation with a quote:")
print(json.dumps(CITATIONS[1], indent=2))
print("\nA claim-only citation:")
print(json.dumps(next(c for c in CITATIONS if c["quote"] is None), indent=2))

# %% [markdown]
# ## Find each quote in the source
#
# A quote that is not in the source is fabricated, and no model is needed to find that out.
# Normalize whitespace and curly quotes so a quote still matches across the RFC's line
# wraps, then look for it as a substring. A match also says which section the quote came
# from, and that section is the text the model reads in the next step.
#
# A citation can name a section without quoting anything from it. There is nothing to match
# in that case, so take the section the citation names and go straight to the model.


# %%
def normalize(text: str) -> str:
    """Collapse whitespace and fold curly quotes, so a quote matches across line wraps."""
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    return re.sub(r"\s+", " ", text.translate(table)).strip()


def find_quote(sections: dict[str, str], quote: str) -> str | None:
    """The number of the section that contains the quote verbatim, or None."""
    needle = normalize(quote)
    for number in sorted(sections, key=lambda n: [int(p) for p in n.split(".")]):
        if needle in normalize(sections[number]):
            return number
    return None


def locate(sections: dict[str, str], citation: dict) -> tuple[str, str | None]:
    """Step 1 for one citation: a status, plus the section step 2 will read."""
    if citation["quote"] is None:
        return "section-only", sections[citation["section"]]
    number = find_quote(sections, citation["quote"])
    if number is None:
        return "missing", None
    return "found", sections[number]


for citation in CITATIONS:
    status, section = locate(SECTIONS, citation)
    where = f"section of {len(section):,} chars" if section else "not in the source"
    print(f"{citation['id']:<18}{status:<14}{where}")

# %% [markdown]
# ## Verify whether the source supports the claim
#
# A citation that still has a quote at this point matches the source word for word. That is
# not enough: the quote can be accurate and the claim built on top of it still wrong.
# Deciding that takes the quote's context, the section step 1 found.
#
# One `Choice` question per surviving citation covers the three ways a section can relate
# to a claim.
# The option with the highest probability is the verdict, and `AUTO_ACCEPT` (0.8 in the
# code above) decides what happens to it:
#
# - confidence at or above 0.8: the verdict stands on its own;
# - below 0.8: a human confirms the verdict before anything acts on it.
#
# Start high, and lower the threshold as you see how the model does on your own documents.

# %%
QUESTIONS = {
    "relation": Choice(
        instructions="How does the section relate to the claim?",
        criteria={
            "supports": "The section states the claim or directly implies that it is true",
            "contradicts": "The section states the opposite of the claim or implies it is false",
            "says_nothing": "The section does not address what the claim asserts, either way",
        },
    ),
}

RELATION_TO_VERDICT = {
    "supports": "verified",
    "contradicts": "contradicted",
    "says_nothing": "unsupported",
}


@json_cache
def ask(claim: str, section: str, relation: Choice, model: str) -> dict:
    started = perf_counter()
    response = client.system_one(
        state={"claim": claim, "section": section},
        questions={"relation": relation},
        model=model,
    )
    answer = response.answers["relation"]
    return {
        "choice": answer.choice,
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
        "seconds": round(perf_counter() - started, 2),
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


def verdict(status: str, answer: dict | None) -> dict:
    """Fold step 1 and step 2 into one of the four labels, plus an auto-or-review flag."""
    if status == "missing":
        # confidence None: no model was called, so there is no model confidence to report
        return {"verdict": "fabricated", "confidence": None, "auto": True}
    return {
        "verdict": RELATION_TO_VERDICT[answer["choice"]],
        "confidence": answer["confidence"],
        "auto": answer["confidence"] >= AUTO_ACCEPT,
    }


def check_citation(sections: dict[str, str], citation: dict) -> dict:
    status, section = locate(sections, citation)
    answer = (
        ask(citation["claim"], section, QUESTIONS["relation"], model=TYPESAFE_MODEL)
        if section is not None
        else None
    )
    return {"id": citation["id"], "status": status, "answer": answer, **verdict(status, answer)}


# %% [markdown]
# ## Check every citation
#
# All eight citations through the same check:

# %%
print(f"{'citation':<18}{'quote':<14}{'relation':<14}{'conf':>6}  {'verdict':<13}{'action':>7}")
for citation in CITATIONS:
    result = check_citation(SECTIONS, citation)
    answer = result["answer"]
    relation = answer["choice"] if answer else "-"
    conf = f"{answer['confidence']:.2f}" if answer else "-"
    action = "auto" if result["auto"] else "review"
    print(
        f"{result['id']:<18}{result['status']:<14}{relation:<14}{conf:>6}"
        f"  {result['verdict']:<13}{action:>7}"
    )

# %% [markdown]
# Four citations came back `verified`, one `fabricated`, one `contradicted`, and two
# `unsupported`.
#
# - `epoch_seconds`, `aud_reject`, `clock_skew`, and `duplicate_names` are the accurate four.
#   All of them came back `verified` at confidence 0.98 or higher, well above `AUTO_ACCEPT`.
# - `sig_reporting` never reached the model. Its quote is not in the RFC, so the string
#   match alone marks it `fabricated`.
# - `exp_required` quotes section 4.1.4 word for word, and the same section says "Use of
#   this claim is OPTIONAL", so it is `contradicted`, at confidence 1.00.
# - `pii_encryption` and `iat_future` came back `unsupported`. `iat_future` scored 0.97,
#   above the threshold, so it was settled without a human. `pii_encryption` scored 0.56,
#   under the threshold, so it went to a human. It shows why the string match is not enough
#   on its own: its quote is in the source word for word, and the section it came from says
#   nothing about the claim.
#
# To point this at your own data, replace `rfc7519.txt` and `citations.json`.
# `load_source()` and `split_sections()` are written for an RFC's layout, so a document of
# another shape needs its own parsing.
#
# The string match is exact after normalization: a quote that is truncated or lightly
# reworded comes back as `fabricated`. A production system that tolerates sloppy quoting
# would need fuzzy matching instead.
#
# ## Open it in the playground
#
# The link holds one citation's claim and section, plus the question. Open it to run the same
# call live in the browser.

# %%
example = next(c for c in CITATIONS if c["id"] == "exp_required")
_, example_section = locate(SECTIONS, example)
playground_link = make_playground_link(
    {"claim": example["claim"], "section": example_section}, QUESTIONS
)
display(Markdown(f"🔗 [Open in the TypeSafe playground]({playground_link})"))
