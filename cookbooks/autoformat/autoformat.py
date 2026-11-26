# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Recovering document structure with TypeSafe
#
# This cookbook takes plain text whose markup has been stripped (lines hard-wrapped
# mid-sentence, no heading markers, no list bullets) and reconstructs the structure as
# Markdown: headings, paragraphs, lists, quotes, code, callouts. The input is a team memo
# in exactly that state.
#
# A text-generation model could rewrite the text into Markdown, but a rewrite can also
# change the words. Here the model never generates text: it answers narrow questions about
# the document (*does this line pick up mid-sentence? what kind of content is this
# block?*), and code does the rendering, so every character of the output comes from the
# input, and every judgment carries a probability.
#
# The whole pipeline is two API requests per document, run in sequence:
#
# - **Pass 1, stitch:** one `Noul` question (a yes/no question whose answer is the
#   probability that yes is correct) per adjacent pair of lines, asking whether the line
#   break split a sentence across the two. All the pairs go in a single request, and lines
#   that continue a split sentence get merged back into blocks.
# - **Pass 2, classify:** one `Choice` question (pick one option from a list, with a
#   probability for every option) per merged block, choosing among heading, paragraph, list
#   item, quote, code, or callout (a note, tip, or warning set apart from the main text).
#   The blocks only exist once pass 1 has answered, so this is a second request; it also
#   carries companion questions for every block (heading level, step order, callout kind)
#   whose answers are read only when the block's type makes them relevant.
# - **Direct evidence stays in code.** Blank lines and explicit markers (`- `, `1.`, `#`)
#   are read directly and never sent to the model to reconsider; this memo kept its blank
#   lines but lost every marker. The model gets only the questions code cannot answer from
#   the text.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run autoformat.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.

# %%
import json
import os
import re
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
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"), timeout=120.0
)

# %%
EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/autoformat"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("autoformat_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %% [markdown]
# ## The document: a team memo that lost its formatting
#
# The test document is a memo about a build-system migration, in the state it arrives in a
# plain-text inbox: paragraphs hard-wrapped mid-sentence, a shell command sitting on a bare
# line, two lists with no bullets or numbers, a warning with nothing marking it as one.

# %%
RAW = download("build-memo.txt").read_text(encoding="utf-8")
print(RAW[:560])

# %% [markdown]
# Splitting lines, tracking blank lines and tagging ids all happen before the request; no
# model is involved. Each line gets a short id (`L014| `). The model reads the ids as
# ordinary text in the state, and questions and answers refer to lines by them. The
# [semantic search cookbook](https://console.typesafe.ai/docs/cookbooks/semantic_find)
# uses the same scheme.

# %%
def to_lines(text: str) -> list[dict]:
    lines, gap = [], False
    for raw in text.split("\n"):
        stripped = re.sub(r"[\t ]+", " ", raw).strip()
        if not stripped:
            gap = bool(lines)  # a leading blank is not a break
            continue
        lines.append({"text": stripped, "gap": gap})
        gap = False
    return lines


def tag(items: list[dict], prefix: str) -> str:
    return "\n".join(
        f"{chr(10) if item['gap'] else ''}{prefix}{i:03d}| {item['text']}"
        for i, item in enumerate(items)
    )


def line_id(i: int) -> str:
    return f"L{i:03d}"


def block_id(i: int) -> str:
    return f"B{i:03d}"


LINES = to_lines(RAW)
print(f"{len(LINES)} non-blank lines. The model sees, e.g.:")
print("\n".join(tag(LINES, "L").splitlines()[19:24]))

# %% [markdown]
# ## Pass 1: stitching split sentences
#
# One Noul question per adjacent pair of lines, all in one request; pairs separated by a
# blank line are skipped. The question is deliberately narrow ("does this line pick up
# mid-sentence?"), which is close to an objective fact about the text. It also quotes both
# lines, so the model judges the exact text in question rather than finding it by id
# among 28 lines.

# %%
def join_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f'Line {line_id(i - 1)} is "{LINES[i - 1]["text"]}" and line {line_id(i)} is '
            f'"{LINES[i]["text"]}". Does line {line_id(i)} pick up mid-sentence, continuing a '
            f"sentence left unfinished at the end of line {line_id(i - 1)}?"
        ),
        criteria=NoulCriteria(
            true="The line starts in the middle of a sentence that began on the previous line - the line break tore the sentence apart",
            false="The line begins a new sentence, item, heading, or thought of its own",
        ),
    )


def as_json(questions: dict) -> str:
    return json.dumps(
        {k: q.model_dump(mode="json", exclude_none=True) for k, q in questions.items()}
    )


def stitch_questions(make) -> dict:
    return {line_id(i): make(i) for i in range(1, len(LINES)) if not LINES[i]["gap"]}


@json_cache
def stitch(state: str, questions_json: str, model: str) -> dict:
    started = perf_counter()
    response = client.system_one(
        state=state, questions=json.loads(questions_json), model=model
    )
    return {
        "joins": [
            response.answers[line_id(i)].noul if line_id(i) in response.answers else 0.0
            for i in range(len(LINES))
        ],
        "seconds": round(perf_counter() - started, 2),
        "usage": [response.usage.input_tokens, response.usage.output_tokens],
    }


result = stitch(
    tag(LINES, "L"), as_json(stitch_questions(join_question)), model=TYPESAFE_MODEL
)
print(f"{sum(1 for l in LINES if not l['gap']) - 1} pair questions, one request, "
      f"{result['seconds']}s")

# %% [markdown]
# The cutoff for merging depends on how the previous line ends. After a dangling line (one
# with no sentence-ending punctuation), a join probability of 0.2 or above merges the
# pair; after terminal punctuation (`.` `!` `?` `:` `;`), the cutoff rises to 0.5.

# %%
JOIN_AFTER_DANGLING, JOIN_AFTER_TERMINAL = 0.2, 0.5


def ends_terminal(text: str) -> bool:
    return re.search(r'[.!?:;…]["\')\]]*$', text) is not None


def merge(joins: list[float]) -> list[dict]:
    blocks = []
    for i, line in enumerate(LINES):
        bar = (
            JOIN_AFTER_TERMINAL
            if i and ends_terminal(LINES[i - 1]["text"])
            else JOIN_AFTER_DANGLING
        )
        if blocks and not line["gap"] and joins[i] >= bar:
            blocks[-1]["text"] += " " + line["text"]
            blocks[-1]["lines"].append(i)
        else:
            blocks.append({"text": line["text"], "lines": [i], "gap": line["gap"]})
    return blocks


blocks = merge(result["joins"])
healed = len(LINES) - len(blocks)
print(f"{len(LINES)} lines -> {len(blocks)} blocks ({healed} line breaks healed)")
for i, block in enumerate(blocks):
    n = len(block["lines"])
    print(f"{block_id(i)}  {n} line{'s' if n > 1 else ' '}  {block['text'][:62]}")

# %% [markdown]
# ## Pass 2: classifying blocks
#
# Each stitched block gets a `Choice` question: *what kind of content is this?* These three
# dicts, plus the step question's true/false criteria inside `classify_questions` below,
# are the entire specification of the classifier. To adapt the pipeline to your own
# documents, edit these descriptions.

# %%
TYPE_CRITERIA = {
    "heading": "A short label or title that names the document or the section that follows it - not a full sentence of content",
    "paragraph": "Running prose: one or more complete sentences of explanatory or narrative text",
    "list_item": "One entry in a list of parallel items - an ingredient, a feature, a task, an attendee; reads as one of several sibling entries",
    "quote": "Words attributed to a person or source - quoted speech, a citation, an excerpt someone else wrote",
    "code": "Computer code, a shell command, terminal output, or a config snippet meant to be read verbatim",
    "callout": "A warning, tip, or important note that interrupts the flow to flag something the reader must not miss",
}
HLEVEL_CRITERIA = {
    "title": "The title of the whole document",
    "section": "A major section heading within the document",
    "subsection": "A minor heading nested under a section",
}
CALLOUT_CRITERIA = {
    "note": "Neutral extra information the reader should be aware of",
    "tip": "A helpful suggestion or shortcut that makes things easier",
    "warning": "A caution about something that can go wrong or cause harm",
}

# %% [markdown]
# If the type comes back `heading`, the renderer needs a heading level; if `list_item`,
# whether order matters; if `callout`, which kind. The types are not known yet, and waiting
# for them would mean a third round trip, so the companion questions are asked up front in
# the same request. Most of these answers are never read: the step probability of a
# paragraph goes unused. An extra question adds little, since the state is most of the
# tokens and is sent once either way, while an extra round trip adds a full request of
# latency.

# %%
HEADING_MAX_CHARS = 90  # longer blocks can't render as headings, so don't ask


def classify_questions(texts: list[str]) -> dict:
    questions = {}
    for i, text in enumerate(texts):
        bid = block_id(i)
        questions[f"type_{bid}"] = Choice(
            instructions=f"What kind of content is block {bid}?", criteria=TYPE_CRITERIA
        )
        if len(text) <= HEADING_MAX_CHARS:
            questions[f"hlevel_{bid}"] = Choice(
                instructions=f"As a heading, what level would block {bid} occupy in this document's structure?",
                criteria=HLEVEL_CRITERIA,
            )
        questions[f"step_{bid}"] = Noul(
            instructions=f"Is block {bid} an instruction in a sequence where the order of the items matters?",
            criteria=NoulCriteria(
                true="It is one step of a procedure - the items around it must happen in order",
                false="Order is irrelevant - it is a loose collection, or not a list item at all",
            ),
        )
        questions[f"callout_{bid}"] = Choice(
            instructions=f"What kind of aside is block {bid}?", criteria=CALLOUT_CRITERIA
        )
    return questions


@json_cache
def classify(texts: list[str], gaps: list[bool], questions_json: str, model: str) -> dict:
    tagged = tag([{"text": t, "gap": g} for t, g in zip(texts, gaps)], "B")
    questions = json.loads(questions_json)
    started = perf_counter()
    response = client.system_one(state=tagged, questions=questions, model=model)
    judgments = []
    for i in range(len(texts)):
        bid = block_id(i)
        type_answer = response.answers[f"type_{bid}"]
        hlevel = response.answers.get(f"hlevel_{bid}")
        judgments.append(
            {
                "type": type_answer.choice,
                "confidence": type_answer.confidence,
                "probabilities": type_answer.probabilities,
                "hlevel": hlevel.choice if hlevel else "section",
                "step": response.answers[f"step_{bid}"].noul,
                "callout": response.answers[f"callout_{bid}"].choice,
            }
        )
    return {
        "judgments": judgments,
        "n_questions": len(questions),
        "seconds": round(perf_counter() - started, 2),
        "usage": [response.usage.input_tokens, response.usage.output_tokens],
    }


texts = [b["text"] for b in blocks]
classified = classify(
    texts, [b["gap"] for b in blocks], as_json(classify_questions(texts)), model=TYPESAFE_MODEL
)
for block, judgment in zip(blocks, classified["judgments"]):
    block.update(judgment)
print(f"{classified['n_questions']} questions about {len(blocks)} blocks, one request, "
      f"{classified['seconds']}s\n")
print(f"{'block':<6}{'type':<11}{'conf':<6}{'companion used':<18}text")
for i, b in enumerate(blocks):
    companion = {
        "heading": f"level={b['hlevel']}",
        "list_item": f"step={b['step']:.2f}",
        "callout": f"kind={b['callout']}",
    }.get(b["type"], "-")
    print(f"{block_id(i):<6}{b['type']:<11}{b['confidence']:.2f}  {companion:<18}"
          f"{b['text'][:46]}")

# %% [markdown]
# The companion column shows the up-front answers in use: the three "Things to do before
# Monday" lines carry step probabilities of 0.55 to 0.68 (they will render as a numbered
# list), the three team lines sit at 0.07 to 0.08 (bulleted), and B014, the three-line
# warning about the doctor script, was classified as a callout of kind `warning`.
#
# ## Rendering
#
# Code assembles the page from the judgments. Consecutive list items become one list,
# numbered when the mean of the items' step probabilities is at least 0.5. That threshold
# is a group-level decision no single question asked directly.

# %%
STEP_THRESHOLD = 0.5
HEADING_MARK = {"title": "#", "section": "##", "subsection": "###"}
CALLOUT_MARK = {"note": "NOTE", "tip": "TIP", "warning": "WARNING"}


def to_markdown(blocks: list[dict]) -> str:
    groups = []
    for b in blocks:
        if b["type"] in ("list_item", "code") and groups and groups[-1][0] == b["type"]:
            groups[-1][1].append(b)
        else:
            groups.append((b["type"], [b]))
    parts = []
    for kind, items in groups:
        if kind == "list_item":
            ordered = sum(b["step"] for b in items) / len(items) >= STEP_THRESHOLD
            parts.append("\n".join(
                f"{n + 1}. {b['text']}" if ordered else f"- {b['text']}"
                for n, b in enumerate(items)
            ))
        elif kind == "code":
            parts.append("```\n" + "\n".join(b["text"] for b in items) + "\n```")
        elif kind == "heading":
            parts.append(f"{HEADING_MARK[items[0]['hlevel']]} {items[0]['text']}")
        elif kind == "quote":
            parts.append(f"> {items[0]['text']}")
        elif kind == "callout":
            parts.append(f"> [!{CALLOUT_MARK[items[0]['callout']]}]\n> {items[0]['text']}")
        else:
            parts.append(items[0]["text"])
    return "\n\n".join(parts) + "\n"


markdown = to_markdown(blocks)
print(markdown)

# %% [markdown]
# Every word above is from the input. The pipeline only chose boundaries, types, and
# markup.
#
# ## Open it in the playground
#
# This share link holds the stitched blocks and the full pass-2 question set. Open it to
# re-run the classification live.

# %%
playground_link = make_playground_link(
    tag(blocks, "B"),
    classify_questions([b["text"] for b in blocks]),
)
display(Markdown(f"🔗 [Open in the TypeSafe playground]({playground_link})"))

# %% [markdown]
# ----------------
#
# # Appendix
#
# ## Where the join thresholds come from
#
# The per-line join probabilities from pass 1:

# %%
print("join  line")
for i, line in enumerate(LINES):
    join = "    " if i == 0 or line["gap"] else f"{result['joins'][i]:.2f}"
    print(f"{join}  {line_id(i)}| {line['text'][:66]}")

# %% [markdown]
# On this memo the probabilities land in two separate bands: line breaks that split a
# sentence score 0.61 and up, and breaks the author meant score 0.16 or less, so any cutoff
# between the two would work here. The code still sets its cutoff by **how the previous
# line ends**, a fact code can read directly:
#
# - After a *dangling* line (one with no sentence-ending punctuation), the sentence may not
#   be finished, so a probability of 0.2 or above is enough to join the lines.
# - After *terminal* punctuation (a character that ends a sentence or clause: `.` `!` `?`
#   `:` `;`), the cutoff rises to 0.5. `L015| The platform team` follows a colon, and a
#   colon can lead into either a list or the rest of a sentence. It scores 0.16 here; a
#   higher score would still need to clear 0.5 before the list merged into the sentence
#   introducing it.
#
# ## Why the question is "mid-sentence" and not "same paragraph"
#
# The first version of this pipeline asked the obvious question: "are these two lines part
# of the same paragraph?" A run of short lines under a heading (a list typed without
# bullets) *is* a paragraph in the loose sense: the lines sit together and share a topic.
# Asked about paragraphs, the model says yes to every pair, and the stitch pass merges the
# whole list into one long block.

# %%
def naive_join_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f'Line {line_id(i - 1)} is "{LINES[i - 1]["text"]}" and line {line_id(i)} is '
            f'"{LINES[i]["text"]}". Are lines {line_id(i - 1)} and {line_id(i)} part of the '
            "same paragraph?"
        ),
        criteria=NoulCriteria(
            true="The two lines belong to the same paragraph of running text",
            false="The two lines belong to different paragraphs or different pieces of content",
        ),
    )


naive = stitch(
    tag(LINES, "L"), as_json(stitch_questions(naive_join_question)), model=TYPESAFE_MODEL
)
print(f"{'':14}{'mid-sentence':>13}{'same paragraph':>16}")
for i in (15, 16, 17, 20, 21):
    print(f"{line_id(i)}{'':2}{LINES[i]['text'][:36]:<38}"
          f"{result['joins'][i]:>7.2f}{naive['joins'][i]:>13.2f}")
print(f"\nblocks after merge: {len(blocks)} (mid-sentence) vs "
      f"{len(merge(naive['joins']))} (same paragraph)")

# %% [markdown]
# With the paragraph wording, every list item after the first in its list scores 0.61 to
# 0.72, far above the 0.2 cutoff after a dangling line, and each list collapses into a
# single block. "Same
# paragraph" asks the model to judge whether the topic carries over, and between list items
# it does. "Picks up mid-sentence" asks about the text itself. When a judgment call feeds a
# threshold, the question should name the narrowest fact that decides it. Here the wording
# is the difference between 17 blocks and 13.
#
# ## The lowest-confidence block

# %%
uncertain = min(blocks, key=lambda b: b["confidence"])
print(f'"{uncertain["text"]}"')
print(f"confidence {uncertain['confidence']:.2f}: ", end="")
print(", ".join(f"{k} {v:.2f}" for k, v in
                sorted(uncertain["probabilities"].items(), key=lambda kv: -kv[1])[:3]))

# %% [markdown]
# This is the sentence that introduces the team list. It reads as a paragraph, but it also
# tells readers to check something before they act, which is what a callout does. The
# probabilities spread accordingly (paragraph 0.71, callout 0.25, heading 0.03). A UI can
# surface that, for example by underlining for review any block whose type confidence is
# under 0.7, which would catch this block and nothing else on the page.
