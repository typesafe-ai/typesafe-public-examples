# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Date extraction
#
# *Read a date's parts off the text with Jev, then resolve them to a `date` in code.*
#
# The function you build here, `extract_date(document, role)`, takes a document and a
# phrase naming the date you want, such as "the deadline to return the form", and hands
# back a `date` with a confidence. It flags a low-confidence read, a read whose parts do not
# make a date, and a date the document never states. The date can be spelled out ("August
# 14, 2027") or written relative to today ("tomorrow", "next Thursday").
#
# Jev answers `Choice` questions about the date in one call: what kind of date it is, and
# which month, day, year, or weekday the text names. Code turns those answers into a `date`.
# The model reads what the text says and never does the calendar math.

# %% [markdown]
# ```mermaid
#   %%{init: {"fontFamily": "Inter, sans-serif", "flowchart": {"rankSpacing": 35, "wrappingWidth": 300}}}%%
# flowchart LR
#     doc["document +<br/>which date<br/><i>#quot;the deadline#quot;</i>"] --> req
#
#     subgraph req["Jev"]
#         read["<b>Choice</b> questions: what kind<br/>of date, and which parts<br/>does the text name?<br/><i>month, day, year, or today,<br/>tomorrow, next Thursday</i>"]
#     end
#
#     req -- "the parts, as<br/>the text states them" --> math["your code does<br/>the calendar math<br/>from today's date"]
#     math --> check{"a valid date, and<br/>confident in every<br/>part it used?"}
#     check green_ok@-- "yes" --> use["use the date"]
#     check blue_review@-- "no" --> review["send to a person"]
#
#     %% Branch colours (styles.css): node classes and green_/blue_ edge ids.
#     class use neonGreen
#     class review neonBlue
# ```

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run date_extraction_cookbook.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.

# %%
import json
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
TODAY = date(
    2026, 7, 30
)  # fixed reference "today" so relative dates resolve reproducibly
REVIEW_BELOW = 0.60  # threshold: a date below this confidence is flagged for a human

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
YEAR_WINDOW = list(range(1900, 2051))  # 1900..2050

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/date_extraction_cookbook"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("date_extraction_cookbook_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

# %%
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
    base_url=os.environ.get("TYPESAFE_BASE_URL"),
    timeout=30.0,
)

# %% [markdown]
# ## The questions
#
# Seven `Choice` questions go out in one call. `mode` says how the date is written:
# `absolute`
# for a date that names a month, `relative` for one written relative to today, and `none`
# when the document does not state the date at all.
#
# The other six read the pieces. An absolute date needs `month`, `day`, and `year`. A
# relative one needs `day_anchor`: today, tomorrow, the day after, or a named weekday. When
# it names a weekday, `weekday` and `week_offset` say which one and which week. Code reads
# only the pieces `mode` calls for.
#
# `year` lists one option per year from 1900 to 2050, plus two escapes. `none` means the text
# states no year and code fills one in. `out_of_range` means the text states a year outside
# the list, and code flags that instead of guessing. If a list that long bothers you, pull
# the year-like numbers out of the text first and offer the model only those.


# %%
def date_questions(role: str) -> dict[str, Choice]:
    """Seven typed choices that read a date's shape and parts off the text -- no math."""
    absent = "The document does not state this, or it is not this kind of date."
    return {
        "mode": Choice(
            instructions=(
                f"How is {role} written? 'absolute' = a calendar date naming a month (e.g. "
                "'August 14', 'the 3rd of March'); 'relative' = given relative to today (today, "
                "tomorrow, the day after tomorrow, or a named weekday such as 'next Thursday'); "
                "'none' = the document does not state this date."
            ),
            criteria={"absolute": None, "relative": None, "none": None},
        ),
        "month": Choice(
            instructions=f"If {role} is an absolute calendar date, which month is it in?",
            criteria={m: None for m in MONTHS} | {"none": absent},
        ),
        "day": Choice(
            instructions=f"If {role} is an absolute calendar date, which day of the month (1-31)?",
            criteria={str(d): None for d in range(1, 32)} | {"none": absent},
        ),
        "year": Choice(
            instructions=(
                f"If {role} is an absolute calendar date, which year? Pick 'none' if the document "
                "states no year (code infers it), or 'out_of_range' if a year is stated but not "
                "in the list."
            ),
            criteria={str(y): None for y in YEAR_WINDOW}
            | {
                "out_of_range": "A year is stated for this date but is outside the listed range.",
                "none": "No year is stated for this date.",
            },
        ),
        "day_anchor": Choice(
            instructions=(
                f"If {role} is relative to today, which day is it? 'today', 'tomorrow', "
                "'day_after' (the day after tomorrow), or 'weekday' (a named day of the week)."
            ),
            criteria={
                "today": None,
                "tomorrow": None,
                "day_after": None,
                "weekday": None,
                "none": absent,
            },
        ),
        "weekday": Choice(
            instructions=f"If {role} names a day of the week, which one?",
            criteria={w: None for w in WEEKDAYS} | {"none": absent},
        ),
        "week_offset": Choice(
            instructions=(
                f"If {role} names a weekday, which week is it in? 'next' for 'next Thursday' or "
                "'Thursday next week'; 'current' for 'this Thursday'; 'none' for a bare weekday "
                "with no qualifier (just 'Thursday' / 'on Thursday')."
            ),
            criteria={"current": None, "next": None, "none": absent},
        ),
    }


# %% [markdown]
# ## Resolve it in code
#
# One call reads all seven parts. The answers then become a `date`, with the year filled in
# when the text states none and a named weekday resolved to the day it points at. Both of
# those count from `TODAY`, which is pinned so relative dates come out the same on every run.
# A date's confidence is the lowest among the parts it used, so a weak answer on any one part
# can send the whole date to review.
#
# "next Thursday" can mean two different days, so code decides which. A weekday with no
# qualifier means the next one on or after today. `next` means the following calendar week,
# and `current` means this week.


# %%
@json_cache
def read_parts(document: str, questions_json: str, model: str) -> dict:
    """One TypeSafe call -> {part: {choice, confidence}} for the seven questions."""
    answers = client.system_one(
        state=document, questions=json.loads(questions_json), model=model
    ).answers
    return {
        part: {"choice": ans.choice, "confidence": ans.confidence}
        for part, ans in answers.items()
    }


def resolve_weekday(today: date, weekday: str, week_offset: str) -> date:
    """Which date a named weekday points to, by our stated convention: a bare weekday is the next
    occurrence on or after today; 'next' is the following calendar week; 'current' is this week."""
    w = WEEKDAYS.index(weekday)
    this_monday = today - timedelta(days=today.weekday())
    if week_offset == "next":
        return this_monday + timedelta(days=7 + w)
    if week_offset == "current":
        return this_monday + timedelta(days=w)
    return today + timedelta(days=(w - today.weekday()) % 7)


def assemble(parts: dict, today: date = TODAY) -> dict:
    """Resolve the parts Jev read into a concrete date, in code. Confidence is the weakest of
    the parts the shape actually used."""
    mode = parts["mode"]["choice"]
    confs = [parts["mode"]["confidence"]]

    def result(resolved: date | None, note: str) -> dict:
        usable = [c for c in confs if c is not None]
        confidence = min(usable) if usable else None
        needs_review = (
            resolved is None or confidence is None or confidence < REVIEW_BELOW
        )
        return {
            "date": resolved,
            "confidence": confidence,
            "needs_review": needs_review,
            "note": note,
        }

    if mode == "none":
        return result(None, "no such date stated")

    if mode == "absolute":
        month, day, year = (
            parts["month"]["choice"],
            parts["day"]["choice"],
            parts["year"]["choice"],
        )
        confs += [
            parts["month"]["confidence"],
            parts["day"]["confidence"],
            parts["year"]["confidence"],
        ]
        if "none" in (month, day) or not day.isdigit() or month not in MONTHS:
            return result(None, "absolute date incomplete")
        if (
            year == "out_of_range"
        ):  # a year is stated but off the list -> flag, don't guess
            return result(None, f"year outside {YEAR_WINDOW[0]}-{YEAR_WINDOW[-1]}")
        if (
            year == "none"
        ):  # no year stated -> infer this year, bumped to next if well past
            try:
                resolved = date(today.year, MONTHS[month], int(day))
            except (
                ValueError
            ):  # e.g. February 30 -- an inconsistent read, not a real date
                return result(None, f"impossible date: {month} {day}")
            if resolved < today - timedelta(days=31):
                resolved = date(today.year + 1, MONTHS[month], int(day))
            return result(resolved, "")
        try:  # a stated, in-range year
            return result(date(int(year), MONTHS[month], int(day)), "")
        except ValueError:
            return result(None, f"impossible date: {year}-{month}-{day}")

    if mode == "relative":
        anchor = parts["day_anchor"]["choice"]
        confs.append(parts["day_anchor"]["confidence"])
        if anchor == "today":
            return result(today, "")
        if anchor == "tomorrow":
            return result(today + timedelta(days=1), "")
        if anchor == "day_after":
            return result(today + timedelta(days=2), "")
        if anchor == "weekday":
            weekday, offset = parts["weekday"]["choice"], parts["week_offset"]["choice"]
            confs += [
                parts["weekday"]["confidence"],
                parts["week_offset"]["confidence"],
            ]
            if weekday not in WEEKDAYS:
                return result(None, "relative weekday not read")
            return result(resolve_weekday(today, weekday, offset), "")
        return result(None, "relative day not read")

    return result(None, f"unrecognized mode: {mode}")


def extract_date(document: str, role: str) -> dict:
    questions = {
        part: question.model_dump(mode="json", exclude_none=True)
        for part, question in date_questions(role).items()
    }
    return assemble(read_parts(document, json.dumps(questions), model=TYPESAFE_MODEL))


# %% [markdown]
# ## Run it
#
# Six dates to find in four short documents: two from a contract that states its years, a
# form deadline written without a year, a survey that closes "today", a review set for "next
# Thursday", and a date the form never mentions. All of them resolve against `TODAY` =
# 2026-07-30, a Thursday.

# %%
CONTRACT = "This agreement is effective January 1, 2025 and expires December 31, 2027."
FORM = "Please return the signed form by August 14."
SURVEY = "Heads up - the customer survey closes today at 5pm."
REVIEW = "Let's schedule the design review for next Thursday."

# (document, question phrase, expected date) -- the expected value is only for the scorecard.
EXAMPLES = [
    (CONTRACT, "the date the agreement takes effect", date(2025, 1, 1)),
    (CONTRACT, "the date the agreement expires", date(2027, 12, 31)),
    (FORM, "the deadline to return the form", date(2026, 8, 14)),
    (FORM, "the date of the kickoff call", None),
    (SURVEY, "the date the survey closes", date(2026, 7, 30)),
    (REVIEW, "the date of the design review", date(2026, 8, 6)),
]

print(f"{'':3}{'question':<38}{'expected':<12}{'got':<12}{'conf':>6}  flags")
print("-" * 84)
for document, role, expected in EXAMPLES:
    r = extract_date(document, role)
    got = r["date"].isoformat() if r["date"] else "none"
    exp = expected.isoformat() if expected else "none"
    mark = "OK" if r["date"] == expected else "XX"
    conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else " n/a"
    flags = "  <== review" if r["needs_review"] else ""
    if r["note"]:
        flags += f"  ({r['note']})"
    print(f"{mark:<3}{role:<38}{exp:<12}{got:<12}{conf:>6}{flags}")

# %% [markdown]
# The contract states both of its years, so those came off the text. The form states no year,
# so code filled in 2026: it takes the current year and moves to the next one only when the
# date is already more than a month past.
#
# The kickoff call is the one the form never mentions. There is a date in that form, just not
# this one. The note `no such date stated` means `mode` came back `none`. The date came back
# empty and the row is flagged for a person, even though its confidence of 0.66 clears the
# gate: a date the document never states always goes to review.

# %% [markdown]
# ## Confidence to route on
#
# A date under `REVIEW_BELOW` = 0.60 goes to a person, and so does any date that could not be
# assembled. The rest go straight through.

# %%
confident = [
    (doc, role)
    for doc, role, _ in EXAMPLES
    if not extract_date(doc, role)["needs_review"]
]
review = [
    (doc, role) for doc, role, _ in EXAMPLES if extract_date(doc, role)["needs_review"]
]
print(f"auto-accept ({len(confident)}):")
for _doc, role in confident:
    print(f"  - {role}")
print(f"\nsend to review ({len(review)}):")
for _doc, role in review:
    r = extract_date(_doc, role)
    print(f"  - {role}  (conf {r['confidence']:.2f} / {r['note'] or 'low confidence'})")

# %% [markdown]
# ## Open it in the playground
#
# The link below carries the "next Thursday" message and the same questions the code sends.
# Open it to see the answers and their confidences, and to change the wording without writing
# any code.

# %%
playground_link = make_playground_link(
    REVIEW, date_questions("the date of the design review")
)
display(
    Markdown(
        f"🔗 [Open in the TypeSafe playground]({playground_link})"
    )
)
