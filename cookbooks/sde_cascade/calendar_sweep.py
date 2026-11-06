"""The data behind Step 6 of the SDE cascade cookbook: the cascade on 1,000 scheduling tasks.

Each task lists the busy times of 2 to 7 people over 1 to 5 work days and asks for one meeting
time that works for everyone. For every task this script:

1. asks the cheap model (gpt-6-luna) and the strong model (gpt-6.1-sol, reasoning effort high)
   for a meeting time, as a JSON object matching ``SCHEMA``;
2. asks Jev six Noul questions about the cheap model's answer, one per scheduling rule. An
   answer that isn't valid JSON matching the schema escalates without asking Jev;
3. grades both models' answers: an answer works if it is valid JSON matching the schema and
   its time is one of the task's valid times in ``calendar_answers.json``.

It writes one row per task to ``calendar_results.csv``, which Step 6 reads to draw the chart.
Every call is cached in ``calendar_cache.json``; with that file present the script replays
without calling anything. Needs OPENAI_API_KEY and TYPESAFE_API_KEY to run live.

    uv run python calendar_sweep.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import urllib.request
from pathlib import Path

import jsonschema
import pandas as pd
from cooksafe import JsonCache
from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

HERE = Path(__file__).parent
DATA_URL = (
    "https://raw.githubusercontent.com/google-deepmind/natural-plan/"
    "ca76db336072ff8931db43bc1ca8d381038cf073/data/calendar_scheduling.json"
)
RESULTS = HERE / "calendar_results.csv"
ANSWERS = json.loads((HERE / "calendar_answers.json").read_text())  # every valid time, per task
cache = JsonCache(HERE / "calendar_cache.json", indent=None)

CHEAP, STRONG = "gpt-6-luna", "gpt-6.1-sol-high"
CONFIGS = {
    CHEAP: {"model": "gpt-6-luna"},
    STRONG: {"model": "gpt-6.1-sol", "reasoning_effort": "high"},
}
PRICES = {CHEAP: (0.10, 0.50), STRONG: (2.00, 10.00)}  # $ per Mtok, input and output
TS_MODEL = "jev-1.14.0"
JEV_PRICE = 0.042  # $ per Mtok of input; output is free

TIME = {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"}
SCHEMA = {
    "type": "object",
    "properties": {
        "day": {"type": "string", "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], "description": "the day of the meeting"},
        "start_time": TIME | {"description": "when the meeting starts, 24-hour HH:MM"},
        "end_time": TIME | {"description": "when the meeting ends, 24-hour HH:MM"},
    },
    "required": ["day", "start_time", "end_time"],
    "additionalProperties": False,
}
FORMAT_LINE = "Return ONLY a JSON object matching this JSON Schema:\n" + json.dumps(SCHEMA, indent=2)

RULES = {
    "busy_conflict": "Does the meeting time in the `answer` overlap a time when any participant in the `task` is busy?",
    "outside_hours": "Is the meeting time in the `answer` outside the work hours or the days the `task` allows?",
    "breaks_preference": "Does the meeting time in the `answer` break a preference stated in the `task`, such as a participant who would rather not meet on a certain day or before or after a certain time?",
    "not_earliest": "Does the `task` ask for the earliest availability, and is there an earlier time that works for every participant?",
    "wrong_length": "Is the meeting in the `answer` a different length from the one the `task` asks for?",
    "no_single_time": "Does the `answer` fail to propose exactly one day and one start and end time?",
}
CHECKS = {
    k: Noul(instructions=v, criteria=NoulCriteria(true="the answer breaks this rule", false="the answer follows this rule"))
    for k, v in RULES.items()
}


def load_tasks() -> dict[str, dict]:
    """The 1,000 tasks, read from NATURAL PLAN at a fixed commit."""
    with urllib.request.urlopen(DATA_URL) as r:
        return json.load(r)


def prompt_for(task: dict) -> str:
    """The task's zero-shot prompt with the schema placed just before its closing SOLUTION:."""
    return re.sub(r"SOLUTION:\s*$", FORMAT_LINE + "\nSOLUTION: ", task["prompt_0shot"].rstrip())


@cache
def ask(config: str, prompt: str) -> dict:
    from openai import OpenAI

    r = OpenAI(timeout=1800).chat.completions.create(
        messages=[{"role": "user", "content": prompt}], **CONFIGS[config]
    )
    return {"text": r.choices[0].message.content, "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}


@cache
def jev_call(state_json: str) -> dict:
    """Jev's six rule checks on one answer."""
    client = TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], timeout=300)
    r = client.system_one(
        state=json.loads(state_json),
        questions={k: q.model_dump(mode="json", exclude_none=True) for k, q in CHECKS.items()},
        model=TS_MODEL,
    )
    return {"p": {k: a.noul for k, a in r.answers.items()}, "in": r.usage.input_tokens}


def parse(text: str) -> dict | None:
    """The answer as a record, or None if it isn't a JSON object matching ``SCHEMA``."""
    try:
        record = json.loads(text)
    except ValueError:
        return None
    return record if jsonschema.Draft202012Validator(SCHEMA).is_valid(record) else None


def works(tid: str, answer: str) -> bool:
    """True if the answer is valid JSON matching ``SCHEMA`` and proposes one of the task's valid times."""
    return parse(answer) in ANSWERS[tid]


def one(tid: str, task: dict) -> dict:
    row = {"id": tid}
    for name, config in (("cheap", CHEAP), ("strong", STRONG)):
        call = ask(config, prompt_for(task))
        answer = call["text"] or ""
        row[f"{name}_cost"] = (call["in"] * PRICES[config][0] + call["out"] * PRICES[config][1]) / 1e6
        row[f"{name}_works"] = int(works(tid, answer))
        if name == "cheap":
            record = parse(answer)
            row["schema_valid"] = int(record is not None)
            if record is None:  # the schema check fails, so the task escalates without Jev
                row.update({f"p_{k}": 1.0 for k in RULES} | {"check_cost": 0.0})
            else:
                checks = jev_call(json.dumps({"task": task["prompt_0shot"], "answer": record}))
                row.update({f"p_{k}": checks["p"][k] for k in RULES})
                row["check_cost"] = checks["in"] * JEV_PRICE / 1e6
            row["flag"] = max(row[f"p_{k}"] for k in RULES)
    return row


def main() -> None:
    tasks = load_tasks()
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda kv: one(*kv), sorted(tasks.items())))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, index=False, float_format="%.6g")
    print(f"wrote {len(df)} rows to {RESULTS.name}")
    for name in ("cheap", "strong"):
        print(f"{name:6} works {df[f'{name}_works'].mean():.3f}  $ per 1,000 tasks {df[f'{name}_cost'].mean() * 1000:.2f}")
    print(f"Jev checks: $ per 1,000 tasks {df.check_cost.mean() * 1000:.3f}")


if __name__ == "__main__":
    main()
