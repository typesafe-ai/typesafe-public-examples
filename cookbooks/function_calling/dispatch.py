"""Turn a sentence into a call to one of your functions.

    dispatcher = Dispatcher(SPEC, TOOLS, client)
    call = dispatcher("show nvda 1h")
    call.kwargs, call.confidence, call.run()

A closed-set argument is a ``Literal``, a set of them is a ``list[Literal[...]]``, and a switch is a
``bool``. ``closed_sets`` reads those off a signature; the spec supplies a question for each and a
line per option. Everything else - free text, numbers, dates - is left alone and keeps its default.

One request per call carries the choice of function and every function's arguments; only the chosen
function's answers are read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import UnionType
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


def _bare(hint: Any) -> Any:
    """Strip ``Annotated[...]`` and ``| None`` down to the underlying type."""
    if hasattr(hint, "__metadata__"):
        hint = get_args(hint)[0]
    if get_origin(hint) in (Union, UnionType):
        named = [a for a in get_args(hint) if a is not type(None)]
        if len(named) == 1:
            return named[0]
    return hint


def _members(hint: Any) -> tuple[str, ...] | None:
    hint = _bare(hint)
    return get_args(hint) if get_origin(hint) is Literal else None


def closed_sets(fn: Callable) -> dict[str, tuple[str, tuple[str, ...] | None]]:
    """Every argument that can be filled from a fixed set: ``name -> (shape, members)``."""
    found = {}
    for name, hint in get_type_hints(fn).items():
        if name == "return":
            continue
        members, bare = _members(hint), _bare(hint)
        if members is not None:
            found[name] = ("choice", members)
        elif get_origin(bare) is list and _members(get_args(bare)[0]) is not None:
            found[name] = ("set", _members(get_args(bare)[0]))
        elif bare is bool:
            found[name] = ("flag", None)
    return found


ROUTE = "__tool__"


def questions_for(
    fn: Callable, entry: Mapping[str, Any], prefix: str = ""
) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for argument, body in entry.get("arguments", {}).items():
        shape, members = closed_sets(fn)[argument]
        qid = f"{prefix}{argument}"
        if shape == "choice":
            questions[qid] = {
                "type": "choice",
                "instructions": body["question"],
                "criteria": {m: body["options"][m] for m in members},
            }
        elif shape == "set":
            for member in members:
                questions[f"{qid}.{member}"] = {
                    "type": "noul",
                    "instructions": body["question"].replace("{}", member),
                    "criteria": {"true": body["options"][member]},
                }
        else:
            questions[qid] = {"type": "noul", "instructions": body["question"]}
        if body.get("stated"):
            questions[f"{qid}?"] = {"type": "noul", "instructions": body["stated"]}
    return questions


def all_questions(
    spec: Mapping[str, Any], tools: Mapping[str, Callable]
) -> dict[str, dict]:
    questions = {
        ROUTE: {
            "type": "choice",
            "instructions": spec["question"],
            "criteria": {n: e["description"] for n, e in spec["functions"].items()},
        }
    }
    for name, fn in tools.items():
        questions.update(questions_for(fn, spec["functions"][name], prefix=f"{name}."))
    return questions


MEMBER = 0.5  # a set member is in above this
STATED = 0.5  # an optional argument is filled above this


@dataclass
class Argument:
    name: str
    value: Any
    decisions: dict[
        str, float
    ]  # every judgement behind it, and the mass on the outcome taken
    distribution: dict[str, float] = field(default_factory=dict)
    omitted: bool = False

    @property
    def probability(self) -> float:
        return prod(self.decisions.values())


@dataclass
class Call:
    fn: Callable
    name: str
    kwargs: dict[str, Any]
    arguments: dict[str, Argument]
    tool: Argument | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def decisions(self) -> dict[str, float]:
        made = {} if self.tool is None else dict(self.tool.decisions)
        for argument in self.arguments.values():
            made.update(argument.decisions)
        return made

    @property
    def confidence(self) -> float:
        """The least certain judgement in the call - one wrong argument is enough to spoil it."""
        return min(self.decisions.values(), default=1.0)

    def weakest(self) -> Argument | None:
        return min(self.arguments.values(), key=lambda a: a.probability, default=None)

    def run(self):
        return self.fn(**self.kwargs)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.kwargs.items())})"


def read(
    fn: Callable, name: str, entry: Mapping[str, Any], answers: Mapping[str, Any]
) -> Call:
    kwargs, arguments = {}, {}
    for argument, body in entry.get("arguments", {}).items():
        shape, members = closed_sets(fn)[argument]
        qid = f"{name}.{argument}"
        stated = answers[f"{qid}?"]["noul"] if f"{qid}?" in answers else None
        if stated is not None and stated < STATED:
            arguments[argument] = Argument(
                argument, None, {f"{argument}?": 1.0 - stated}, omitted=True
            )
            continue
        presence = {} if stated is None else {f"{argument}?": stated}
        if shape == "choice":
            answer = answers[qid]
            arguments[argument] = Argument(
                argument,
                answer["choice"],
                presence | {argument: answer["probabilities"][answer["choice"]]},
                answer["probabilities"],
            )
        elif shape == "set":
            nouls = {m: answers[f"{qid}.{m}"]["noul"] for m in members}
            arguments[argument] = Argument(
                argument,
                [m for m in members if nouls[m] >= MEMBER],
                presence
                | {
                    f"{argument}.{m}": (n if n >= MEMBER else 1.0 - n)
                    for m, n in nouls.items()
                },
                nouls,
            )
        else:
            noul = answers[qid]["noul"]
            arguments[argument] = Argument(
                argument,
                noul >= 0.5,
                presence | {argument: noul if noul >= 0.5 else 1.0 - noul},
                {"true": noul},
            )
        kwargs[argument] = arguments[argument].value
    return Call(fn, name, kwargs, arguments)


class Dispatcher:
    def __init__(self, spec, tools, client, model="jev-1.12"):
        self.spec, self.tools, self.client, self.model = (
            spec,
            dict(tools),
            client,
            model,
        )
        self.questions = all_questions(spec, tools)

    def __call__(self, command: str) -> Call:
        response = self.client.system_one(
            state=command, questions=self.questions, model=self.model
        )
        answers = {k: v.model_dump(mode="json") for k, v in response.answers.items()}
        chosen = answers[ROUTE]
        name = chosen["choice"]
        call = read(self.tools[name], name, self.spec["functions"][name], answers)
        call.tool = Argument(
            ROUTE, name, {ROUTE: chosen["probabilities"][name]}, chosen["probabilities"]
        )
        call.usage = response.usage.model_dump(mode="json")
        return call
