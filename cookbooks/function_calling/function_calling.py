# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "polars", "matplotlib", "numpy",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Calling your own functions with TypeSafe
#
# When you order a "large iced oat latte, no sweetener," the barista does not write your
# sentence down. They mark four options on a cup. This cookbook does the same thing for
# a trading API: a sentence goes in, and out comes a function name and its arguments as
# evaluated enums, each with a confidence.
#
# ```text
# "plot rolling correlation between nvda and spy for the past month"
#     rolling_correlation(symbol='NVDA', benchmark='SPY', window='1mo')   confidence 0.93
#
# "compare nvda amd and msft over the past three months"
#     compare_returns(symbols=['NVDA', 'AMD', 'MSFT'], window='3mo')      confidence 0.93
#
# "show me apple daily with volume"
#     plot_price(symbol='AAPL', resolution='1d', include_volume=True)     confidence 0.69
#
# "what tickers do you have"
#     list_symbols()                                                     confidence 1.00
# ```
#
# Those calls go to ten ordinary functions in a trading assistant. Their arguments take
# values from fixed lists, so they are `Literal`s already:
#
# ```python title="Sketch"
# def plot_price(
#     symbol: Literal["SPY", "NVDA", "AMD", "AAPL", "MSFT", "TSLA"],
#     style: Literal["line", "candles"] = "line",
#     resolution: Literal["1m", "5m", "15m", "1h", "1d"] = "15m",
#     window: Literal["1d", "1w", "1mo", "3mo"] = "1w",
#     include_volume: bool = False,
#     moving_average: Literal["9", "20", "50"] | None = None,
#     log_scale: bool = False,
# ): ...
# ```
#
# An argument whose values come from a fixed list is a closed set. When it takes one value
# out of that list, it gets a `Choice` question over exactly those values, so whatever
# reaches the function is a value the function accepts. You leave the functions alone. What
# you add is a spec that says in plain words what each argument means. By the end you have a
# `Dispatcher` you can point at your own functions.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython polars matplotlib numpy 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel. The code below
# downloads two modules, with `spec.json` and the price bars, from
# [typesafe-public-examples](https://github.com/typesafe-ai/typesafe-public-examples/tree/main/cookbooks/function_calling)
# unless they are already on disk. `trader.py` holds the ten functions, and `dispatch.py`
# holds the code that reads a signature and a spec and makes the call.
#
# The prices are simulated from a fixed random seed, not real market data. The tickers are
# real, but every price, return and chart below is generated.

# %%
import json
import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import SystemOneResponse, TypeSafeClient

EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/function_calling"
)
for name in ["dispatch.py", "trader.py", "spec.json", "bars.parquet"]:
    if not Path(name).exists():
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", name)

from dispatch import ROUTE, Dispatcher, closed_sets  # noqa: E402  (downloaded above)
from trader import TOOLS, chart_theme, load  # noqa: E402


def show(draw) -> None:
    """Display the chart ``draw()`` makes. The docs build sets COOKSAFE_RENDER and also gets
    a dark copy."""
    for theme in ("light", "dark") if os.environ.get("COOKSAFE_RENDER") else ("light",):
        with chart_theme(theme):
            figure = draw()
        display(figure, metadata={"theme": theme})
        plt.close(figure)


print(f"{len(TOOLS)} functions over {load().height:,} one-minute bars")

# %%
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

typesafe = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless runs replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)

CACHE = Path("function_calling_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE and not CACHE.exists():
    urllib.request.urlretrieve(f"{EXAMPLES}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)


@json_cache
def _system_one(state: str, questions: str, model: str) -> dict:
    response = typesafe.system_one(state=state, questions=json.loads(questions), model=model)
    return {
        "answers": {k: a.model_dump(mode="json") for k, a in response.answers.items()},
        "usage": response.usage.model_dump(mode="json"),
    }


class Client:
    """The `system_one` the dispatcher calls, with every answer cached in CACHE."""

    def system_one(self, state, questions: dict, model: str) -> SystemOneResponse:
        raw = _system_one(str(state), json.dumps(questions, sort_keys=True), model=model)
        return SystemOneResponse.model_validate({**raw, "model": model})


client = Client()

# %% [markdown]
# ## Find the closed sets in the signatures
#
# The type hints already say which arguments come from a fixed list, and what is in each
# list. `closed_sets` reads a signature and sorts those arguments into three shapes: a
# **choice** (a `Literal`, so one value out of the list), a **set** (a `list[Literal[...]]`,
# so any number of them), or a **flag** (a `bool`, so on or off). All ten functions are
# defined in `trader.py`.

# %%
for name, fn in TOOLS.items():
    shapes = closed_sets(fn)
    print(
        f"  {name:<20}{len(shapes)}  "
        + ", ".join(f"{a}:{s}" for a, (s, _) in shapes.items())
    )
print(
    f"\n{sum(len(closed_sets(fn)) for fn in TOOLS.values())} fillable arguments in total"
)

# %% [markdown]
# `top_movers` shows what gets left out. Of its three arguments, two are closed sets. The
# third, `limit`, is an `int`, so it never gets a question and keeps its default of 3. Free
# text, numbers and dates work the same way: no question, and the function's default stands.
#
# ## Write the spec
#
# The `Literal` gives you the strings `"1mo"` and `"3mo"`. It does not say that a user typing
# "this quarter" means the second one. The spec says that. It holds a question per argument,
# a line per option, a description per function, and one more question that picks between the
# functions. It lives in `spec.json`, and an LLM can write it for you from the signatures.

# %%
SPEC = json.loads(Path("spec.json").read_text())
for argument in ("style", "moving_average"):
    print(
        json.dumps(
            {argument: SPEC["functions"]["plot_price"]["arguments"][argument]}, indent=2
        )
    )

# %% [markdown]
# The option keys are the strings the function takes, so nothing has to map a label back to
# an argument afterwards. `stated` makes an argument optional. It is a second yes/no question
# asking whether the command says anything about that argument at all. When the answer is no,
# the call leaves that argument out and the function's own default applies.
#
# A set argument gets its question once per member, with `{}` standing in for the member
# name. `"Does the user want {} in the comparison?"` becomes one question per ticker.
#
# Write each question about the idea rather than the words a user might pick, because the
# match is on meaning: "is amd tracking nvidia lately" reaches `rolling_correlation` even
# though neither *tracking* nor *lately* appears anywhere in `spec.json`. Avoid naming a
# question after its parameter: `"Which resolution?"` gives the command nothing to match
# against.
#
# ## Turn the spec into questions
#
# `Dispatcher` builds the questions from the spec once. Each command is then one request
# carrying the choice of function and every function's arguments, and the dispatcher reads
# only the chosen function's answers.

# %%
assistant = Dispatcher(SPEC, TOOLS, client, model=TYPESAFE_MODEL)
print(f"{len(assistant.questions)} questions per command, among them:")
for qid in (
    "__tool__",
    "plot_price.style",
    "plot_price.style?",
    "compare_returns.symbols.NVDA",
):
    question = assistant.questions[qid]
    print(f"  {qid:<30}{question['type']:<8}{str(question['instructions'])[:64]}")

# %% [markdown]
# ## Run fourteen commands
#
# A request occupies one line, and its `confidence` is the least certain judgement behind
# that call.

# %%
COMMANDS = [
    "show nvda 1h",
    "plot rolling correlation between nvda and spy for the past month",
    "when during the day does nvda trade the most",
    "what moved today",
    "what tickers do you have",
    "how did the market do this week",
    "candles for tesla with a 20 period moving average",
    "compare nvda amd and msft over the past three months",
    "how volatile is tsla",
    "biggest losers today",
    "worst drawdown for nvda this quarter, and chart it please",
    "spy stats for the last month",
    "show me apple daily with volume",
    "is amd tracking nvidia lately",
]

CALLS = {command: assistant(command) for command in COMMANDS}
for command, call in CALLS.items():
    print(f'  "{command}"')
    print(
        f"      {str(call):<66}confidence {call.confidence:.2f}"
        f"   tool {call.tool.probability:.2f}"
    )

# %% [markdown]
# Both long commands came out as asked. "plot rolling correlation between nvda and spy for
# the past month" filled three of its four arguments from one sentence and left
# `resolution` at its default. Two of them, `symbol` and `benchmark`, draw from the same six
# tickers, and each ticker landed in the right argument because the questions spell out the
# roles: *the one being measured, named first* against *the second one named, the
# yardstick*. "compare nvda amd and msft over the past three
# months" put three tickers in the set and left the other three out.
#
# Running three of them:

# %%
for command in (
    "plot rolling correlation between nvda and spy for the past month",
    "compare nvda amd and msft over the past three months",
    "when during the day does nvda trade the most",
):
    print(f'"{command}"  ->  {CALLS[command]}')
    show(CALLS[command].run)

# %% [markdown]
# And the ones that answer in text:

# %%
for command in ("how did the market do this week", "biggest losers today"):
    print(f'"{command}"  ->  {CALLS[command]}')
    print(CALLS[command].run(), "\n")

# %% [markdown]
# ## Read the confidence
#
# `confidence` reports the least certain judgement in the call, rather than the product of
# all of them, since one wrong argument is enough to spoil the result. A product answers a
# different question ("is every part right"), and it falls as a function takes more
# arguments, whether or not any one judgement is shaky.
#
# Where that number came from, argument by argument:

# %%
call = CALLS["is amd tracking nvidia lately"]
print(f'"is amd tracking nvidia lately"  ->  {call}   confidence {call.confidence:.2f}')
for name, argument in call.arguments.items():
    top = sorted(argument.distribution.items(), key=lambda kv: -kv[1])[:3]
    shown = "omitted, default stands" if argument.omitted else repr(argument.value)
    print(
        f"  {name:<12}{shown:<26}p {argument.probability:.2f}   "
        + "  ".join(f"{k} {v:.2f}" for k, v in top)
    )
print(f"  weakest argument: {call.weakest().name}")

# %% [markdown]
# `window` and `resolution` are both omitted here, because "lately" does not say how far back
# or on what bars, so `rolling_correlation` runs on its own defaults of one month and hourly
# bars. That is what the `stated` question is for. Without it, the choice would have to name
# some window even though the command names none.
#
# ## Open it in the playground
#
# The link below holds one command and the questions for the function it picked: the choice
# over the ten function descriptions, and `rolling_correlation`'s four arguments. Edit the
# command there and the arguments change with it.

# %%
COMMAND = "plot rolling correlation between nvda and spy for the past month"
picked = CALLS[COMMAND]
playground_link = make_playground_link(
    COMMAND,
    {ROUTE: assistant.questions[ROUTE]}
    | {q: v for q, v in assistant.questions.items() if q.startswith(f"{picked.name}.")},
)
display(
    Markdown(
        f"🔗 [Open in the TypeSafe playground]({playground_link})"
    )
)
