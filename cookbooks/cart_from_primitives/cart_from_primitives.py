# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.1", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Building a shopping cart with TypeSafe
#
# A customer can describe an order in one sentence, but an application needs a structured
# cart with separate items and fields.
#
# <Frame>
#   <iframe src="https://player.vimeo.com/video/1230923854"
#     title="TypeSafe shopping cart demo"
#     className="w-full aspect-video"
#     allow="fullscreen; picture-in-picture"></iframe>
# </Frame>
#
# The TypeSafe part of that demo builds the cart out of several
# [`Choice` questions](/primitives/choice), each of which returns one value. Here it runs on
# a small cafe menu and one hard-coded order.
#
# ## How the cart builder works
#
# The input is one sentence:
#
# ```text
# Can I get a large oat milk latte and a blueberry muffin please
# ```
#
# The application builds this cart:
#
# ```text
# latte
#   size: large
#   milk: oat
# blueberry_muffin
# ```
#
# A `Choice` question returns one option, not a whole cart. The application therefore uses
# two stages:
#
# 1. **Find the items.** Repeatedly ask which remaining menu item the order mentions.
#    Remove each selected item from the next request. Stop when Jev selects
#    `nothing_else`.
# 2. **Fill the fields.** For each found item, send one request with a `Choice` question per
#    known field. Your code assembles the returned values into the cart.
#
# ### Finding an unknown number of items with a loop
#
# The number of ordered items is not known in advance. Each request asks which remaining
# menu item appears first in the order. The loop in your code then:
#
# 1. saves the selected item;
# 2. removes it from the next request's options; and
# 3. sends the full order again with the shorter option list.
#
# A `Choice` question must select an option, so every request also includes
# `nothing_else`. That option tells the loop to stop when the order contains no remaining
# menu item.
#
# ```mermaid
#   %%{init: {"flowchart": {"rankSpacing": 45, "wrappingWidth": 260}}}%%
# flowchart LR
#     order["order<br/><i>large oat milk latte and<br/>a blueberry muffin</i>"]
#
#     subgraph local["your code"]
#         direction TB
#         options["build the options<br/><b>remaining menu items<br/>+ nothing_else</b>"]
#         save["save the selected item<br/>and drop it from<br/>the next request"]
#     end
#
#     subgraph jev["TypeSafe AI model: Jev"]
#         choice["<b>Choice:</b> which remaining<br/>item does the order mention?"]
#     end
#
#     order --> choice
#     options --> choice
#     choice --> result{"selected option"}
#     result green_item@-->|"a menu item"| save
#     save -.-> options
#     result purple_stop@-->|"nothing_else"| stop["stop the loop"]
#     %% Branch colours (styles.css): node classes and green_/blue_/purple_/orange_ edge ids.
#     %% "your code" is not a TypeSafe request, so it opts out of the pink boundary.
#     style local fill:none,stroke:#71717a,stroke-width:1.5px,stroke-dasharray: 6 4
#     class save neonGreen
#     class stop neonPurple
# ```
#
# ### Filling the known fields
#
# After the item loop, the application knows which schema applies to each item. The latte
# schema has `size` and `milk` fields. The application sends both field questions to Jev in
# one request. The muffin has no fields, so it needs no field request.
#
# ```mermaid
#   %%{init: {"flowchart": {"rankSpacing": 40, "wrappingWidth": 260}}}%%
# flowchart TB
#     order["order<br/><i>large oat milk latte and a blueberry muffin</i>"]
#
#     subgraph stage1["stage 1 · find the items"]
#         %% RL: the loop edge makes dagre rank the Jev box first, so RL puts your code on the left.
#         direction RL
#         itemLoop["shrinking-options loop<br/>in your code"]
#         subgraph jev1["TypeSafe AI model: Jev"]
#             itemChoice["<b>one Choice question<br/>per request</b>"]
#         end
#         itemLoop --> itemChoice
#         itemChoice -. "selected item" .-> itemLoop
#     end
#
#     items["found items"]
#
#     subgraph stage2["stage 2 · fill the known fields"]
#         direction LR
#         fieldQuestions["your code builds<br/><b>one Choice per field</b>"]
#         subgraph jev2["TypeSafe AI model: Jev"]
#             fieldBatch["<b>one request per item<br/>with all its field questions</b>"]
#         end
#         fieldQuestions --> fieldBatch
#     end
#
#     cart["your code assembles the cart"]
#
#     order --> stage1
#     stage1 green_done@-->|"nothing_else"| items
#     items --> stage2
#     stage2 green_cart@--> cart
#     %% Branch colours (styles.css): node classes and green_/blue_/purple_/orange_ edge ids.
#     %% The stages group your code with the Jev call, so they are neutral; only the Jev boxes are pink.
#     style stage1 fill:none,stroke:#71717a,stroke-width:1.5px,stroke-dasharray: 6 4
#     style stage2 fill:none,stroke:#71717a,stroke-width:1.5px,stroke-dasharray: 6 4
#     class items,cart neonGreen
# ```

# %% [markdown]
# ## Setup
#
# ```bash
# pip install ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run cart_from_primitives.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel.
#
# Numbers below came from `jev-1.14.0` on 2026-10-29.

# %%
import os
import urllib.request
from pathlib import Path

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
NOTHING_ELSE = "nothing_else"  # the sentinel option that ends the item loop
MAX_ITEM_REQUESTS = 6  # a hard stop, so a bad answer can never loop forever

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
    model=TYPESAFE_MODEL,
    timeout=30.0,
)

# %%
EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/cart_from_primitives"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("cart_from_primitives_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)


def fmt_confidence(confidence: float | None) -> str:
    """A confidence for printing. The API can return None, so never format it blind."""
    return f"{confidence:.2f}" if confidence is not None else "n/a"


# %% [markdown]
# ### Creating the menu
#
# The menu has four items, and only the latte has fields.
#
# Each item's description carries the words a customer would use to order it. Jev matches
# the order against these descriptions. `fields` is the item's own schema. Each field is a
# list of allowed values, and those values are the options of that field's `Choice` question.

# %%
MENU = {
    "latte": {
        "description": "Espresso with steamed milk. Also ordered as a flat white or a milky coffee.",
        "fields": {
            "size": ["small", "medium", "large"],
            "milk": ["whole", "oat", "almond"],
        },
    },
    "filter_coffee": {
        "description": "Plain brewed drip coffee, sometimes ordered as a black coffee.",
        "fields": {},
    },
    "blueberry_muffin": {
        "description": "A blueberry muffin.",
        "fields": {},
    },
    "almond_croissant": {
        "description": "An almond croissant, sometimes ordered as a pastry.",
        "fields": {},
    },
}

ORDER = "Can I get a large oat milk latte and a blueberry muffin please"

# %% [markdown]
# ### Finding the items in the order
#
# Before each request, the loop builds a fresh option set from the items not yet found and
# adds the `nothing_else` sentinel. The order
# is the request's [state](/concepts/state), and the question changes from one request to
# the next. The loop stops when `nothing_else` is selected.
#
# The loop also caps its requests, so a bad answer can never make it loop forever. It ends
# when the sentinel is selected, the menu runs out, or the cap is reached.


# %%
def item_question(remaining: dict[str, dict], found: list[str]) -> Choice:
    """Build one request's question from the items not found yet."""
    already = ", ".join(found) if found else "nothing yet"
    criteria = {name: item["description"] for name, item in remaining.items()}
    criteria[NOTHING_ELSE] = (
        "The order mentions no menu item other than the ones already accounted for."
    )
    return Choice(
        instructions=(
            f"This is a cafe order. Items already accounted for: {already}. "
            "Which remaining menu item does the order mention? If the order mentions more "
            "than one remaining item, choose the one mentioned first."
        ),
        criteria=criteria,
    )


@json_cache
def ask_item(order: str, question: dict, model: str) -> dict:
    """One item request. The question arrives as its API JSON, which also keys the cache."""
    answer = client.system_one(
        state=order, questions={"item": question}, model=model
    ).choices["item"]
    return {"choice": answer.choice, "confidence": answer.confidence}


def find_items(order: str) -> list[dict]:
    remaining = list(MENU)
    found: list[dict] = []
    item_requests = 0
    for _ in range(MAX_ITEM_REQUESTS):
        if not remaining:
            break
        question = item_question(
            {name: MENU[name] for name in remaining},
            [item["name"] for item in found],
        )
        result = ask_item(
            order, question.model_dump(mode="json", exclude_none=True), model=TYPESAFE_MODEL
        )
        item_requests += 1
        print(
            f"item request {item_requests}: {len(remaining)} items + sentinel "
            f"-> {result['choice']} (confidence {fmt_confidence(result['confidence'])})"
        )
        if result["choice"] == NOTHING_ELSE:
            break
        remaining.remove(result["choice"])
        found.append({"name": result["choice"], "confidence": result["confidence"]})
    return found


items = find_items(ORDER)
print(f"\nfound {len(items)} items: {[item['name'] for item in items]}")

# %% [markdown]
# On this order, the first two item requests each found an item. The third selected the
# sentinel and ended the loop.
#
# Because each found item leaves the option list, the loop finds each menu item at most
# once. The cart also has no quantity field, so an order for "two flat whites" still
# produces one `latte`. To handle quantities, add a `quantity` field to each item's schema,
# with values such as `one`, `two` and `three`.

# %% [markdown]
# ### Filling in a found item's fields
#
# For each found item, we build one `Choice` question per known field and send them together
# in one `system_one` call against the same order. This is the
# [fan-out pattern](/patterns/fan-out).
#
# Each field question also includes `unspecified`, so a field the customer did not mention
# gets that option instead of a forced concrete value.
#
# An item with no fields skips the call entirely.

# %%
UNSPECIFIED = "unspecified"


def field_questions(item_name: str) -> dict[str, Choice]:
    questions = {}
    for field, values in MENU[item_name]["fields"].items():
        criteria = {value: None for value in values}
        criteria[UNSPECIFIED] = (
            f"The order does not say which {field} for the {item_name}."
        )
        questions[field] = Choice(
            instructions=f"Which {field} did the order ask for, for the {item_name}?",
            criteria=criteria,
        )
    return questions


@json_cache
def ask_fields(order: str, questions: dict[str, dict], model: str) -> dict:
    """One field request. The questions arrive as their API JSON, which also keys the cache."""
    response = client.system_one(state=order, questions=questions, model=model)
    return {
        field: {"choice": answer.choice, "confidence": answer.confidence}
        for field, answer in response.choices.items()
    }


def fill_fields(order: str, item_name: str) -> dict:
    if not MENU[item_name]["fields"]:
        return {}
    questions = {
        field: question.model_dump(mode="json", exclude_none=True)
        for field, question in field_questions(item_name).items()
    }
    return ask_fields(order, questions, model=TYPESAFE_MODEL)


cart = []
for item in items:
    fields = fill_fields(ORDER, item["name"])
    cart.append(
        {"name": item["name"], "confidence": item["confidence"], "fields": fields}
    )

# %% [markdown]
# ### The assembled cart
#
# This step makes no TypeSafe calls. It prints the cart assembled from the single-value
# answers collected above.

# %%
print(f'order: "{ORDER}"\n')
for line in cart:
    print(f"{line['name']}  (item confidence {fmt_confidence(line['confidence'])})")
    for field, answer in line["fields"].items():
        conf = fmt_confidence(answer["confidence"])
        print(f"    {field:<6} {answer['choice']:<12} confidence {conf}")

# %% [markdown]
# ## Open it in the playground
#
# The link below holds the first item-finding request: the full order as the state, and
# one `Choice` question over the four menu items plus `nothing_else`. The loop above repeats
# that request, each time without the item it just found. Edit the order there and the
# answer changes with it.

# %%
playground_link = make_playground_link(ORDER, {"item": item_question(MENU, [])})
display(
    Markdown(
        f"🔗 [Open in the TypeSafe playground]({playground_link})"
    )
)

# %% [markdown]
# ## What's next
#
# - [Primitives](/primitives) is the reference for the question types and the fields each
#   answer carries.
# - The [parallel questions cookbook](/cookbooks/parallel_questions) measures what sending
#   the field questions in one request saves against asking them one at a time.
# - The [function calling cookbook](/cookbooks/function_calling) turns a sentence into a call
#   to one of your own functions the same way: one `Choice` question picks the function, and
#   one per argument fills its values. It handles an argument the user didn't mention with a
#   separate yes/no question instead of an `unspecified` option.
# - [Confidence](/confidence) covers reading a confidence value as a signal.
