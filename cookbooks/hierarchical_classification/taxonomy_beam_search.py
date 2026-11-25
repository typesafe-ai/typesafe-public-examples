# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

# %% [markdown]
# # Hierarchical Classification With Parallel Beam Search
#
# Many kinds of data are hierarchies: taxonomies, file systems, websites, codebases, org
# charts, biological ontologies, LLM skills and moderation policies. Hierarchical
# classification walks a hierarchy from the root to the correct leaf node; that leaf is the
# final classification. We start at the root, ask a System One `Choice` question over its
# children, follow the most probable child, and repeat until we reach a leaf
# (**Greedy Search**).
#
# Because System One questions can run in parallel, we can also follow several paths at once
# with **Beam Search**, so one wrong early choice doesn't decide the result. At each level,
# beam search scores every path by the geometric mean of its edge probabilities,
# `product(edge_probabilities) ** (1 / decisions)`, keeps the best `K`, and prunes the rest.
# The geometric mean, unlike the raw product, doesn't penalize a deep leaf just for taking
# more decisions.
#
# Asking one question per node also helps you observe and test the classifier:
# - Observability
#     - identify which nodes your misclassifications occur most in
#     - measure the number of times each node and edge is traversed
# - Testability
#     - unit test and measure the impact of hierarchy updates on classification performance
#
# ### Hierarchies used in this cookbook
#
# - **[CPC 2026.05](https://www.cooperativepatentclassification.org/):** patent subject
#   matter, from broad technology sections to narrow inventions. Published by the European
#   Patent Office and the US Patent and Trademark Office.
# - **[Shopify 2026-02](https://github.com/Shopify/product-taxonomy):** retail product
#   categories, from store departments to specific product types. From Shopify's Product
#   Taxonomy (MIT).
# - **[MeSH 2026](https://www.nlm.nih.gov/mesh/):** biomedical subjects from broad domains to
#   specific conditions. MeSH is a DAG, so one descriptor can appear under several parents;
#   we give each of its official tree numbers a separate path. Source: U.S. National Library
#   of Medicine. This is the January 2026 build of MeSH 2026 and may not be the most current
#   MeSH data.
#
# Attribution and licences for the three are in
# [`DATA_LICENSE.md`](https://github.com/typesafe-ai/typesafe-public-examples/blob/main/cookbooks/hierarchical_classification/DATA_LICENSE.md).
#
# ### Methods
#
# - **Greedy search:** choose the highest-probability child and discard every alternative.
#   One early mistake cannot be recovered.
# - **Beam search:** retain `K` plausible paths and classify every frontier in parallel.
#   When an early choice is close, a clearer decision further down can put the right path
#   back on top. The leaf of the path with the highest geometric-mean probability is the
#   final classification.
# - **System One Choice questions:** each node asks one `Choice` question over its children,
#   and the answer's probabilities become the edge probabilities. The beam's `K` questions
#   at each level run in parallel, so beam search takes one round of calls per level, as
#   greedy search does.
# - **Formula:**
#   - `path_score = product(edge_probabilities) ** (1 / decisions)`
#     - used for pruning and comparing paths
#     - for hierarchies deeper than about 10 levels, compute it as `exp(mean(log(probs)))`
#       to avoid floating-point precision loss
#   - `separation = top_path_score / second_path_score`
#     - reported only, not used for pruning
#     - near `1×`, the top two paths are almost tied; the further above `1×`, the clearer
#       the winner

# %% [markdown]
# ## Setup
#
# Requires Python 3.10 or later.
#
# ```bash
# pip install typesafe-sdk ipython 'cooksafe>=0.3.0,<0.4.0'
# ```
#
# With uv, `uv run taxonomy_beam_search.py` installs the dependencies for you.
#
# The code replays saved answers, so it runs as published without an API key. To change a
# request, set `TYPESAFE_API_KEY`; in Colab, add it in the Secrets panel. The first run
# downloads the three hierarchies (about 28 MB) into `datasets/`.
#
# Numbers below came from `jev-1.14.0` on 2026-11-03.

# %% [markdown]
# ## Load and visualize the example hierarchies
#
# We download a pinned version of each taxonomy, turn it into a tree of direct children,
# and draw each search as a static SVG.

# %%
import html
import os
import shutil
import textwrap
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, TypeAlias
from xml.etree import ElementTree
from zipfile import ZipFile

from cooksafe import JsonCache
from IPython.display import Markdown, display
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

Tree: TypeAlias = dict[str, "Tree"]


class Report(Markdown):
    """Markdown that renders in a notebook and prints as plain text from a terminal."""

    def __str__(self) -> str:
        return self.data


class Hierarchy(NamedTuple):
    """One query and a complete hierarchy.

    :param slug: filename-safe taxonomy name.
    :param name: display name.
    :param source_url: hierarchy source.
    :param node_count: number of loaded hierarchy nodes.
    :param document: unstructured text classified by Jev.
    :param expected_leaf: expected final classification.
    :param tree: nested direct-child menus.
    """

    slug: str
    name: str
    source_url: str
    node_count: int
    document: str
    expected_leaf: str
    tree: Tree


EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/"
    "main/cookbooks/hierarchical_classification"
)
CPC_URL = (
    "https://www.cooperativepatentclassification.org/sites/default/files/"
    "cpc/archive/CPCSchemeXML202605.zip"
)
SHOPIFY_URL = (
    "https://raw.githubusercontent.com/Shopify/product-taxonomy/"
    "v2026-02/dist/en/categories.txt"
)
# NLM updates the current MeSH file in place, so the pinned January 2026 build is hosted.
MESH_URL = f"{EXAMPLES_URL}/datasets/mesh_descriptors_2026.zip"
MESH_CATEGORIES = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}


def _download(url: str, path: Path) -> Path:
    """Download a pinned dataset once.

    :param url: pinned dataset URL.
    :param path: local cache path.
    :returns: local dataset path.
    """

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "typesafe-taxonomy/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as file:
            shutil.copyfileobj(response, file)
    temporary_path.replace(path)
    return path


def _insert(tree: Tree, path: tuple[str, ...]) -> None:
    subtree_value: Tree = tree
    for label in path:
        subtree_value = subtree_value.setdefault(label, {})


def _cpc_title(item: ElementTree.Element) -> str:
    class_title: ElementTree.Element | None = item.find("class-title")
    if class_title is None:
        return ""
    return " ".join(" ".join(class_title.itertext()).split())


def _load_cpc(path: Path) -> tuple[Tree, int]:
    titles: dict[str, str] = {}
    levels: dict[str, int] = {}
    parent_by_symbol: dict[str, str] = {}
    children_by_symbol: defaultdict[str, list[str]] = defaultdict(list)

    def visit(item: ElementTree.Element, parent_symbol: str | None) -> None:
        symbol: str | None = item.findtext("classification-symbol")
        next_parent: str | None = parent_symbol
        if symbol:
            title: str = _cpc_title(item)
            if title:
                titles[symbol] = title
            levels[symbol] = min(levels.get(symbol, 99), int(item.attrib["level"]))
            if (
                parent_symbol
                and parent_symbol != symbol
                and symbol not in parent_by_symbol
            ):
                parent_by_symbol[symbol] = parent_symbol
                children_by_symbol[parent_symbol].append(symbol)
            next_parent = symbol
        for child in item.findall("classification-item"):
            visit(child, next_parent)

    with ZipFile(path) as zip_file:
        names = sorted(
            name
            for name in zip_file.namelist()
            if name.startswith("cpc-scheme-") and name.endswith(".xml")
        )
        for name in names:
            root = ElementTree.fromstring(zip_file.read(name))
            for item in root.findall("classification-item"):
                visit(item, None)

    labels: dict[str, str] = {
        symbol: f"{symbol} {titles.get(symbol, '')}".strip() for symbol in levels
    }

    def build(symbol: str) -> Tree:
        return {
            labels[child]: build(child) for child in children_by_symbol.get(symbol, [])
        }

    root_symbols: list[str] = sorted(
        symbol for symbol, level in levels.items() if level == 2
    )
    tree: Tree = {labels[symbol]: build(symbol) for symbol in root_symbols}
    return tree, len(labels)


def _load_shopify(path: Path) -> tuple[Tree, int]:
    tree: Tree = {}
    category_count: int = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        _, path_text = line.split(" : ", maxsplit=1)
        category_path: tuple[str, ...] = tuple(path_text.strip().split(" > "))
        _insert(tree, category_path)
        category_count += 1
    return tree, category_count


def _load_mesh(path: Path) -> tuple[Tree, int]:
    """Load every official MeSH tree-number path.

    A descriptor may have multiple tree numbers because MeSH is a DAG. Expanding
    those positions into paths makes it usable by the tree-oriented beam search.

    :param path: MeSH descriptor XML ZIP.
    :returns: expanded tree and position count.
    """

    with ZipFile(path) as zip_file:
        root: ElementTree.Element = ElementTree.fromstring(
            zip_file.read("desc2026.xml")
        )
    names_by_tree_number: dict[str, str] = {
        tree_number.text: descriptor_record.findtext("DescriptorName/String", "")
        for descriptor_record in root.findall("DescriptorRecord")
        for tree_number in descriptor_record.findall("TreeNumberList/TreeNumber")
        if tree_number.text
    }
    tree: Tree = {}
    for tree_number in sorted(names_by_tree_number):
        parts: list[str] = tree_number.split(".")
        prefixes: list[str] = [
            ".".join(parts[:index]) for index in range(1, len(parts) + 1)
        ]
        category_code: str = tree_number[0]
        category_path: tuple[str, ...] = (
            f"{category_code} {MESH_CATEGORIES[category_code]}",
            *(f"{prefix} {names_by_tree_number[prefix]}" for prefix in prefixes),
        )
        _insert(tree, category_path)
    position_count: int = len(names_by_tree_number) + len(tree)
    return tree, position_count


def load_hierarchies(data_directory: Path = Path("datasets")) -> tuple[Hierarchy, ...]:
    """Load the three public taxonomies.

    :param data_directory: cache directory for the pinned raw files.
    :returns: CPC, Shopify, and MeSH examples.
    """

    cpc_tree, cpc_nodes = _load_cpc(
        _download(CPC_URL, data_directory / "CPCSchemeXML202605.zip")
    )
    shopify_tree, shopify_nodes = _load_shopify(
        _download(SHOPIFY_URL, data_directory / "shopify_categories_2026-02.txt")
    )
    mesh_tree, mesh_nodes = _load_mesh(
        _download(MESH_URL, data_directory / "mesh_descriptors_2026.zip")
    )
    return (
        Hierarchy(
            slug="cpc",
            name="CPC patents",
            source_url=CPC_URL,
            node_count=cpc_nodes,
            document=(
                "Patent abstract: a freestanding structural wooden perch for poultry or "
                "pet birds. The elevated roost has crossbars sized for bird feet and mounts "
                "inside an aviary."
            ),
            expected_leaf="A01K31/12 Perches for poultry or birds, e.g. roosts",
            tree=cpc_tree,
        ),
        Hierarchy(
            slug="shopify",
            name="Shopify products",
            source_url=SHOPIFY_URL,
            node_count=shopify_nodes,
            document=(
                "Furniture listing: a wall-mounted window shelf bed. This padded floating shelf "
                "uses suction cups and a washable cushion as a sunny perch for one cat."
            ),
            expected_leaf="Cat Window Beds & Perches",
            tree=shopify_tree,
        ),
        Hierarchy(
            slug="mesh",
            name="MeSH biomedical subjects",
            source_url=MESH_URL,
            node_count=mesh_nodes,
            document=(
                "Clinical abstract: Crohn disease with transmural ileocolonic inflammation, "
                "skip lesions, abdominal pain, and chronic diarrhea. Colonoscopy showed "
                "cobblestoning and biopsy found noncaseating granulomas; treatment with "
                "infliximab produced remission."
            ),
            expected_leaf="C06.405.469.432.500 Crohn Disease",
            tree=mesh_tree,
        ),
    )


NODE_W, NODE_H = 300, 38
COL_W, ROW_H = 360, 50
PAD_X = 28
EDGE_TOP_K = 5


def subtree(tree: Tree, path: tuple[str, ...]) -> Tree:
    """Return the direct-child menu below ``path``.

    :param tree: taxonomy root.
    :param path: path from the taxonomy root.
    :returns: child mapping at the path.
    """

    subtree_value: Tree = tree
    for label in path:
        subtree_value = subtree_value[label]
    return subtree_value


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _truncate(value: str, length: int = 33) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def _build_nodes(hierarchy: Hierarchy, result: dict) -> dict:
    records: dict[tuple[str, ...], dict] = {
        tuple(record["parent"]): record for record in result["records"]
    }
    best_path: tuple[str, ...] = tuple(result["beam"][0]["path"])
    greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
    retained: set[tuple[str, ...]] = {tuple(path) for path in result["retained_paths"]}

    def grow(path: tuple[str, ...]) -> list[dict]:
        record: dict | None = records.get(path)
        if record is None:
            return []
        children: list[dict] = []
        probabilities: dict[str, float] = record["probabilities"]
        ranked: list[tuple[str, float]] = sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )
        shown_labels: set[str] = {
            label for label, p in ranked[:EDGE_TOP_K] if round(p, 2) > 0
        }
        shown_labels.update(
            label
            for label, _ in ranked
            if path + (label,) in retained
            or path + (label,) == best_path[: len(path) + 1]
            or path + (label,) == greedy_path[: len(path) + 1]
        )
        for label, probability in ranked:
            if label not in shown_labels:
                continue
            child_path: tuple[str, ...] = path + (label,)
            on_best_path: bool = child_path == best_path[: len(child_path)]
            on_greedy_path: bool = child_path == greedy_path[: len(child_path)]
            kind: str = (
                "winner"
                if on_best_path
                else "greedy"
                if on_greedy_path
                else "beam"
                if child_path in retained
                else "alt"
            )
            children.append(
                {
                    "label": label,
                    "probability": probability,
                    "decision": len(probabilities) > 1,
                    "kind": kind,
                    "children": grow(child_path),
                }
            )
        return children

    return {
        "label": hierarchy.name,
        "probability": None,
        "kind": "root",
        "children": grow(()),
    }


def _layout(root: dict) -> tuple[int, int]:
    rows: list[int] = [0]
    maximum_depth: list[int] = [0]

    def walk(node: dict, depth: int) -> None:
        node["depth"] = depth
        maximum_depth[0] = max(maximum_depth[0], depth)
        if node["children"]:
            for child in node["children"]:
                walk(child, depth + 1)
            node["row"] = (node["children"][0]["row"] + node["children"][-1]["row"]) / 2
        else:
            node["row"] = rows[0]
            rows[0] += 1

    walk(root, 0)
    return maximum_depth[0], rows[0]


def render_svg(hierarchy: Hierarchy, result: dict, path: Path) -> None:
    """Write a standalone SVG of the beam-search traversal, drawn left to right.

    :param hierarchy: taxonomy demonstration.
    :param result: beam-search result from the notebook.
    :param path: output SVG path.
    """

    root: dict = _build_nodes(hierarchy, result)
    maximum_depth, row_count = _layout(root)
    document_lines: list[str] = textwrap.wrap(
        hierarchy.document,
        width=105,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    document_y: int = 124
    greedy_y: int = document_y + (len(document_lines) - 1) * 21 + 34
    beam_y: int = greedy_y + 25
    method_y: int = beam_y + 29
    legend_y: int = method_y + 23
    header_height: int = legend_y + 32
    width: int = PAD_X * 2 + maximum_depth * COL_W + NODE_W
    height: int = header_height + max(row_count, 1) * ROW_H + 34
    edges: list[str] = []
    nodes: list[str] = []

    def node_x(node: dict) -> float:
        return PAD_X + node["depth"] * COL_W

    def node_y(node: dict) -> float:
        return header_height + node["row"] * ROW_H

    def walk(node: dict) -> None:
        x_value, y_value = node_x(node), node_y(node)
        for child in node["children"]:
            child_x, child_y = node_x(child), node_y(child)
            x1, y1 = x_value + NODE_W, y_value + NODE_H / 2
            x2, y2 = child_x, child_y + NODE_H / 2
            bend: float = COL_W * 0.38
            edges.append(
                f'<path class="edge {child["kind"]}" '
                f'd="M{x1:.0f},{y1:.0f} C{x1 + bend:.0f},{y1:.0f} '
                f'{x2 - bend:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"/>'
            )
            if child["decision"]:
                edges.append(
                    f'<text class="prob" x="{x2 - 7:.0f}" y="{y2 - 5:.0f}" '
                    f'text-anchor="end">{child["probability"]:.2f}</text>'
                )
            walk(child)

        kind: str = node["kind"]
        label: str = _truncate(node["label"], 36)
        nodes.append(
            f'<g class="node {kind}"><title>{_escape(node["label"])}</title>'
            f'<rect x="{x_value:.0f}" y="{y_value:.0f}" width="{NODE_W}" '
            f'height="{NODE_H}" rx="7"/>'
            f'<text x="{x_value + 11:.0f}" y="{y_value + 24:.0f}">'
            f"{_escape(label)}</text></g>"
        )

    walk(root)
    best: dict = result["beam"][0]
    best_path: tuple[str, ...] = tuple(best["path"])
    greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
    beam_leaf: str = best_path[-1] if best_path else "no leaf"
    greedy_leaf: str = greedy_path[-1] if greedy_path else "no leaf"
    beam_width: int = result["beam_width"]
    separation_ratio: float = result["separation_ratio"]
    document_text: str = "".join(
        f'<text class="document" x="24" y="{document_y + index * 21}">'
        f"{_escape(line)}</text>"
        for index, line in enumerate(document_lines)
    )
    legend_text: str = (
        "green = greedy and beam (same route)"
        if greedy_path == best_path
        else "orange = greedy · green = beam winner"
    ) + " · purple = retained beam · dashed = pruned"
    separation_text: str = (
        ">999×" if separation_ratio > 999 else f"{separation_ratio:.2f}×"
    )

    svg: str = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
  width="{width}" height="{height}" role="img" aria-label="{_escape(hierarchy.name)} taxonomy beam search">
<style>
  .bg {{ fill:#f6f7fb }}
  text {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace }}
  .eyebrow {{ font-size:14px; font-weight:700; letter-spacing:1.2px; fill:#4f46e5 }}
  .title {{ font:700 28px system-ui,-apple-system,"Segoe UI",sans-serif; fill:#181b28 }}
  .document-label {{ font:700 12px system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:1px; fill:#777c91 }}
  .document {{ font:500 17px system-ui,-apple-system,"Segoe UI",sans-serif; fill:#303449 }}
  .copy {{ font-size:14px; fill:#5c6178 }}
  .result {{ font-size:14px; font-weight:700 }}
  .greedy-result {{ fill:#c2410c }}
  .beam-result {{ fill:#15803d }}
  .edge {{ fill:none; stroke:#c9cee0; stroke-width:2 }}
  .edge.winner {{ stroke:#15803d; stroke-width:2.5 }}
  .edge.greedy {{ stroke:#ea580c; stroke-width:2.5 }}
  .edge.beam {{ stroke:#4f46e5; stroke-width:2.2 }}
  .edge.alt {{ opacity:.48 }}
  .prob {{ font-size:12px; font-weight:600; fill:#5c6178 }}
  .node rect {{ stroke-width:1.7 }}
  .node text {{ font-size:13px }}
  .node.root rect {{ fill:#f0f2f8; stroke:#e2e5f0 }}
  .node.root text {{ fill:#5c6178 }}
  .node.winner rect {{ fill:#e4f5ea; stroke:#15803d }}
  .node.winner text {{ fill:#15803d; font-weight:700 }}
  .node.greedy rect {{ fill:#fff0e8; stroke:#ea580c }}
  .node.greedy text {{ fill:#c2410c; font-weight:700 }}
  .node.beam rect {{ fill:#ecebfd; stroke:#4f46e5 }}
  .node.beam text {{ fill:#181b28 }}
  .node.alt rect {{ fill:#fff; stroke:#e2e5f0; stroke-dasharray:3 3 }}
  .node.alt text {{ fill:#5c6178 }}
</style>
<rect class="bg" width="{width}" height="{height}" rx="14"/>
<text class="eyebrow" x="24" y="32">{hierarchy.node_count:,} NODES</text>
<text class="title" x="24" y="69">Greedy vs parallel beam search</text>
<text class="document-label" x="24" y="99">DOCUMENT</text>
{document_text}
<text class="result greedy-result" x="24" y="{greedy_y}">GREEDY TOP-1 → {_escape(_truncate(greedy_leaf, 105))}</text>
<text class="result beam-result" x="24" y="{beam_y}">BEAM K={beam_width} → {_escape(_truncate(beam_leaf, 105))}</text>
<text class="copy" x="24" y="{method_y}">separation (top ÷ second path score) = {separation_text}</text>
<text class="copy" x="24" y="{legend_y}">{legend_text}</text>
{"".join(edges)}{"".join(nodes)}
</svg>'''
    path.write_text(svg)


# %% [markdown]
# ## Implement greedy and beam search
#
# Each node's children become one `Choice` question. Both strategies record every
# distribution they receive, so the diagrams can label each edge with its probability.

# %%
HIERARCHIES = load_hierarchies()
TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
BEAM_WIDTH, MAX_DEPTH, EPSILON = 3, 12, 1e-9
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
    retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0),
)
CACHE = Path("hierarchical_classification_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    _download(f"{EXAMPLES_URL}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)


@json_cache
def choose(state: str, question: Choice, model: str) -> dict[str, float]:
    """Ask one atomic direct-child question and return its distribution."""
    keys = question.criteria
    if len(keys) == 1:
        return {label: 1.0 for label in keys.values()}
    response = client.system_one(
        state=state, questions={"child": question}, model=model
    )
    probabilities = response.answers["child"].probabilities
    return {label: probabilities[key] for key, label in keys.items()}


def child_question(labels: tuple[str, ...]) -> Choice:
    """Build the direct-child Choice; its option keys map back to the labels."""
    keys = {f"c{i}": label for i, label in enumerate(labels)}
    return Choice(
        instructions="Which direct child category best matches this document?",
        criteria=keys,
    )


def extend_candidate(
    candidate: dict, label: str, probabilities: dict[str, float]
) -> dict:
    """Append one edge and recompute its geometric-mean path score."""
    is_decision: bool = len(probabilities) > 1
    # Use log space for very deep trees to avoid floating-point precision loss.
    probability_product: float = candidate["probability_product"] * (
        max(probabilities[label], EPSILON) if is_decision else 1.0
    )
    decision_count: int = candidate["decision_count"] + is_decision
    return {
        "path": candidate["path"] + (label,),
        "probability_product": probability_product,
        "decision_count": decision_count,
        "score": probability_product ** (1 / decision_count) if decision_count else 1.0,
    }


def choice_record(path: tuple[str, ...], probabilities: dict[str, float]) -> dict:
    """Package one sibling decision for the traversal diagram."""
    return {"parent": path, "probabilities": probabilities}


def beam_search(hierarchy: Hierarchy) -> dict:
    """Parallel width-three beam search using geometric-mean probability."""
    beam = [{"path": (), "probability_product": 1.0, "decision_count": 0, "score": 1.0}]
    records, retained_paths = [], {()}

    for _ in range(MAX_DEPTH):
        expandable = [
            candidate
            for candidate in beam
            if subtree(hierarchy.tree, candidate["path"])
        ]
        finished = [
            candidate
            for candidate in beam
            if not subtree(hierarchy.tree, candidate["path"])
        ]
        if not expandable:
            break
        with ThreadPoolExecutor(max_workers=BEAM_WIDTH) as executor:
            distributions = list(
                executor.map(
                    lambda candidate: choose(
                        hierarchy.document,
                        child_question(tuple(subtree(hierarchy.tree, candidate["path"]))),
                        model=TYPESAFE_MODEL,
                    ),
                    expandable,
                )
            )

        expanded = []
        round_records = []
        for candidate, probabilities in zip(expandable, distributions, strict=True):
            round_records.append(choice_record(candidate["path"], probabilities))
            candidate_expanded = []
            for label in probabilities:
                candidate_expanded.append(
                    extend_candidate(candidate, label, probabilities)
                )
            expanded.extend(candidate_expanded)
        beam = sorted(
            finished + expanded,
            key=lambda candidate: candidate["score"],
            reverse=True,
        )[:BEAM_WIDTH]
        retained_paths.update(candidate["path"] for candidate in beam)
        records.extend(round_records)

    beam = sorted(beam, key=lambda candidate: candidate["score"], reverse=True)
    return {
        "beam": beam,
        "records": records,
        "retained_paths": sorted(retained_paths, key=lambda path: (len(path), path)),
    }


def greedy_search(hierarchy: Hierarchy) -> dict:
    """Follow only the locally highest-probability child."""
    path, probability_product, decision_count, records = (), 1.0, 0, []
    for _ in range(MAX_DEPTH):
        labels = tuple(subtree(hierarchy.tree, path))
        if not labels:
            break
        probabilities = choose(hierarchy.document, child_question(labels), model=TYPESAFE_MODEL)
        records.append(choice_record(path, probabilities))
        label = max(probabilities, key=probabilities.get)
        if len(probabilities) > 1:
            probability_product *= max(probabilities[label], EPSILON)
            decision_count += 1
        path += (label,)
    score: float = (
        probability_product ** (1 / decision_count) if decision_count else 1.0
    )
    return {"path": path, "score": score, "records": records}


def compare_searches(hierarchy: Hierarchy) -> dict:
    """Run beam and greedy, then merge their queried nodes for rendering."""
    result = beam_search(hierarchy)
    greedy = greedy_search(hierarchy)
    recorded_paths = {tuple(record["parent"]) for record in result["records"]}
    result["records"].extend(
        record
        for record in greedy["records"]
        if tuple(record["parent"]) not in recorded_paths
    )
    result["greedy"] = greedy
    result["beam_width"] = BEAM_WIDTH
    top_score: float = result["beam"][0]["score"]
    second_score: float = result["beam"][1]["score"]
    result["separation_ratio"] = top_score / max(second_score, EPSILON)
    return result


# %% [markdown]
# ## Compare the methods
#
# Run both strategies on three labeled examples, compare their leaves against the
# expected classifications, and visualize the routes they explored.

# %%
with ThreadPoolExecutor(max_workers=len(HIERARCHIES)) as executor:
    results = list(executor.map(compare_searches, HIERARCHIES))

rows: list[dict[str, str | int | bool]] = []
for hierarchy, result in zip(HIERARCHIES, results, strict=True):
    svg_path: Path = Path(f"{hierarchy.slug}_tree.svg")
    render_svg(hierarchy, result, svg_path)
    beam_path: tuple[str, ...] = tuple(result["beam"][0]["path"])
    greedy_path: tuple[str, ...] = tuple(result["greedy"]["path"])
    beam_leaf: str = beam_path[-1]
    greedy_leaf: str = greedy_path[-1]
    rows.append(
        {
            "hierarchy": hierarchy.name,
            "nodes": hierarchy.node_count,
            "expected leaf": hierarchy.expected_leaf,
            "greedy leaf": greedy_leaf,
            "beam K=3 leaf": beam_leaf,
            "greedy correct": greedy_leaf == hierarchy.expected_leaf,
            "beam correct": beam_leaf == hierarchy.expected_leaf,
            "mean p": f"{result['beam'][0]['score']:.2f}",
            "top/second": f"{result['separation_ratio']:.2f}×",
        }
    )

greedy_correct_count: int = sum(bool(row["greedy correct"]) for row in rows)
beam_correct_count: int = sum(bool(row["beam correct"]) for row in rows)
recovered_names: str = ", ".join(
    str(row["hierarchy"])
    for row in rows
    if not row["greedy correct"] and row["beam correct"]
)
table_lines: list[str] = [
    "| Hierarchy | Expected leaf | Greedy leaf | Beam K=3 leaf | Greedy correct | Beam correct |",
    "| --- | --- | --- | --- | --- | --- |",
]
table_lines.extend(
    "| "
    + " | ".join(
        (
            str(row["hierarchy"]),
            str(row["expected leaf"]),
            str(row["greedy leaf"]),
            str(row["beam K=3 leaf"]),
            "yes" if row["greedy correct"] else "no",
            "yes" if row["beam correct"] else "no",
        )
    )
    + " |"
    for row in rows
)
display(
    Report(
        "## Results\n\n"
        f"Beam search matched {beam_correct_count} of {len(rows)} expected leaves; "
        f"greedy search matched {greedy_correct_count} of {len(rows)}. "
        f"Keeping three paths recovered the expected classification for {recovered_names}.\n\n"
        + "\n".join(table_lines)
        + "\n\nThe diagrams show why the methods differ.\n\n"
        + "\n\n".join(
            f"### {hierarchy.name}\n\n![]({hierarchy.slug}_tree.svg)"
            for hierarchy in HIERARCHIES
        )
    )
)
