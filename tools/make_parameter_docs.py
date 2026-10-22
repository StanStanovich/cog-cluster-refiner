#!/usr/bin/env python3
"""
Generate docs/PARAMETERS.md, the reference of every option of every script.

The descriptions are taken from the help texts of the command-line parsers, so the reference
cannot drift away from the code. Run it after changing an option:

    python tools/make_parameter_docs.py            # rewrite docs/PARAMETERS.md
    python tools/make_parameter_docs.py --check    # exit with an error if the file is out of date
                                                   # (the unit tests do this too)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "PARAMETERS.md"
sys.path.insert(0, str(ROOT / "scripts"))


def load(path: Path):
    """Import a Python file by path (the step scripts start with a digit, so `import` cannot be used)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# (title, file, what it does) in pipeline order
PROGRAMS = [
    ("run_pipeline.py", ROOT / "run_pipeline.py", "Runs all steps for several iterations."),
    ("01_build_hmms.py", ROOT / "scripts" / "01_build_hmms.py", "Step 1: one HMM profile per cluster."),
    ("02_search_hmms.py", ROOT / "scripts" / "02_search_hmms.py", "Step 2: search all sequences with all profiles."),
    ("03_merge_clusters.py", ROOT / "scripts" / "03_merge_clusters.py", "Step 3: expand and merge clusters."),
    ("04_deduplicate_clusters.py", ROOT / "scripts" / "04_deduplicate_clusters.py",
     "Step 4: keep every sequence in exactly one cluster."),
    ("05_add_unclustered.py", ROOT / "scripts" / "05_add_unclustered.py",
     "Step 5: add sequences that are in no cluster (once)."),
]

def render_glossary() -> str:
    common = load(ROOT / "scripts" / "common.py")
    merge = load(ROOT / "scripts" / "03_merge_clusters.py")
    top, size, cov, length = (common.TOP_FRACTION, common.SIZE_FACTOR_FULL_AT,
                              common.COVERAGE_WEIGHT, common.LENGTH_RATIO_WEIGHT)
    return f"""\
| term | meaning |
|---|---|
| **seeds** | The sequences that belong to a cluster at the start of a step, before anything is added to it. |
| **bit score** | HMMER's score of a sequence against a profile (higher = better match). It depends on the profile (its length and conservation), so scores of *different* profiles are only comparable after normalising, e.g. relative to the best score the profile gives to its own members. |
| **representatives** | The subset of a cluster's members that is aligned in a test: the best-scoring {top:.0%} of the slots plus an even spread over the rest of the ranking. The size cap is set by `--rep-sample-size`, `--rep-size` or `--sample-size`. |
| **consensus** | For each alignment column, the most common residue among the non-gap residues (ties are broken alphabetically). |
| **consensus strength** | The share of the non-gap residues of a column that equal the consensus residue (1 = perfectly conserved). The *mean consensus strength* of an alignment is the average over all columns. |
| **fit** (of a sequence) | The share of the sequence's residues that equal the consensus residue of their column (0 to 1). |
| **core column** | A column in which fewer than `--core-gap-tolerance` (default {common.CORE_GAP_TOLERANCE}) of a group's sequences have a gap, i.e. a column most of the group actually occupies. |
| **coverage** | The share of a cluster's core columns in which a sequence has a residue (a fragment covering 60% of the family gets 0.6). |
| **normalized fit** | `fit x ({1 - cov:g} + {cov:g} x coverage) x ({1 - length:g} + {length:g} x length ratio) x size factor`. *Length ratio* = shorter / longer of the sequence length and the representatives' mean length. *Size factor* = ln(cluster size + 1) / ln({size}), at most 1, so clusters with fewer than about {size} members are penalised. |
| **overlap** | Sequences shared by two expanded clusters divided by the size of the smaller cluster. |
| **combined weight** | `{merge.OVERLAP_WEIGHT:g} x overlap + {merge.CORRELATION_WEIGHT:g} x max(score correlation, 0)`. |
| **bad member** | A member whose fit is below the "member fit" threshold of the check in question. |
"""


CONSTANTS = [
    # (name, module, file, meaning)
    ("TOP_FRACTION", "common", "scripts/common.py",
     "Share of the representative slots that go to the best-scoring members; the other slots are spread evenly over the ranking."),
    ("SIZE_FACTOR_FULL_AT", "common", "scripts/common.py",
     "Cluster size from which the size factor ln(n + 1) / ln(50) reaches 1. Smaller clusters are penalised on purpose, because the consensus of a tiny cluster is unreliable."),
    ("COVERAGE_WEIGHT", "common", "scripts/common.py",
     "Weight of coverage in the normalized fit: factor = (1 - weight) + weight x coverage. A fragment can lose at most half of its score."),
    ("LENGTH_RATIO_WEIGHT", "common", "scripts/common.py",
     "Weight of the length ratio in the normalized fit: factor = (1 - weight) + weight x ratio. A length mismatch can cost at most 30% of the score."),
    ("CORE_GAP_TOLERANCE", "common", "scripts/common.py",
     "Default of `--core-gap-tolerance` (also used for the coverage term in steps 3, 4 and 5)."),
    ("EXPANSION_FIT_REPS", "merge", "scripts/03_merge_clusters.py",
     "Number of cluster members aligned with a candidate to compute its normalized fit during expansion."),
    ("OVERLAP_WEIGHT", "merge", "scripts/03_merge_clusters.py",
     "Weight of the overlap in the combined weight of a pair of clusters."),
    ("CORRELATION_WEIGHT", "merge", "scripts/03_merge_clusters.py",
     "Weight of the score correlation (negative values count as 0) in the combined weight."),
]

SCHEDULE_TEXT = """\
A schedule file is JSON. `iterations` has one entry per iteration; each entry can have a `merge`
section (options of step 3) and a `dedup` section (options of step 4). `add_unclustered` holds the
options of step 5. **Keys are the option names with underscores instead of hyphens**, so
`"score_window": 0.45` means `--score-window 0.45`; all options are described in the tables above.
Keys starting with an underscore are ignored (use them for notes). Choose a schedule with
`--schedule <name>` (a shipped one, see `--list-schedules`), `--schedule <path>` (your own file) or
`--schedule none` (the defaults of the scripts in every iteration); the schedule that was used is
saved in `<workdir>/run_info.json`. The driver checks every key
against the real scripts before it starts, and options given with `--merge-args`, `--dedup-args` or
`--add-args` are applied after the schedule and override it.

Shipped schedules (in `schedules/`):

* `relaxing_3_iterations.json` - three iterations that accept progressively weaker evidence.
* `relaxing_3_iterations_any_two_shared.json` - the same values, but a pair of clusters is tried as soon
  as it shares two sequences (`min_overlap` 0, `min_shared` 2); the alignment checks decide.
"""


def type_of(action: argparse.Action) -> str:
    if action.choices:
        return "one of: " + ", ".join(map(str, action.choices))
    if action.nargs == 0 or action.const is True:
        return "flag"
    if action.type is float:
        return "number"
    if action.type is int:
        return "integer"
    return "text" + (" (one or more)" if action.nargs in ("+", "*") else "")


def default_of(action: argparse.Action) -> str:
    if action.required:
        return "required"
    if action.default is None or action.default == argparse.SUPPRESS:
        return "none"
    if action.default is False:
        return "off"
    if isinstance(action.default, float):
        return f"{action.default:g}"
    return "none" if action.default == "" else str(action.default)


def escape(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def render_program(title: str, path: Path, summary: str) -> str:
    parser = load(path).build_parser()
    lines = [f"## `{title}`", "", summary, ""]
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.option_strings and a.dest != "help"]
        if not actions:
            continue
        heading = group.title if group.title not in ("options", "optional arguments") else "options"
        lines += [f"**{heading}**", "", "| option | type | default | description |", "|---|---|---|---|"]
        for a in actions:
            lines.append(f"| `{a.option_strings[0]}` | {type_of(a)} | {default_of(a)} | {escape(a.help or '')} |")
        lines.append("")
    return "\n".join(lines)


def render_constants() -> str:
    modules = {"common": load(ROOT / "scripts" / "common.py"),
               "merge": load(ROOT / "scripts" / "03_merge_clusters.py")}
    lines = ["| name | value | defined in | meaning |", "|---|---|---|---|"]
    for name, module, file, meaning in CONSTANTS:
        lines.append(f"| `{name}` | {getattr(modules[module], name)} | `{file}` | {escape(meaning)} |")
    return "\n".join(lines) + "\n"


def render() -> str:
    parts = [
        "# Parameter reference",
        "",
        "<!-- Generated by tools/make_parameter_docs.py from the help texts of the scripts. Do not edit by hand. -->",
        "",
        "Every option of every script, with its default and what it does. `python scripts/<script> --help` "
        "shows the same descriptions. Options whose type is *number* are plain decimals (for example `0.35`), "
        "and a default of *none* means the option is optional and unset.",
        "",
        "Values that change from iteration to iteration are set in a [schedule file](#schedule-files); "
        "the numbers in the tables below are the defaults of the scripts.",
        "",
        "## Terms used in the descriptions",
        "",
        render_glossary(),
    ]
    for title, path, summary in PROGRAMS:
        parts += [render_program(title, path, summary)]
    parts += [
        "## Fixed constants",
        "",
        "These numbers are part of the method and are not command-line options. They are named constants "
        "in the code, in case you want to experiment.",
        "",
        render_constants(),
        "## Schedule files",
        "",
        SCHEDULE_TEXT,
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only check that docs/PARAMETERS.md is up to date")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != text:
            sys.exit("docs/PARAMETERS.md is out of date. Run: python tools/make_parameter_docs.py")
        print("docs/PARAMETERS.md is up to date.")
        return
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
