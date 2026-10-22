#!/usr/bin/env python3
"""
Run the whole refinement pipeline.

One iteration =
    1. build a profile for every cluster          (scripts/01_build_hmms.py)
    2. search all sequences against all profiles  (scripts/02_search_hmms.py)
    3. expand and merge clusters                  (scripts/03_merge_clusters.py)
    4. remove duplicate cluster memberships       (scripts/04_deduplicate_clusters.py)

Once, after the iteration given by --add-unclustered-after (default: the first), sequences that
are not in any cluster are added (needs --unclustered):
    1.+2. new profiles of the current clusters, searched against clustered + unclustered sequences
    5. add the unclustered sequences that fit      (scripts/05_add_unclustered.py)
    4. remove duplicate cluster memberships again, just in case

The output of each stage is the input of the next, and the next iteration starts with the
enlarged clusters. Every iteration gets its own folder, so nothing is overwritten:

    <workdir>/iter_1/{profiles,alignments,hits,merged,dedup,logs}/
    <workdir>/iter_1/unclustered/{profiles,alignments,hits,added,dedup,logs}/   (only once)
    <workdir>/iter_2/...
    <workdir>/final_clusters/        copy of the last result

Parameters can change from iteration to iteration (typically: more relaxed in later ones). They
are read from a schedule file. The default is schedules/relaxing_3_iterations.json; choose another
one with --schedule <name> (see --list-schedules) or --schedule <path to your own file>.

No sequence that is in the input clusters is ever dropped: every step checks this, and at the
end the final clusters are compared with the input once more.

Examples
--------
    python run_pipeline.py --input clusters/ --workdir results/ --unclustered unclustered.fasta --threads 8
    python run_pipeline.py --input clusters/ --schedule relaxing_3_iterations_any_two_shared   # another shipped schedule
    python run_pipeline.py --input clusters/ --schedule my_schedule.json                       # your own schedule
    python run_pipeline.py --list-schedules                                                    # what is available
    python run_pipeline.py --input clusters/ --iterations 2 --schedule none                    # script defaults, no relaxing
    python run_pipeline.py --input clusters/ --merge-args "--min-overlap 0.15"   # extra options, applied to every iteration
"""

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
SCHEDULES_DIR = ROOT / "schedules"
DEFAULT_SCHEDULE = "relaxing_3_iterations"
sys.path.insert(0, str(SCRIPTS))

from common import (HELP_CLUSTERS, list_fasta_files, load_clusters, make_parser,  # noqa: E402
                    require_programs, write_fasta)


def run_step(script: str, *args: str) -> None:
    """Run one pipeline script; stop the whole pipeline if it fails."""
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n$ {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: {script} failed (exit code {result.returncode}); see the messages above. "
                 f"The pipeline was stopped; nothing after this step was run.")


def flags(params: Dict, extra: str = "") -> List[str]:
    """{'score_window': 0.35} -> ['--score-window', '0.35'], followed by the user's extra options."""
    out: List[str] = []
    for key, value in params.items():
        if not key.startswith("_"):
            out += [f"--{key.replace('_', '-')}", str(value)]
    return out + shlex.split(extra)


def available_schedules() -> Dict[str, Path]:
    """The schedule files shipped in the schedules/ folder, by name (file name without .json)."""
    return {path.stem: path for path in sorted(SCHEDULES_DIR.glob("*.json"))}


def resolve_schedule(name: str) -> Optional[Path]:
    """Find the schedule file for `--schedule`.

    `name` can be 'none', the path of a JSON file, or the name of a shipped schedule
    (with or without .json). Returns None for 'none'.
    """
    if name.lower() == "none":
        return None
    if Path(name).is_file():
        return Path(name)
    shipped = available_schedules()
    key = name[:-5] if name.endswith(".json") else name
    if key in shipped:
        return shipped[key]
    sys.exit(f"ERROR: schedule '{name}' not found. It must be 'none', the path of a JSON file, or one of "
             f"the shipped schedules: {', '.join(shipped)}.")


def load_schedule(name: str) -> Dict:
    path = resolve_schedule(name)
    return {} if path is None else json.loads(path.read_text())


class ListSchedulesAction(argparse.Action):
    """`--list-schedules`: print the shipped schedules and exit (like --help)."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for name, path in available_schedules().items():
            marker = " (default)" if name == DEFAULT_SCHEDULE else ""
            description = json.loads(path.read_text()).get("description", "")
            print(f"{name}{marker}\n    {textwrap.fill(description, 88, subsequent_indent='    ')}\n")
        parser.exit()


def tool_version(command: List[str], pattern: str) -> str:
    """First output line of `command` that contains `pattern`; 'unknown' if it cannot be run or fails."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            for line in (out.stdout + out.stderr).splitlines():
                if pattern in line:
                    return line.strip().lstrip("# ")[:100]
        return "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_run_info(workdir: Path, args, schedule_path: Optional[Path], schedule: Dict) -> None:
    """Record how this run was made (arguments, schedule, program versions) in <workdir>/run_info.json."""
    info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "arguments": vars(args),
        "schedule_file": str(schedule_path) if schedule_path else None,
        "schedule": schedule,
        "python": sys.version.split()[0],
        "hmmsearch": tool_version(["hmmsearch", "-h"], "HMMER"),
        "clustalo": tool_version(["clustalo", "--version"], ""),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


def write_all_sequences(cluster_dir: Path, out_fasta: Path) -> int:
    """Write each distinct sequence of all clusters to one FASTA file (the search target)."""
    _, seq_db = load_clusters(cluster_dir)
    write_fasta(out_fasta, seq_db.items())
    return len(seq_db)


def build_and_search(clusters: Path, stage: Path, extra_targets: List[str], args) -> None:
    """Steps 1 and 2: profiles for `clusters`, searched against all their sequences (+ extra targets)."""
    run_step("01_build_hmms.py", "--clusters", str(clusters),
             "--alignments", str(stage / "alignments"), "--profiles", str(stage / "profiles"),
             "--threads", str(args.threads))
    stage.mkdir(parents=True, exist_ok=True)
    n_sequences = write_all_sequences(clusters, stage / "clustered_sequences.fasta")
    print(f"\nCollected {n_sequences} distinct clustered sequences for searching.")
    run_step("02_search_hmms.py", "--profiles", str(stage / "profiles"),
             "--targets", str(stage / "clustered_sequences.fasta"), *extra_targets,
             "--out", str(stage / "hits"), "--evalue", str(args.evalue), "--cpu", str(args.threads))


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("Run the cluster refinement pipeline (steps 1-4 per iteration, step 5 once).")

    io = parser.add_argument_group("input / output")
    io.add_argument("--input", required=True, help=HELP_CLUSTERS.replace("Folder", "Starting clusters: folder", 1))
    io.add_argument("--workdir", default="results",
                    help="Folder for all results. It must not contain iteration folders (iter_1, ...) "
                         "from an earlier run; use a new folder for every run.")
    io.add_argument("--unclustered",
                    help="Optional FASTA file with sequences that are in no cluster. They are added "
                         "once (step 5) after the iteration given by --add-unclustered-after. Without "
                         "this option step 5 is skipped.")

    plan = parser.add_argument_group("iterations")
    plan.add_argument("--iterations", type=int, default=3,
                      help="Number of iterations. Each iteration builds profiles from the clusters it "
                           "starts with, so more iterations let clusters keep growing; watch the "
                           "cluster counts printed at the end to see when they stop changing.")
    plan.add_argument("--schedule", default=DEFAULT_SCHEDULE,
                      help="Parameters of every iteration (later iterations are usually more relaxed). "
                           "Give the name of a shipped schedule (see --list-schedules), the path of "
                           "your own JSON file, or 'none' to use the defaults of the scripts in every "
                           "iteration. If there are more iterations than entries, the last entry is "
                           "reused. The schedule that was used is saved in <workdir>/run_info.json.")
    plan.add_argument("--list-schedules", action=ListSchedulesAction,
                      help="Print the shipped schedules with a description of each, then exit.")
    plan.add_argument("--add-unclustered-after", type=int, default=1,
                      help="Number of the iteration after which the unclustered sequences are added "
                           "(step 5, followed by another deduplication). Only used with --unclustered.")

    run = parser.add_argument_group("settings")
    run.add_argument("--threads", type=int, default=4,
                     help="CPU threads for hmmbuild (step 1) and hmmsearch (step 2).")
    run.add_argument("--evalue", type=float, default=1e-5,
                     help="E-value cutoff of hmmsearch (step 2): only hits at or below this are "
                          "reported and can be used for expansion and merging.")

    extra = parser.add_argument_group("extra options for individual steps")
    extra.add_argument("--merge-args", default="",
                       help="Extra command-line options for step 3 as one quoted string, for example "
                            "\"--min-overlap 0.15\". Applied to every iteration, after (and overriding) "
                            "the schedule.")
    extra.add_argument("--dedup-args", default="",
                       help="Extra command-line options for step 4 as one quoted string.")
    extra.add_argument("--add-args", default="",
                       help="Extra command-line options for step 5 as one quoted string.")
    return parser


def load_script(filename: str):
    """Import a script whose file name starts with a digit (so a normal import is impossible)."""
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_options(schedule: Dict, iteration_params: List[Dict], args) -> None:
    """Check every option of every iteration against the real scripts BEFORE anything is run.

    A typo in the third iteration should not be discovered hours into a run.
    """
    required = {"03_merge_clusters.py": ["--clusters", "x", "--hits", "x"],
                "04_deduplicate_clusters.py": ["--clusters", "x"],
                "05_add_unclustered.py": ["--clusters", "x", "--unclustered", "x", "--hits", "x"]}
    checks = []
    for i in range(1, args.iterations + 1):
        params = iteration_params[min(i, len(iteration_params)) - 1] if iteration_params else {}
        checks.append((f"step 3, iteration {i}", "03_merge_clusters.py", flags(params.get("merge", {}), args.merge_args)))
        checks.append((f"step 4, iteration {i}", "04_deduplicate_clusters.py", flags(params.get("dedup", {}), args.dedup_args)))
    if args.unclustered:
        checks.append(("step 5", "05_add_unclustered.py", flags(schedule.get("add_unclustered", {}), args.add_args)))

    parsers = {}
    for where, script, options in checks:
        if script not in parsers:
            parsers[script] = load_script(script).build_parser()
        try:
            parsers[script].parse_args(required[script] + options)
        except SystemExit:
            sys.exit(f"ERROR: the options for {where} were rejected (see the message above). Nothing was run.")


def main() -> None:
    args = build_parser().parse_args()

    require_programs("clustalo", "hmmbuild", "hmmsearch")
    if args.iterations < 1:
        sys.exit("ERROR: --iterations must be at least 1.")
    if args.unclustered and not Path(args.unclustered).is_file():
        sys.exit(f"ERROR: file not found: {args.unclustered}")
    if args.unclustered and not 1 <= args.add_unclustered_after <= args.iterations:
        sys.exit("ERROR: --add-unclustered-after must be between 1 and --iterations.")
    if not args.unclustered:
        print("NOTE: no --unclustered file given; the step that adds unclustered sequences is skipped.")

    schedule_path = resolve_schedule(args.schedule)
    schedule = {} if schedule_path is None else json.loads(schedule_path.read_text())
    iteration_params = schedule.get("iterations", [])
    if iteration_params and args.iterations > len(iteration_params):
        first_reused = len(iteration_params) + 1
        reused = str(first_reused) if first_reused == args.iterations else f"{first_reused}-{args.iterations}"
        print(f"NOTE: the schedule defines {len(iteration_params)} iterations; "
              f"the last one is reused for iteration {reused}.")

    validate_options(schedule, iteration_params, args)

    workdir = Path(args.workdir)
    old = [workdir / name for name in [f"iter_{i}" for i in range(1, args.iterations + 1)] + ["final_clusters"]
           if (workdir / name).exists()]
    if old:
        sys.exit(f"ERROR: {old[0]} already exists. Use a new --workdir (results of different runs must "
                 f"not be mixed or overwritten).")
    write_run_info(workdir, args, schedule_path, schedule)
    current = Path(args.input)
    original_ids = set(load_clusters(current)[1])
    summary = [f"start: {len(list_fasta_files(current))} clusters, {len(original_ids)} sequences"]

    for i in range(1, args.iterations + 1):
        params = iteration_params[min(i, len(iteration_params)) - 1] if iteration_params else {}
        it = workdir / f"iter_{i}"
        print(f"\n{'=' * 70}\nITERATION {i} of {args.iterations}  (input: {current})\n{'=' * 70}")

        build_and_search(current, it, [], args)
        run_step("03_merge_clusters.py", "--clusters", str(current), "--hits", str(it / "hits"),
                 "--out", str(it / "merged"), "--log-dir", str(it / "logs"), "--overwrite",
                 *flags(params.get("merge", {}), args.merge_args))
        run_step("04_deduplicate_clusters.py", "--clusters", str(it / "merged"),
                 "--out", str(it / "dedup"), "--log-dir", str(it / "logs"), "--overwrite",
                 *flags(params.get("dedup", {}), args.dedup_args))
        current = it / "dedup"
        summary.append(f"after iteration {i}: {len(list_fasta_files(current))} clusters")

        # ---- one-time addition of unclustered sequences ---------------------------------
        if args.unclustered and i == args.add_unclustered_after:
            stage = it / "unclustered"
            print(f"\n{'-' * 70}\nADDING UNCLUSTERED SEQUENCES  (clusters: {current})\n{'-' * 70}")
            build_and_search(current, stage, [args.unclustered], args)
            run_step("05_add_unclustered.py", "--clusters", str(current), "--unclustered", args.unclustered,
                     "--hits", str(stage / "hits"), "--out", str(stage / "added"),
                     "--log-dir", str(stage / "logs"), "--overwrite",
                     *flags(schedule.get("add_unclustered", {}), args.add_args))
            run_step("04_deduplicate_clusters.py", "--clusters", str(stage / "added"),
                     "--out", str(stage / "dedup"), "--log-dir", str(stage / "logs"), "--overwrite",
                     *flags(params.get("dedup", {}), args.dedup_args))
            current = stage / "dedup"
            n_now = len(load_clusters(current)[1])
            summary.append(f"after adding unclustered sequences: {len(list_fasta_files(current))} clusters, "
                           f"{n_now - len(original_ids)} sequences added")

    # ---- final check: nothing from the input may be missing ----------------------------------
    final_ids = set(load_clusters(current)[1])
    missing = original_ids - final_ids
    if missing:
        sys.exit(f"ERROR: {len(missing)} sequences of the input clusters are missing from the final "
                 f"clusters (e.g. {sorted(missing)[:5]}). Results were not copied to final_clusters/.")

    final = workdir / "final_clusters"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(current, final)

    print("\n" + "=" * 70)
    print("\n".join(summary))
    print(f"Verified: all {len(original_ids)} sequences of the input clusters are in the final clusters.")
    print(f"Final clusters: {final}")


if __name__ == "__main__":
    main()
