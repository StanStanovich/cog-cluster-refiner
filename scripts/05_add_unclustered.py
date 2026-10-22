#!/usr/bin/env python3
"""
Step 5 - add sequences that are not in any cluster yet (done once, after the first iteration).

Input
-----
* the current clusters (one FASTA per cluster)
* a FASTA file with the unclustered sequences
* hit tables from step 2, made with the profiles of the *current* clusters and searched against
  BOTH the clustered sequences and the unclustered ones (the clustered ones are needed to know
  what a typical member scores)

What happens to each unclustered sequence
-----------------------------------------
1. Take the `--top-n` clusters whose profile scores it best.
2. Clear winner? The best score must be at least `--top-ratio` times the second best.
   (Skipped if there is only one hit.)
3. For each of those clusters, the sequence must pass
     a. the score window: its score lies within +/- `--score-window` of the median score of the
        cluster's own members (skipped for clusters with fewer than `--window-min-members`
        scored members), and
     b. the alignment fit: normalized fit >= `--min-fit` (see common.normalized_fit).
4. Of the clusters that passed, the one with the best fit is chosen.
5. Post-add check: the cluster plus the new sequence are aligned; the sequence is added only if
   few members fit badly (`--post-member-fit`, `--post-max-bad-fraction`).

Every sequence is judged against the clusters as they were before this step, so the result does
not depend on the order of the sequences, and a sequence ends up in at most one cluster.
Sequences that are already in a cluster are ignored. Sequences that fail stay unclustered.
No sequence that is already in a cluster is ever removed.

Example
-------
    python scripts/05_add_unclustered.py --clusters dedup/ --unclustered unclustered.fasta \\
        --hits hits/ --out enlarged/ --log-dir logs/

Output
------
    <out>/<cluster>.fasta                   clusters including the added sequences
    <log-dir>/add_unclustered_report.tsv    one row per unclustered sequence: outcome and the tests it passed
    <log-dir>/unassigned.fasta              the sequences that could not be added
    <log-dir>/add_unclustered_summary.json  parameters and counts
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (HELP_CLUSTERS, HELP_OVERWRITE, load_clusters, load_hit_tables, make_parser,
                    normalized_fit, pick_representatives, prepare_output_dir, read_fasta_records,
                    require_programs, validate_cluster_alignment, write_clusters, write_fasta)


# =============================================================================
# The individual tests (small functions, so they are easy to read and to test)
# =============================================================================

def score_ratio(top_hits: List[Tuple[str, float]]) -> Optional[float]:
    """Best score divided by second-best score; None if there is only one hit.

    `top_hits` is a list of (cluster_id, score), best first.
    """
    if len(top_hits) < 2:
        return None
    best, second = top_hits[0][1], top_hits[1][1]
    return best / second if second > 0 else float("inf")


def in_score_window(score: float, member_scores: List[float], window: float,
                    min_members: int) -> Tuple[bool, Optional[float]]:
    """Is `score` close to what the cluster's own members score?

    Returns (passed, median member score). Clusters with fewer than `min_members`
    scored members are too small to define a window and pass automatically (median is None).
    """
    if len(member_scores) < min_members:
        return True, None
    median = float(np.median(member_scores))
    return median * (1 - window) <= score <= median * (1 + window), median


# =============================================================================
# Command line
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("Add unclustered sequences to the clusters they fit best.")

    io = parser.add_argument_group("input / output")
    io.add_argument("--clusters", required=True, help=HELP_CLUSTERS)
    io.add_argument("--unclustered", required=True,
                    help="FASTA file with the sequences that are in no cluster yet. Sequences whose ID "
                         "is already in a cluster are ignored.")
    io.add_argument("--hits", required=True,
                    help="Folder with hit tables from step 2, made with profiles of exactly these "
                         "clusters and searched against the clustered AND the unclustered sequences "
                         "(the clustered ones tell the script what a typical member scores).")
    io.add_argument("--out", default="enlarged_clusters",
                    help="Folder where the enlarged cluster FASTA files are written.")
    io.add_argument("--log-dir", default="logs",
                    help="Folder for add_unclustered_report.tsv (outcome and tests for every sequence), "
                         "unassigned.fasta (sequences that could not be added) and "
                         "add_unclustered_summary.json.")
    io.add_argument("--overwrite", action="store_true", help=HELP_OVERWRITE)

    tests = parser.add_argument_group("tests a sequence has to pass")
    tests.add_argument("--top-n", type=int, default=3,
                       help="Number of best-scoring clusters (by profile bit score) that are examined "
                            "for each unclustered sequence. Only these can receive it.")
    tests.add_argument("--top-ratio", type=float, default=1.35,
                       help="The best cluster's score must be at least this many times the second "
                            "best cluster's score, so that the sequence has a clear winner. Higher is "
                            "stricter; 1 switches the test off. Sequences with a single hit skip it.")
    tests.add_argument("--score-window", type=float, default=0.30,
                       help="The sequence's score against a cluster must lie within plus or minus this "
                            "fraction of the median score of the cluster's own members (0.30 means 70% "
                            "to 130% of the median).")
    tests.add_argument("--window-min-members", type=int, default=6,
                       help="Clusters with fewer scored members than this are too small to define a "
                            "score window and skip that test.")
    tests.add_argument("--min-fit", type=float, default=0.30,
                       help="Minimum normalized fit (see 'How fit is measured' in the README; roughly "
                            "0 to 1) of the sequence to a cluster. Higher is stricter.")
    tests.add_argument("--rep-size", type=int, default=40,
                       help="Maximum number of cluster members that are aligned with the new sequence "
                            "to compute its fit (best-scoring members plus an even spread). Larger "
                            "values are more reliable but slower.")
    tests.add_argument("--post-member-fit", type=float, default=0.22,
                       help="Post-add check: a member whose fit to the cluster's consensus (with the "
                            "new sequence aligned in) is below this value counts as a bad member.")
    tests.add_argument("--post-max-bad-fraction", type=float, default=0.22,
                       help="Post-add check: the sequence is not added if more than this fraction of "
                            "the cluster's members are bad.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    cfg = parse_args()
    require_programs("clustalo")
    out_dir = prepare_output_dir(cfg.out, cfg.overwrite)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    clusters, seq_db = load_clusters(cfg.clusters)
    new_seqs: Dict[str, str] = dict(read_fasta_records(cfg.unclustered))
    if not new_seqs:
        sys.exit(f"ERROR: no sequences found in {cfg.unclustered}")
    already_clustered = [s for s in new_seqs if s in seq_db]
    new_ids = [s for s in new_seqs if s not in seq_db]
    print(f"Loaded {len(clusters)} clusters ({len(seq_db)} sequences) and {len(new_seqs)} unclustered "
          f"sequences ({len(already_clustered)} of them are already in a cluster and are ignored).")

    all_seqs = {**seq_db, **{s: new_seqs[s] for s in new_ids}}  # sequences of both kinds, by ID
    hits = {cid: table for cid, table in load_hit_tables(cfg.hits, universe=set(all_seqs)).items()
            if cid in clusters}

    # What does each cluster's profile say about its own members, and about the new sequences?
    member_scores: Dict[str, Dict[str, float]] = {}
    hits_of_new: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    new_id_set = set(new_ids)
    for cid, table in hits.items():
        members = set(clusters[cid])
        own = table[table["Subject"].isin(members)]
        member_scores[cid] = dict(zip(own["Subject"], own["Score"]))
        for subject, score in table.loc[table["Subject"].isin(new_id_set), ["Subject", "Score"]].itertuples(index=False):
            hits_of_new[subject].append((cid, float(score)))

    ranked_cache: Dict[str, List[str]] = {}

    def ranked_members(cid: str) -> List[str]:
        """Cluster members, best-scoring first (members without a hit go last, in file order)."""
        if cid not in ranked_cache:
            scored = [s for s in hits[cid]["Subject"] if s in member_scores[cid]] if cid in hits else []
            scored_set = set(scored)
            ranked_cache[cid] = scored + [s for s in clusters[cid] if s not in scored_set]
        return ranked_cache[cid]

    # ---- judge every unclustered sequence ------------------------------------------------
    report: List[Dict] = []
    additions: Dict[str, List[str]] = defaultdict(list)

    for number, seq_id in enumerate(new_ids, start=1):
        if number % 100 == 0:
            print(f"  judged {number}/{len(new_ids)} sequences; {sum(map(len, additions.values()))} added so far")
        row: Dict = {"sequence": seq_id, "status": "", "reason": "", "top_hits": "", "score_ratio": "",
                     "best_cluster": "", "best_fit": "", "post_bad_fraction": "", "post_mean_consensus": ""}
        for i in range(1, cfg.top_n + 1):
            row.update({f"candidate{i}_{k}": "" for k in
                        ("cluster", "hmm_score", "window_pass", "cluster_median", "fit", "fit_pass")})
        report.append(row)

        top_hits = sorted(hits_of_new.get(seq_id, []), key=lambda h: h[1], reverse=True)[:cfg.top_n]
        if not top_hits:
            row.update(status="no_hmm_hits", reason="no profile has a hit for this sequence")
            continue
        row["top_hits"] = "; ".join(f"{cid}:{score:.2f}" for cid, score in top_hits)

        # 2. clear winner?
        ratio = score_ratio(top_hits)
        row["score_ratio"] = "single_hit" if ratio is None else f"{ratio:.3f}"
        if ratio is not None and ratio < cfg.top_ratio:
            row.update(status="failed_score_ratio", reason=f"score ratio {ratio:.3f} < {cfg.top_ratio}")
            continue

        # 3. score window and alignment fit for each of the top clusters
        best_cluster, best_fit = None, -1.0
        for i, (cid, hmm_score) in enumerate(top_hits, start=1):
            row[f"candidate{i}_cluster"] = cid
            row[f"candidate{i}_hmm_score"] = f"{hmm_score:.2f}"

            scores = [s for s in member_scores[cid].values() if not np.isnan(s)]
            window_ok, median = in_score_window(hmm_score, scores, cfg.score_window, cfg.window_min_members)
            row[f"candidate{i}_cluster_median"] = "small_cluster" if median is None else f"{median:.2f}"
            row[f"candidate{i}_window_pass"] = str(window_ok)
            if not window_ok:
                row[f"candidate{i}_fit"], row[f"candidate{i}_fit_pass"] = "skipped", "False"
                continue

            reps = pick_representatives(ranked_members(cid), cfg.rep_size)
            fit = normalized_fit(seq_id, all_seqs, reps, cluster_size=len(clusters[cid]))
            if fit is None:
                row[f"candidate{i}_fit"], row[f"candidate{i}_fit_pass"] = "alignment_failed", "False"
                continue
            row[f"candidate{i}_fit"] = f"{fit:.3f}"
            row[f"candidate{i}_fit_pass"] = str(fit >= cfg.min_fit)
            if fit >= cfg.min_fit and fit > best_fit:
                best_cluster, best_fit = cid, fit

        if best_cluster is None:
            row.update(status="no_valid_cluster", reason="failed the window or fit test in every candidate cluster")
            continue
        row.update(best_cluster=best_cluster, best_fit=f"{best_fit:.3f}")

        # 5. does the cluster still look coherent with the new sequence in it?
        ok, post = validate_cluster_alignment(clusters[best_cluster] + [seq_id], all_seqs,
                                              cfg.post_member_fit, cfg.post_max_bad_fraction)
        if "bad_fraction" in post:
            row["post_bad_fraction"] = f"{post['bad_fraction']:.3f}"
            row["post_mean_consensus"] = f"{post['mean_consensus']:.3f}"
        if not ok:
            row.update(status="failed_post_validation", reason=post["reason"])
            continue

        additions[best_cluster].append(seq_id)
        row.update(status="assigned", reason="ok")

    # ---- verify, then write ---------------------------------------------------------------
    enlarged = {cid: members + additions.get(cid, []) for cid, members in clusters.items()}
    placed = [s for members in enlarged.values() for s in members]
    assigned = {s for added in additions.values() for s in added}
    lost = set(seq_db) - set(placed)
    if lost or len(placed) != len(set(placed)):
        sys.exit("ERROR: sequences were lost or duplicated while adding new sequences. Nothing written.")

    write_clusters(out_dir, enlarged, all_seqs)

    pd.DataFrame(report).to_csv(log_dir / "add_unclustered_report.tsv", sep="\t", index=False)
    unassigned = [(s, new_seqs[s]) for s in new_ids if s not in assigned]
    write_fasta(log_dir / "unassigned.fasta", unassigned)

    outcomes = pd.Series([r["status"] for r in report]).value_counts().to_dict()
    summary = {"parameters": vars(cfg), "unclustered_in_file": len(new_seqs),
               "ignored_already_clustered": len(already_clustered), "assigned": len(assigned),
               "unassigned": len(unassigned), "outcomes": outcomes}
    (log_dir / "add_unclustered_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"Added {len(assigned)} of {len(new_ids)} unclustered sequences "
          f"({len(additions)} clusters grew). Outcomes: {outcomes}")
    print(f"Verified: all {len(seq_db)} clustered sequences are still in the output.")
    print(f"Wrote clusters to {out_dir}; report, summary and unassigned.fasta to {log_dir}")


if __name__ == "__main__":
    main()
