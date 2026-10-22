#!/usr/bin/env python3
"""
Step 3 - make clusters bigger: expand them with similar sequences, then merge similar clusters.

Input: the current clusters (one FASTA per cluster) and the hit tables from step 2.
Only sequences that already belong to some cluster are considered.

How it works
------------
A. EXPANSION (one cluster at a time)
   1. Look at how strongly the cluster's own profile scores the cluster's own
      sequences ("seeds") and take the median seed score.
   2. Candidates are sequences from other clusters whose score against this
      profile is within +/- `--score-window` of that median.
   3. A candidate is accepted only if, aligned with representative seeds,
        (a) the alignment stays well conserved, and
        (b) the candidate's normalized fit (see common.normalized_fit) is high enough.
   After expansion the same sequence can sit in several clusters, so clusters overlap.

B. MERGE CANDIDATES
   Two expanded clusters become a candidate pair when
     - "overlap path":      they share enough sequences, or
     - "score-signal path": they share 2+ sequences that the two profiles score similarly.

C. VALIDATION (best candidate pairs first)
   - Pre-merge:  align representatives of both clusters. Reject if the alignment is poorly
                 conserved, if neither cluster fits its own conserved columns, or if one
                 cluster fits far better than the other.
   - Post-merge: align the complete merged cluster. Reject if too many members fit badly.
   If both checks pass, the smaller cluster is absorbed into the larger one.

D. OUTPUT
   Merged clusters as FASTA files, plus logs explaining every decision.
   The script aborts rather than write output if any input sequence would be lost.

Example
-------
    python scripts/03_merge_clusters.py --clusters clusters/ --hits hits/ --out merged/ --log-dir logs/

Output
------
    <out>/<cluster>.fasta         merged clusters (the larger cluster of a merge keeps its name)
    <log-dir>/merge_decisions.csv one row per candidate pair tried: metrics, accept/reject, reason
    <log-dir>/merge_history.json  parameters, expansion statistics, list of accepted merges
    <log-dir>/expanded_clusters.json  members of each cluster after step A (for tracing sequences)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (CORE_GAP_TOLERANCE, HELP_CLUSTERS, HELP_OVERWRITE, Alignment, load_clusters,
                    load_hit_tables, make_parser, normalized_fit, pick_representatives, prepare_output_dir,
                    require_programs, run_clustalo, validate_cluster_alignment, write_clusters, write_fasta)

# A cluster is stored as a table with two columns:
#   Subject  sequence ID
#   Score    bit score of that sequence against the cluster's profile (NaN if it had no hit)
ClusterTable = pd.DataFrame
Candidate = Tuple[str, str, float, Dict]  # (cluster_1, cluster_2, combined_score, metrics)

# ---- fixed constants of the method (listed with explanations in docs/PARAMETERS.md) ------------
# Number of representatives used when scoring a candidate's fit during expansion.
EXPANSION_FIT_REPS = 40
# Combined weight of a cluster pair = OVERLAP_WEIGHT * overlap + CORRELATION_WEIGHT * max(corr, 0).
OVERLAP_WEIGHT = 0.7
CORRELATION_WEIGHT = 0.3


# =============================================================================
# Small helpers
# =============================================================================

def ranked_ids(table: ClusterTable) -> List[str]:
    """Sequence IDs of a cluster table, best-scoring first (ties keep their order)."""
    return table.sort_values("Score", ascending=False, kind="mergesort")["Subject"].tolist()


def unchanged_cluster(seed_ids) -> ClusterTable:
    """A cluster table for a cluster that is passed on without expansion."""
    ids = sorted(seed_ids)
    return pd.DataFrame({"Subject": ids, "Score": [np.nan] * len(ids)})


# =============================================================================
# A. Expansion
# =============================================================================

def filter_candidates_by_alignment(candidates: List[str], seed_hits: ClusterTable,
                                   seq_db: Dict[str, str], cfg) -> Tuple[List[str], Dict[str, int]]:
    """Keep only the candidates that really look like members of the seed cluster.

    Returns the accepted candidates and a count of why the others were rejected.
    """
    counts = {"rejected_consensus": 0, "rejected_fit": 0, "alignment_failed": 0}
    reps = pick_representatives(ranked_ids(seed_hits), cfg.rep_sample_size)
    if not reps or not candidates:
        return [], counts
    fit_reps = pick_representatives(ranked_ids(seed_hits), min(EXPANSION_FIT_REPS, len(seed_hits)))
    use_consensus_test = cfg.expand_consensus_drop > 0 and len(reps) >= 2  # one sequence cannot be aligned

    with TemporaryDirectory() as tmp:
        fasta, aligned = Path(tmp) / "test.fa", Path(tmp) / "test.aln.fa"
        rep_records = [(r, seq_db[r]) for r in reps]

        if use_consensus_test:
            # Reference: how conserved is the alignment of the seeds alone?
            write_fasta(fasta, rep_records)
            reference = run_clustalo(fasta, aligned) and Alignment.from_fasta(aligned)
            if reference:
                min_mean_consensus = float(np.mean(list(reference.member_fits().values()))) * cfg.expand_consensus_drop
            else:  # without a reference the conservation test cannot be applied; the fit test still is
                print("  WARNING: could not align the seed sequences; conservation test skipped for this cluster")
                use_consensus_test = False

        accepted = []
        for candidate in candidates:
            # (a) adding the candidate must not destroy the conservation of the alignment
            if use_consensus_test:
                write_fasta(fasta, rep_records + [(candidate, seq_db[candidate])])
                if not run_clustalo(fasta, aligned):
                    counts["alignment_failed"] += 1
                    continue
                alignment = Alignment.from_fasta(aligned)
                if alignment is None:
                    counts["alignment_failed"] += 1
                    continue
                if alignment.mean_consensus < min_mean_consensus:
                    counts["rejected_consensus"] += 1
                    continue
            # (b) the candidate must fit the cluster well enough
            fit = normalized_fit(candidate, seq_db, fit_reps, cluster_size=len(seed_hits))
            if fit is None:
                counts["alignment_failed"] += 1
            elif fit < cfg.expand_min_fit:
                counts["rejected_fit"] += 1
            else:
                accepted.append(candidate)
    return accepted, counts


def expand_cluster(seed_ids: List[str], hits: Optional[ClusterTable],
                   seq_db: Dict[str, str], cfg) -> Tuple[ClusterTable, Dict]:
    """Add similar sequences from other clusters to one cluster.

    Returns the expanded cluster table and a dictionary of statistics.
    A cluster that cannot be evaluated (no profile hits for its own sequences) is
    returned unchanged.
    """
    seeds = set(seed_ids)
    stats = {"seeds": len(seeds), "candidates": 0, "accepted": 0,
             "rejected_consensus": 0, "rejected_fit": 0, "alignment_failed": 0}

    seed_hits = hits[hits["Subject"].isin(seeds)] if hits is not None else None
    if seed_hits is None or seed_hits.empty or pd.isna(seed_hits["Score"].median()):
        return unchanged_cluster(seeds), stats

    median_score = seed_hits["Score"].median()
    low, high = median_score * (1 - cfg.score_window), median_score * (1 + cfg.score_window)
    in_window = hits["Score"].between(low, high) & ~hits["Subject"].isin(seeds)
    candidates = hits.loc[in_window, "Subject"].tolist()

    accepted, rejections = filter_candidates_by_alignment(candidates, seed_hits, seq_db, cfg)
    stats.update(candidates=len(candidates), accepted=len(accepted), **rejections)

    members = seeds | set(accepted)
    table = hits[hits["Subject"].isin(members)]
    without_hit = sorted(members - set(table["Subject"]))  # seeds the profile did not hit
    if without_hit:
        table = pd.concat([table, unchanged_cluster(without_hit)], ignore_index=True)
    return table.reset_index(drop=True), stats


# =============================================================================
# B. Merge candidates
# =============================================================================

def relative_scores(scores: pd.Series, reference: float) -> pd.Series:
    """Scores divided by a reference score (NaN scores are dropped).

    If the reference is not a positive number, nothing can be normalised and 0.0 is returned
    for each score (so that the pair does not pass the score conditions by accident).
    """
    scores = scores.dropna()
    return scores / reference if reference > 0 else scores * 0.0


def evaluate_pair(table1: ClusterTable, table2: ClusterTable, cfg) -> Tuple[bool, float, Dict]:
    """Decide whether two expanded clusters are similar enough to be tried for a merge.

    Returns (is_candidate, combined_score, metrics).
    """
    ids1, ids2 = set(table1["Subject"]), set(table2["Subject"])
    n_shared = len(ids1 & ids2)

    overlap = corr = mean_score = max_score = 0.0
    if n_shared:
        overlap = n_shared / min(len(ids1), len(ids2))

        # Put the two profiles' scores for the *same* sequences side by side.
        paired = table1.merge(table2, on="Subject", suffixes=("_1", "_2"))
        scored = paired.dropna(subset=["Score_1", "Score_2"])
        if len(scored) > 1:
            with np.errstate(invalid="ignore", divide="ignore"):  # constant scores give NaN -> treated as 0
                corr = float(np.nan_to_num(np.corrcoef(scored["Score_1"], scored["Score_2"])[0, 1]))

        # Bit scores depend on the profile (length, conservation), so the two profiles' scores
        # cannot be compared directly. Divide each profile's scores by the best score that profile
        # gives to any member of its own cluster: 1.0 then means "as good as its best member".
        # (The reference must come from outside the shared set; dividing by the best *shared*
        # score would make the best relative score 1.0 by definition.)
        relative1 = relative_scores(paired["Score_1"], table1["Score"].max())
        relative2 = relative_scores(paired["Score_2"], table2["Score"].max())
        mean_score = float(np.nan_to_num((relative1.mean() + relative2.mean()) / 2))
        both = np.concatenate([relative1, relative2])
        max_score = float(both.max()) if both.size else 0.0

    combined = overlap * OVERLAP_WEIGHT + max(corr, 0.0) * CORRELATION_WEIGHT

    overlap_path = (n_shared >= cfg.min_shared
                    and overlap >= cfg.min_overlap
                    and combined >= cfg.min_overlap)
    signal_path = (n_shared >= 1
                   and corr >= cfg.signal_min_corr
                   and mean_score >= cfg.signal_min_mean_score
                   and max_score >= cfg.signal_min_max_score)

    metrics = {
        "shared": n_shared,
        "overlap": overlap,
        "score_corr": corr,
        "combined_score": combined,
        "mean_relative_score": mean_score,
        "max_relative_score": max_score,
        "path": "signal" if signal_path and not overlap_path else "overlap",
    }
    return bool(overlap_path or signal_path), combined, metrics


def find_candidate_merges(clusters: Dict[str, ClusterTable], cfg) -> List[Candidate]:
    """All cluster pairs worth testing, best combined score first.

    Pairs that share no sequence can never qualify, so only pairs that co-occur
    in at least one sequence's cluster list are examined.
    """
    # Rank clusters by size, largest first: in each pair, the larger cluster is "cluster 1"
    # and is the one that survives a merge.
    order = sorted(clusters, key=lambda cid: len(clusters[cid]), reverse=True)
    rank = {cid: i for i, cid in enumerate(order)}

    clusters_of_sequence = defaultdict(list)
    for cid, table in clusters.items():
        for seq_id in table["Subject"]:
            clusters_of_sequence[seq_id].append(cid)

    pairs = set()
    for cids in clusters_of_sequence.values():
        for a, b in combinations(cids, 2):
            pairs.add((a, b) if rank[a] < rank[b] else (b, a))

    candidates: List[Candidate] = []
    for cid1, cid2 in sorted(pairs, key=lambda p: (rank[p[0]], rank[p[1]])):
        is_candidate, combined, metrics = evaluate_pair(clusters[cid1], clusters[cid2], cfg)
        if is_candidate:
            candidates.append((cid1, cid2, combined, metrics))
    return sorted(candidates, key=lambda c: c[2], reverse=True)


# =============================================================================
# C. Validation and merging
# =============================================================================

def pre_merge_check(table1: ClusterTable, table2: ClusterTable,
                    seq_db: Dict[str, str], cfg) -> Tuple[bool, Dict]:
    """Align representatives of both clusters and check that they belong together."""
    reps1 = pick_representatives(ranked_ids(table1), cfg.rep_sample_size)
    reps2 = pick_representatives(ranked_ids(table2), cfg.rep_sample_size)
    reps = list(dict.fromkeys(reps1 + reps2))  # a sequence shared by both clusters is aligned once

    with TemporaryDirectory() as tmp:
        fasta, aligned = Path(tmp) / "pair.fa", Path(tmp) / "pair.aln.fa"
        write_fasta(fasta, [(r, seq_db[r]) for r in reps])
        alignment = run_clustalo(fasta, aligned) and Alignment.from_fasta(aligned)
    if not alignment:
        return False, {"reason": "clustalo_failed"}

    fit1, fit2, n_core = alignment.core_group_fits(reps1, reps2, cfg.core_gap_tolerance)
    info = {"mean_consensus": alignment.mean_consensus, "fit_1": fit1, "fit_2": fit2,
            "core_columns": n_core, "shared_reps": len(set(reps1) & set(reps2)),
            "reps_1": len(reps1), "reps_2": len(reps2)}

    if alignment.mean_consensus < cfg.pre_min_consensus:
        info["reason"] = "low_consensus"            # the two clusters do not align well at all
    elif max(fit1, fit2) < cfg.pre_min_fit:
        info["reason"] = "no_core_fit"              # neither cluster fits the shared core columns
    elif abs(fit1 - fit2) > cfg.pre_max_fit_diff:
        info["reason"] = "fit_diff_too_large"       # one cluster would be swallowed by the other
    else:
        info["reason"] = "ok"
    return info["reason"] == "ok", info


def post_merge_check(member_ids: List[str], seq_db: Dict[str, str], cfg,
                     groups: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, Dict]:
    """Align the complete merged cluster and check that few members fit badly.

    The decision uses the bad fraction of the whole merged cluster. `groups` (the two clusters
    that are merged) only adds `bad_fraction_1` / `bad_fraction_2` to the log, so that you can see
    whether one of the two parts fits the merged consensus much worse than the other.
    """
    return validate_cluster_alignment(member_ids, seq_db, cfg.post_member_fit,
                                      cfg.post_max_bad_fraction, groups)


def merge_clusters(expanded: Dict[str, ClusterTable], candidates: List[Candidate],
                   seq_db: Dict[str, str], cfg) -> Tuple[Dict[str, ClusterTable], List[Dict], List[Dict]]:
    """Try every candidate pair, best first, and merge the ones that pass both checks.

    Returns (final clusters, one decision row per pair tried, list of accepted merges).
    """
    clusters = {cid: table.copy() for cid, table in expanded.items()}
    decisions: List[Dict] = []
    merges: List[Dict] = []

    for number, (cid1, cid2, _, metrics) in enumerate(candidates, start=1):
        if cid1 not in clusters or cid2 not in clusters:
            continue  # one of them was absorbed by an earlier merge

        table1, table2 = clusters[cid1], clusters[cid2]
        row = {"cluster_1": cid1, "cluster_2": cid2, "size_1": len(table1), "size_2": len(table2),
               **{k: metrics[k] for k in ("path", "shared", "overlap", "score_corr", "combined_score",
                                    "mean_relative_score", "max_relative_score")}}

        ok, pre = pre_merge_check(table1, table2, seq_db, cfg)
        row.update({f"pre_{k}": v for k, v in pre.items() if k != "reason"})
        if not ok:
            decisions.append({**row, "decision": "reject", "reason": f"pre_merge:{pre['reason']}"})
            continue

        merged = pd.concat([table1, table2], ignore_index=True).drop_duplicates("Subject")
        ok, post = post_merge_check(merged["Subject"].tolist(), seq_db, cfg,
                                    groups={"1": table1["Subject"].tolist(), "2": table2["Subject"].tolist()})
        row.update({f"post_{k}": v for k, v in post.items() if k != "reason"})
        if not ok:
            decisions.append({**row, "decision": "reject", "reason": f"post_merge:{post['reason']}"})
            continue

        merged = merged.sort_values("Score", ascending=False, kind="mergesort").reset_index(drop=True)
        clusters[cid1] = merged
        del clusters[cid2]
        merges.append({"kept": cid1, "absorbed": cid2, "size_after": len(merged)})
        decisions.append({**row, "decision": "accept", "reason": "ok", "merged_size": len(merged)})
        print(f"  [{number}/{len(candidates)}] merged {cid2} into {cid1} -> {len(merged)} sequences")

    return clusters, decisions, merges


# =============================================================================
# Command line and main program
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("Expand clusters with similar sequences and merge similar clusters.")

    io = parser.add_argument_group("input / output")
    io.add_argument("--clusters", required=True, help=HELP_CLUSTERS)
    io.add_argument("--hits", required=True,
                    help="Folder with the hit tables (CSV) written by step 2, made with profiles of "
                         "exactly these clusters.")
    io.add_argument("--out", default="merged_clusters",
                    help="Folder where the merged cluster FASTA files are written. The larger cluster "
                         "of a merge keeps its name.")
    io.add_argument("--log-dir", default="logs",
                    help="Folder for merge_decisions.csv (one row per pair tried), merge_history.json "
                         "(parameters, statistics, accepted merges) and expanded_clusters.json.")
    io.add_argument("--overwrite", action="store_true", help=HELP_OVERWRITE)

    expand = parser.add_argument_group("A. expansion (adding sequences from other clusters)")
    expand.add_argument("--score-window", type=float, default=0.35,
                        help="A sequence from another cluster becomes a candidate if its bit score "
                             "against this cluster's profile lies within plus or minus this fraction of "
                             "the median score of the cluster's own members (0.35 means 65% to 135% of "
                             "the median). Higher values admit more, and more weakly related, candidates.")
    expand.add_argument("--expand-min-fit", type=float, default=0.24,
                        help="Minimum normalized fit (see 'How fit is measured' in the README; roughly "
                             "0 to 1) that a candidate needs to join the cluster. Higher is stricter. "
                             "Clusters with fewer than about 50 members are penalised by the size "
                             "factor, so they need a correspondingly higher raw fit.")
    expand.add_argument("--expand-consensus-drop", type=float, default=0.65,
                        help="Conservation test: after the candidate is added to the aligned "
                             "representatives, the alignment's mean consensus strength must stay at or "
                             "above this fraction of the seeds' mean fit. On simulated data this test "
                             "almost never rejected anything; the 'expansion totals' line printed by this "
                             "script shows whether it does on yours. 0 switches it off, which also saves "
                             "one alignment per candidate.")
    expand.add_argument("--rep-sample-size", type=int, default=50,
                        help="Maximum number of cluster members that are aligned together in the "
                             "expansion tests and in the pre-merge check: the best-scoring 85% of the "
                             "slots plus an even spread over the rest of the ranking. Larger values "
                             "give a more reliable consensus but slower alignments.")

    cand = parser.add_argument_group("B. merge candidates (which pairs of clusters are tried)")
    cand.add_argument("--min-overlap", type=float, default=0.12,
                      help="Overlap path: shared sequences divided by the size of the smaller cluster "
                           "must reach this value, and so must the combined weight (0.7 x overlap + "
                           "0.3 x score correlation, negative correlations counted as 0). Because the "
                           "correlation is usually not positive, the effective minimum overlap is "
                           "about this value divided by 0.7. Lower values try more pairs; 0 tries every "
                           "pair that shares --min-shared sequences.")
    cand.add_argument("--min-shared", type=int, default=1,
                      help="Overlap path: minimum number of sequences the two expanded clusters must "
                           "share.")
    cand.add_argument("--signal-min-corr", type=float, default=0.75,
                      help="Score-signal path: minimum Pearson correlation between the two profiles' "
                           "scores of the shared sequences. At least 2 shared sequences with scores "
                           "are needed, and with very few the value is unreliable (2 points always "
                           "give exactly +1 or -1). A value above 1 switches this path off.")
    cand.add_argument("--signal-min-mean-score", type=float, default=0.35,
                      help="Score-signal path: minimum mean score of the shared sequences, as a "
                           "fraction of the best score each profile gives to its own members "
                           "(0 to 1).")
    cand.add_argument("--signal-min-max-score", type=float, default=0.55,
                      help="Score-signal path: at least one shared sequence must reach this fraction "
                           "of the best score a profile gives to its own members (0 to 1). Rejects "
                           "pairs that are linked only by weak hits.")

    pre = parser.add_argument_group("C1. pre-merge check (representatives of both clusters aligned together)")
    pre.add_argument("--pre-min-consensus", type=float, default=0.28,
                     help="Minimum mean consensus strength (average over alignment columns of the "
                          "share of the most common residue; 0 to 1) of the joint alignment. Low for "
                          "unrelated sequences. Higher is stricter.")
    pre.add_argument("--pre-min-fit", type=float, default=0.24,
                     help="The better of the two clusters must reach this mean fit (share of residues "
                          "equal to the joint consensus, core columns only; 0 to 1). Higher is stricter.")
    pre.add_argument("--pre-max-fit-diff", type=float, default=0.55,
                     help="Maximum allowed difference between the two clusters' mean fits. A large "
                          "difference means one cluster would be swallowed by the other. Higher is more "
                          "permissive; 1 switches this test off.")
    pre.add_argument("--core-gap-tolerance", type=float, default=CORE_GAP_TOLERANCE,
                     help="A column of an alignment is a 'core' column if fewer than this fraction of "
                          "the sequences of a cluster have a gap there. Core columns are used to compare "
                          "the two clusters here, and for the coverage term of the fit score "
                          "everywhere in the pipeline.")

    post = parser.add_argument_group("C2. post-merge check (the complete merged cluster aligned)")
    post.add_argument("--post-member-fit", type=float, default=0.22,
                      help="A member whose fit to the merged cluster's consensus is below this value "
                           "counts as a bad member.")
    post.add_argument("--post-max-bad-fraction", type=float, default=0.22,
                      help="The merge is rejected if more than this fraction of all members of the "
                           "merged cluster are bad. Higher is more permissive. Note that this limits "
                           "the share of bad members overall, not within each of the two clusters.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    cfg = parse_args()
    require_programs("clustalo")
    out_dir = prepare_output_dir(cfg.out, cfg.overwrite)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- load -----------------------------------------------------------------
    clusters, seq_db = load_clusters(cfg.clusters)
    hits = load_hit_tables(Path(cfg.hits), universe=set(seq_db))
    print(f"Loaded {len(clusters)} clusters ({len(seq_db)} sequences) and {len(hits)} hit tables.")

    # ---- A. expansion -----------------------------------------------------------
    print("A. Expanding clusters...")
    expanded: Dict[str, ClusterTable] = {}
    expansion_stats: Dict[str, Dict] = {}
    for number, (cid, seed_ids) in enumerate(clusters.items(), start=1):
        expanded[cid], expansion_stats[cid] = expand_cluster(seed_ids, hits.get(cid), seq_db, cfg)
        s = expansion_stats[cid]
        if s["candidates"]:
            print(f"  [{number}/{len(clusters)}] {cid}: seeds={s['seeds']} "
                  f"candidates={s['candidates']} accepted={s['accepted']}")

    totals = {k: sum(st[k] for st in expansion_stats.values())
              for k in ("candidates", "accepted", "rejected_consensus", "rejected_fit", "alignment_failed")}
    print("  expansion totals: " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    if totals["alignment_failed"]:
        print(f"  WARNING: {totals['alignment_failed']} alignments failed; those candidates were rejected. "
              f"If this is many, check that Clustal Omega works.")

    expanded_members = {cid: table["Subject"].tolist() for cid, table in expanded.items()}
    (log_dir / "expanded_clusters.json").write_text(json.dumps(expanded_members, indent=2))

    # ---- B. merge candidates ----------------------------------------------------
    print("B. Looking for merge candidates...")
    candidates = find_candidate_merges(expanded, cfg)
    print(f"  {len(candidates)} candidate pairs")

    # ---- C. validate and merge --------------------------------------------------
    print("C. Validating and merging...")
    final, decisions, merges = merge_clusters(expanded, candidates, seq_db, cfg)
    print(f"  {len(merges)} merges accepted; {len(final)} clusters remain")

    # ---- D. verify and write ------------------------------------------------------
    sequences_out = set().union(*(set(table["Subject"]) for table in final.values()))
    lost = set(seq_db) - sequences_out
    if lost:
        sys.exit(f"ERROR: {len(lost)} sequences would be lost (e.g. {sorted(lost)[:5]}). Nothing written.")

    ordered = {cid: table.sort_values("Score", ascending=False, kind="mergesort")["Subject"].tolist()
               for cid, table in final.items()}
    n_written = write_clusters(out_dir, ordered, seq_db)

    pd.DataFrame(decisions).round(4).to_csv(log_dir / "merge_decisions.csv", index=False)
    history = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(cfg),
        "clusters_in": len(clusters),
        "clusters_out": len(final),
        "sequences": len(seq_db),
        "expansion": expansion_stats,
        "merges": merges,
    }
    (log_dir / "merge_history.json").write_text(json.dumps(history, indent=2, default=str))

    print(f"Verified: all {len(seq_db)} sequences are present in the output.")
    print(f"Wrote {n_written} cluster files to {out_dir} and logs to {log_dir}")


if __name__ == "__main__":
    main()
