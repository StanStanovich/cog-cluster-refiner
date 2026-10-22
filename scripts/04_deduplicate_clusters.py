#!/usr/bin/env python3
"""
Step 4 - make sure every sequence ends up in exactly one cluster.

After step 3 a sequence can be a member of several clusters. For each such sequence
this script
    1. measures how well it fits each of the clusters it is in (see common.normalized_fit),
    2. keeps it in the best-fitting cluster and removes it from the others.

The fit is computed against a random sample of at most `--sample-size` cluster members.
The sample is reproducible: it depends only on `--seed`, the cluster name and the
sequence ID.

Duplicates are processed in alphabetical order and removals take effect immediately, so
later decisions see the already-cleaned clusters. Ties are broken by cluster name.
Clusters that end up empty are not written.

Example
-------
    python scripts/04_deduplicate_clusters.py --clusters merged/ --out dedup/ --log-dir logs/

Output
------
    <out>/<cluster>.fasta         cleaned clusters
    <log-dir>/dedup_report.tsv    one row per duplicated sequence: where it was kept and what it fit
"""

import argparse
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import (HELP_CLUSTERS, HELP_OVERWRITE, load_clusters, make_parser, normalized_fit,
                    prepare_output_dir, require_programs, write_clusters)


def stable_seed(*parts) -> int:
    """A seed that is identical on every run and machine.

    (Python's built-in hash() of a string changes between runs, so it is not suitable.)
    """
    return zlib.crc32("|".join(map(str, parts)).encode())


def fit_to_cluster(seq_id: str, cluster_id: str, members: List[str], seq_db: Dict[str, str],
                   sample_size: int, seed: int) -> float:
    """Normalized fit of `seq_id` to a cluster, using a reproducible random sample of members."""
    others = [m for m in members if m != seq_id]  # a sequence must not be judged against itself
    rng = random.Random(stable_seed(seed, cluster_id, seq_id))
    sample = others if len(others) <= sample_size else rng.sample(others, sample_size)
    fit = normalized_fit(seq_id, seq_db, sample, cluster_size=len(members))
    return 0.0 if fit is None else fit


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__.split("\n\n")[0])
    io = parser.add_argument_group("input / output")
    io.add_argument("--clusters", required=True, help=HELP_CLUSTERS)
    io.add_argument("--out", default="dedup_clusters",
                    help="Folder where the cleaned cluster FASTA files are written. Clusters that end "
                         "up empty are not written.")
    io.add_argument("--log-dir", default="logs",
                    help="Folder where dedup_report.tsv is written: for every duplicated sequence the "
                         "cluster it was kept in and its fit to each cluster it was removed from.")
    io.add_argument("--overwrite", action="store_true", help=HELP_OVERWRITE)

    run = parser.add_argument_group("settings")
    run.add_argument("--sample-size", type=int, default=200,
                     help="Maximum number of members of a cluster that are aligned together with the "
                          "duplicated sequence to compute its fit. Larger clusters are sampled at "
                          "random. Larger values are more reliable but slower.")
    run.add_argument("--seed", type=int, default=1,
                     help="Seed of the random member sampling. The same seed and the same input always "
                          "give the same sample, so results are reproducible.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    require_programs("clustalo")
    out_dir = prepare_output_dir(args.out, args.overwrite)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    clusters, seq_db = load_clusters(args.clusters)
    clusters_of_sequence = defaultdict(list)
    for cluster_id, members in clusters.items():
        for seq_id in members:
            clusters_of_sequence[seq_id].append(cluster_id)
    duplicates = {s: c for s, c in clusters_of_sequence.items() if len(c) > 1}
    print(f"Loaded {len(clusters)} clusters with {len(seq_db)} unique sequences; "
          f"{len(duplicates)} sequences are in more than one cluster.")

    # ---- resolve duplicates one by one ---------------------------------------------
    report = []
    n_removed = 0
    for number, (seq_id, cluster_ids) in enumerate(sorted(duplicates.items()), start=1):
        fits = {cid: fit_to_cluster(seq_id, cid, clusters[cid], seq_db, args.sample_size, args.seed)
                for cid in cluster_ids}
        best = max(fits, key=lambda cid: (fits[cid], cid))  # highest fit; ties -> larger cluster name

        for cid in cluster_ids:
            if cid != best:
                clusters[cid] = [m for m in clusters[cid] if m != seq_id]
                n_removed += 1

        report.append({
            "sequence_id": seq_id,
            "kept_in": best,
            "kept_fit": round(fits[best], 4),
            "removed_from": ";".join(f"{cid}:{fits[cid]:.4f}" for cid in cluster_ids if cid != best),
        })
        if number % 50 == 0 or number == len(duplicates):
            print(f"  resolved {number}/{len(duplicates)} duplicated sequences")

    # ---- verify, then write -----------------------------------------------------------
    sequences_out = {seq_id for members in clusters.values() for seq_id in members}
    lost = set(seq_db) - sequences_out
    if lost:
        sys.exit(f"ERROR: {len(lost)} sequences would be lost (e.g. {sorted(lost)[:5]}). Nothing written.")

    n_written = write_clusters(out_dir, clusters, seq_db)
    pd.DataFrame(report, columns=["sequence_id", "kept_in", "kept_fit", "removed_from"]) \
        .to_csv(log_dir / "dedup_report.tsv", sep="\t", index=False)

    print(f"Removed {n_removed} duplicate placements. Verified: all {len(seq_db)} sequences are present.")
    print(f"Wrote {n_written} cluster files ({len(clusters) - n_written} became empty) to {out_dir}")


if __name__ == "__main__":
    main()
