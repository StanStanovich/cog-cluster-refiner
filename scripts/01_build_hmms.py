#!/usr/bin/env python3
"""
Step 1 - build one HMMER profile per cluster.

For every FASTA file in the cluster folder (one file = one cluster):
    * 2+ sequences -> align them (Clustal Omega or MAFFT), then run `hmmbuild`
    * 1 sequence   -> run `hmmbuild` directly on that sequence

Example
-------
    python scripts/01_build_hmms.py --clusters clusters/ --profiles profiles/

Output
------
    <alignments>/<cluster>.aln.fasta   multiple sequence alignment (clusters with 2+ sequences)
    <profiles>/<cluster>.hmm           HMMER3 profile; the profile name is the cluster name
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import (HELP_CLUSTERS, list_fasta_files, make_parser, read_fasta_records, require_programs,
                    run_clustalo)


def align_with_mafft(fasta: Path, out_aln: Path, threads: int) -> bool:
    """Align with MAFFT (it writes the alignment to stdout)."""
    try:
        with open(out_aln, "w") as out:
            subprocess.run(["mafft", "--auto", "--thread", str(threads), str(fasta)],
                           stdout=out, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_profile(alignment: Path, hmm_out: Path, name: str, threads: int) -> bool:
    """Run `hmmbuild` on an aligned FASTA file (a single sequence counts as an alignment)."""
    cmd = ["hmmbuild", "--amino", "--informat", "afa", "--cpu", str(threads),
           "-n", name, str(hmm_out), str(alignment)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  hmmbuild failed for {name}: {result.stderr.strip() or result.stdout.strip()}")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__.split("\n\n")[0])
    io = parser.add_argument_group("input / output")
    io.add_argument("--clusters", required=True, help=HELP_CLUSTERS)
    io.add_argument("--alignments", default="alignments",
                    help="Folder where the multiple sequence alignment of each cluster with 2 or more "
                         "sequences is written (<cluster>.aln.fasta).")
    io.add_argument("--profiles", default="profiles",
                    help="Folder where the HMMER profile of each cluster is written (<cluster>.hmm). "
                         "The profile is named after the cluster; step 2 relies on that.")

    run = parser.add_argument_group("settings")
    run.add_argument("--aligner", choices=["clustalo", "mafft"], default="clustalo",
                     help="Program that aligns clusters with 2 or more sequences before hmmbuild. "
                          "Clustal Omega is also used by the later steps; MAFFT has to be installed "
                          "separately and is only an alternative for this step.")
    run.add_argument("--threads", type=int, default=1,
                     help="CPU threads used by the aligner and by hmmbuild.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    require_programs(args.aligner, "hmmbuild")
    Path(args.alignments).mkdir(parents=True, exist_ok=True)
    Path(args.profiles).mkdir(parents=True, exist_ok=True)

    fasta_files = list_fasta_files(args.clusters)
    if not fasta_files:
        sys.exit(f"ERROR: no FASTA files found in {args.clusters}")

    built, failed = 0, []
    for fasta in fasta_files:
        name = fasta.stem
        n_seqs = sum(1 for _ in read_fasta_records(fasta))
        hmm_path = Path(args.profiles) / f"{name}.hmm"

        if n_seqs == 0:
            print(f"Skipping {fasta.name}: no sequences")
            continue

        if n_seqs == 1:
            alignment_path = fasta  # nothing to align
        else:
            alignment_path = Path(args.alignments) / f"{name}.aln.fasta"
            if args.aligner == "mafft":
                aligned_ok = align_with_mafft(fasta, alignment_path, args.threads)
            else:
                aligned_ok = run_clustalo(fasta, alignment_path, args.threads)
            if not aligned_ok:
                print(f"  alignment failed for {name} ({n_seqs} sequences)")
                failed.append(name)
                continue

        if build_profile(alignment_path, hmm_path, name, args.threads):
            built += 1
        else:
            failed.append(name)

    print(f"Built {built} profiles from {len(fasta_files)} clusters; {len(failed)} failed.")
    if failed:
        print("Failed clusters (they will be left unchanged by later steps): " + ", ".join(failed))
    if built == 0 or len(failed) > len(fasta_files) / 2:
        sys.exit("ERROR: more than half of the clusters failed. This usually means that Clustal Omega or "
                 "HMMER is not working: try running one of the commands by hand on a single cluster.")


if __name__ == "__main__":
    main()
