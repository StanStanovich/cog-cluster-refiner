#!/usr/bin/env python3
"""
Step 2 - search every sequence against every cluster profile and save the hits as tables.

Runs `hmmsearch` once per (profile, target FASTA) pair and writes one CSV per profile.
The targets should contain every sequence that is currently in a cluster.

Example
-------
    python scripts/02_search_hmms.py --profiles profiles/ --targets all_sequences.fasta --out hits/

Output
------
    <out>/<profile>.hits.csv   one row per hit, best score first, with columns

        query_profile   name of the profile (= cluster ID)
        target_seq      ID of the matching sequence
        accession       target accession reported by HMMER ("-" if none)
        evalue          full-sequence E-value
        score           full-sequence bit score
        bias            bias correction of the score
        target_file     which target FASTA the hit came from

Profiles without any hit below the E-value cutoff get no file.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import make_parser, require_programs

# Number of whitespace-separated columns before the free-text description in a
# HMMER --tblout line.
TBLOUT_FIXED_COLUMNS = 18


def run_hmmsearch(profile: Path, target: Path, tblout: Path, evalue: float, cpu: int) -> None:
    """Run hmmsearch and write the per-sequence hit table to `tblout`."""
    cmd = ["hmmsearch", "--cpu", str(cpu), "-E", str(evalue), "--tblout", str(tblout),
           str(profile), str(target)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hmmsearch failed for {profile.name} vs {target.name}:\n{result.stderr}")


def parse_tblout(tblout: Path, profile_name: str, target_name: str) -> List[Dict]:
    """Turn a HMMER --tblout file into a list of row dictionaries."""
    rows = []
    with open(tblout) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < TBLOUT_FIXED_COLUMNS:
                continue
            # tblout columns: 0 target name, 1 target accession, 2 query name, 3 query accession,
            #                 4 full-sequence E-value, 5 full-sequence score, 6 full-sequence bias, ...
            rows.append({
                "query_profile": profile_name,
                "target_seq": fields[0],
                "accession": fields[1],
                "evalue": float(fields[4]),
                "score": float(fields[5]),
                "bias": float(fields[6]),
                "target_file": target_name,
            })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__.split("\n\n")[0])
    io = parser.add_argument_group("input / output")
    io.add_argument("--profiles", required=True,
                    help="Folder with the .hmm profiles written by step 1; every profile is searched.")
    io.add_argument("--targets", required=True, nargs="+",
                    help="One or more FASTA files to search with every profile. Include every sequence "
                         "that is currently in a cluster (run_pipeline.py writes this file for you); "
                         "when unclustered sequences are to be added (step 5) include them as well. "
                         "The hit tables record which file each hit came from.")
    io.add_argument("--out", default="hits", help="Folder where one CSV hit table per profile is written.")

    run = parser.add_argument_group("settings")
    run.add_argument("--evalue", type=float, default=1e-5,
                     help="Report only hits whose full-sequence E-value is at or below this. Lower means "
                          "fewer, more reliable hits. Steps 3 and 5 can only use hits that are reported "
                          "here, so a very strict value limits how far clusters can grow.")
    run.add_argument("--cpu", type=int, default=4,
                     help="CPU threads used by each hmmsearch run. Profiles are searched one after "
                          "the other, so this is the only parallelism in this step.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    require_programs("hmmsearch")
    profiles = sorted(Path(args.profiles).glob("*.hmm"))
    if not profiles:
        sys.exit(f"ERROR: no .hmm files found in {args.profiles}")
    targets = [Path(t) for t in args.targets]
    for target in targets:
        if not target.is_file():
            sys.exit(f"ERROR: target FASTA not found: {target}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_with_hits = 0
    for number, profile in enumerate(profiles, start=1):
        name = profile.stem
        rows: List[Dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            for target in targets:
                tblout = Path(tmp) / "hits.tbl"
                run_hmmsearch(profile, target, tblout, args.evalue, args.cpu)
                rows.extend(parse_tblout(tblout, name, target.name))

        if rows:
            table = pd.DataFrame(rows).sort_values("score", ascending=False)
            table.to_csv(out_dir / f"{name}.hits.csv", index=False)
            n_with_hits += 1
        else:
            print(f"  no hits for {name}")

        if number % 100 == 0 or number == len(profiles):
            print(f"Searched {number}/{len(profiles)} profiles")

    print(f"Wrote hit tables for {n_with_hits} of {len(profiles)} profiles to {out_dir}")


if __name__ == "__main__":
    main()
