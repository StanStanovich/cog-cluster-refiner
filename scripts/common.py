"""
common.py - helpers shared by the pipeline scripts.

What lives here
---------------
* Minimal FASTA reading / writing (no Biopython needed).
* Loading a folder of cluster FASTA files (one file = one cluster).
* Loading the hit tables written by step 2.
* A thin wrapper around Clustal Omega.
* `Alignment`: a multiple sequence alignment with per-column consensus
  statistics, used to judge how well a sequence "fits" a group of sequences.
* `normalized_fit`: the score that decides whether a sequence belongs in a cluster.
* `validate_cluster_alignment`: checks that a whole cluster aligns well.

Vocabulary used throughout the project
--------------------------------------
consensus residue   The most common non-gap residue in an alignment column
                    (ties are broken alphabetically).
consensus strength  Fraction of the non-gap residues in a column that equal the
                    consensus residue (1.0 = column is perfectly conserved).
fit (of a sequence) Fraction of the sequence's residues that equal the consensus
                    residue of the column they sit in.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FASTA_SUFFIXES = (".fasta", ".fa", ".faa")
GAP = "-"

# ---- fixed constants of the method (listed with explanations in docs/PARAMETERS.md) ------------
# Share of the representative slots given to the best-scoring members (see pick_representatives).
TOP_FRACTION = 0.85
# A cluster with this many members (or more) gets no size penalty in `normalized_fit`.
SIZE_FACTOR_FULL_AT = 50
# Weight of the coverage term in `normalized_fit`: factor = (1 - weight) + weight * coverage.
COVERAGE_WEIGHT = 0.5
# Weight of the length-ratio term in `normalized_fit`: factor = (1 - weight) + weight * ratio.
LENGTH_RATIO_WEIGHT = 0.3

# A column is a "core" column of a group of sequences if fewer than this fraction of them have a
# gap there. One definition, used both for coverage and for the pre-merge comparison of two clusters.
CORE_GAP_TOLERANCE = 0.6


# =============================================================================
# Command-line helpers
# =============================================================================

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter, but shows "(default: ...)" only where it is informative."""

    def _get_help_string(self, action):
        text = (action.help or "").replace("%", "%%")  # argparse expands % in help; write a literal %
        informative = (action.default is not None and action.default is not False
                       and action.default != argparse.SUPPRESS and not action.required)
        return text + " (default: %(default)s)" if informative else text


HELP_CLUSTERS = ("Folder with one FASTA file per cluster (.fasta, .fa or .faa). The file name without "
                 "extension is the cluster ID. Sequences must be proteins; a sequence's ID is the first "
                 "word of its header line.")
HELP_OVERWRITE = ("Delete FASTA files that are already in the output folder. Without this option the "
                  "script refuses to write into a folder that contains FASTA files, so that results of "
                  "different runs are never mixed.")


def make_parser(description: str) -> argparse.ArgumentParser:
    """An argument parser with the formatting used by all scripts (no silent option abbreviations)."""
    return argparse.ArgumentParser(description=description, formatter_class=HelpFormatter,
                                   allow_abbrev=False)


# =============================================================================
# FASTA input / output
# =============================================================================

def read_fasta_records(path) -> Iterator[Tuple[str, str]]:
    """Yield (sequence_id, sequence) for every record in a FASTA file.

    The ID is the first word of the header line, i.e. ">abc some text" -> "abc".
    """
    seq_id: Optional[str] = None
    chunks: List[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks)
                words = line[1:].split(maxsplit=1)
                seq_id, chunks = (words[0] if words else ""), []
            else:
                chunks.append(line)
    if seq_id is not None:
        yield seq_id, "".join(chunks)


def write_fasta(path, records: Iterable[Tuple[str, str]], line_width: int = 60) -> None:
    """Write (header, sequence) pairs to a FASTA file. `header` is written after '>'."""
    with open(path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for start in range(0, len(seq), line_width):
                out.write(seq[start:start + line_width] + "\n")


def list_fasta_files(folder) -> List[Path]:
    """All FASTA files directly inside `folder`, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: directory not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in FASTA_SUFFIXES)


def load_clusters(folder) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Read a folder of cluster FASTA files (one file per cluster).

    Returns
    -------
    clusters : {cluster_id: [sequence_id, ...]}   cluster_id = file name without extension
    seq_db   : {sequence_id: sequence}            every sequence in every cluster
    """
    clusters: Dict[str, List[str]] = {}
    seq_db: Dict[str, str] = {}
    n_conflicts = 0

    for path in list_fasta_files(folder):
        members: Dict[str, None] = {}  # used as an ordered set
        for seq_id, seq in read_fasta_records(path):
            members[seq_id] = None
            if seq_id in seq_db and seq_db[seq_id] != seq:
                n_conflicts += 1  # same ID, different sequence: keep the first one
            seq_db.setdefault(seq_id, seq)
        if members:
            clusters[path.stem] = list(members)

    if not clusters:
        sys.exit(f"ERROR: no FASTA files with sequences found in {folder}")
    if n_conflicts:
        print(f"WARNING: {n_conflicts} sequence IDs occur with different sequences; "
              f"the first occurrence was used.", file=sys.stderr)
    return clusters, seq_db


def write_clusters(folder, clusters: Dict[str, Sequence[str]], seq_db: Dict[str, str]) -> int:
    """Write one FASTA file per non-empty cluster; headers are '>sequence_id cluster_id'.

    Returns the number of files written.
    """
    folder = Path(folder)
    written = 0
    for cluster_id, members in clusters.items():
        if not members:
            continue
        write_fasta(folder / f"{cluster_id}.fasta", ((f"{sid} {cluster_id}", seq_db[sid]) for sid in members))
        written += 1
    return written


def prepare_output_dir(folder, overwrite: bool = False) -> Path:
    """Create an output folder for cluster FASTA files.

    Refuses to mix new output with FASTA files left over from an earlier run,
    unless `overwrite` is True (then the old FASTA files are deleted).
    """
    folder = Path(folder)
    leftovers = [p for p in folder.iterdir() if p.suffix.lower() in FASTA_SUFFIXES] if folder.is_dir() else []
    if leftovers and not overwrite:
        sys.exit(f"ERROR: {folder} already contains {len(leftovers)} FASTA files. "
                 f"Choose another folder or pass --overwrite.")
    for path in leftovers:
        path.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =============================================================================
# Hit tables (written by step 2) and representative sampling
# =============================================================================

def load_hit_tables(hits_dir, universe: set) -> Dict[str, pd.DataFrame]:
    """Read the CSV tables from step 2 -> {cluster_id: table with columns Subject, Score}.

    Only sequences in `universe` are kept, and each sequence keeps its best score.
    Rows are sorted best score first.
    """
    tables: Dict[str, pd.DataFrame] = {}
    paths = sorted(Path(hits_dir).glob("*.csv"))
    if not paths:
        sys.exit(f"ERROR: no hit tables (*.csv) found in {hits_dir}. Run step 2 first, or check the folder name.")
    for path in paths:
        try:
            df = pd.read_csv(path, usecols=["query_profile", "target_seq", "score"],
                             dtype={"query_profile": str, "target_seq": str})
        except ValueError:
            sys.exit(f"ERROR: {path} is not a hit table from step 2 "
                     f"(expected the columns query_profile, target_seq and score).")
        if df.empty:
            continue
        cluster_id = df["query_profile"].iloc[0]
        df = df.rename(columns={"target_seq": "Subject", "score": "Score"})
        df = df[df["Subject"].isin(universe)][["Subject", "Score"]]
        df = df.sort_values("Score", ascending=False, kind="mergesort").drop_duplicates("Subject")
        tables[cluster_id] = df.reset_index(drop=True)
    return tables


def pick_representatives(ranked_ids: Sequence[str], n: int) -> List[str]:
    """Choose up to `n` sequences to stand in for a cluster.

    `ranked_ids` must be sorted best first. 85% of the slots go to the best-ranked
    members. The remaining slots are spread evenly over the whole ranking so that
    weaker, more divergent members are included too.
    """
    ranked_ids = list(ranked_ids)
    if len(ranked_ids) <= n:
        return ranked_ids
    n_top = max(1, int(n * TOP_FRACTION))
    n_spread = n - n_top
    top = ranked_ids[:n_top]
    spread = ranked_ids[::max(1, len(ranked_ids) // n_spread)][:n_spread] if n_spread else []
    return list(dict.fromkeys(top + spread))[:n]


# =============================================================================
# External programs
# =============================================================================

def require_programs(*names: str) -> None:
    """Exit with a helpful message if any of the given programs is not on the PATH."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        sys.exit(f"ERROR: could not find {', '.join(missing)} on your PATH.\n"
                 f"Install with:  conda install -c bioconda hmmer clustalo")


def run_clustalo(in_fasta, out_aln, threads: int = 1) -> bool:
    """Align `in_fasta` with Clustal Omega, writing aligned FASTA to `out_aln`.

    Returns True on success, False if Clustal Omega failed (callers treat that as
    "could not evaluate" rather than crashing the whole run).
    """
    cmd = ["clustalo", "-i", str(in_fasta), "-o", str(out_aln),
           "--force", "--outfmt=fasta", f"--threads={threads}"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# =============================================================================
# Alignment statistics
# =============================================================================

class Alignment:
    """A multiple sequence alignment held as a character matrix, with column consensus.

    Rows are sequences, columns are alignment positions. The consensus of every
    column is computed once, when the object is created.
    """

    def __init__(self, ids: Sequence[str], seqs: Sequence[str]):
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sequence IDs in alignment")
        if not seqs or len({len(s) for s in seqs}) != 1 or len(seqs[0]) == 0:
            raise ValueError("alignment is empty or its sequences differ in length")

        self.ids: List[str] = list(ids)
        self.index: Dict[str, int] = {sid: i for i, sid in enumerate(self.ids)}
        self.matrix = np.array([list(s.upper()) for s in seqs])  # (n_sequences, n_columns); case-insensitive
        self._compute_consensus()

    @classmethod
    def from_fasta(cls, path) -> Optional["Alignment"]:
        """Load an aligned FASTA file. Returns None if it is missing or malformed."""
        try:
            records = list(read_fasta_records(path))
            return cls([sid for sid, _ in records], [seq for _, seq in records])
        except (OSError, ValueError):
            return None

    # ---- column consensus ---------------------------------------------------

    def _compute_consensus(self) -> None:
        residue_types = [c for c in np.unique(self.matrix) if c != GAP]  # sorted alphabetically
        n_columns = self.matrix.shape[1]

        if not residue_types:  # alignment consists only of gaps
            self.consensus = np.full(n_columns, "", dtype="<U1")
            self.has_consensus = np.zeros(n_columns, dtype=bool)
            self.consensus_strength = np.zeros(n_columns)
            return

        # counts[k, j] = how many sequences have residue type k in column j
        counts = np.stack([(self.matrix == r).sum(axis=0) for r in residue_types])
        residues_per_column = counts.sum(axis=0)

        self.has_consensus = residues_per_column > 0
        self.consensus = np.array(residue_types)[counts.argmax(axis=0)]  # ties -> first (alphabetical)
        self.consensus_strength = np.where(
            self.has_consensus, counts.max(axis=0) / np.maximum(residues_per_column, 1), 0.0)

    @property
    def mean_consensus(self) -> float:
        """Average consensus strength over all columns: how conserved the alignment is overall."""
        return float(self.consensus_strength.mean())

    # ---- fit of individual sequences -----------------------------------------

    def _fits(self, rows: Sequence[int], column_mask: np.ndarray) -> np.ndarray:
        """Fit of each given row, looking only at the selected columns."""
        block = self.matrix[rows][:, column_mask]
        consensus = self.consensus[column_mask]
        comparable = (block != GAP) & self.has_consensus[column_mask]  # residue present, consensus exists
        matches = comparable & (block == consensus)
        n_comparable = comparable.sum(axis=1)
        return np.divide(matches.sum(axis=1), n_comparable,
                         out=np.zeros(len(rows)), where=n_comparable > 0)

    def member_fits(self) -> Dict[str, float]:
        """{sequence_id: fit} for every sequence in the alignment."""
        all_columns = np.ones(self.matrix.shape[1], dtype=bool)
        fits = self._fits(range(len(self.ids)), all_columns)
        return dict(zip(self.ids, fits.tolist()))

    def fit_of(self, seq_id: str) -> float:
        """Fit of a single sequence."""
        all_columns = np.ones(self.matrix.shape[1], dtype=bool)
        return float(self._fits([self.index[seq_id]], all_columns)[0])

    # ---- core columns ---------------------------------------------------------------

    def core_columns(self, rows: Sequence[int], gap_tolerance: float = CORE_GAP_TOLERANCE) -> np.ndarray:
        """Boolean mask of the columns in which fewer than `gap_tolerance` of the given rows have a gap.

        These are the columns that most sequences of the group actually occupy. Looking only at
        them keeps an insertion that is present in just one or two sequences from distorting the
        comparison, and makes the comparison robust to groups that only share one domain.
        """
        return (self.matrix[list(rows)] == GAP).mean(axis=0) < gap_tolerance

    # ---- how much of the family does a sequence span? ---------------------------

    def core_coverage(self, seq_id: str, reference_ids: Iterable[str],
                      gap_tolerance: float = CORE_GAP_TOLERANCE) -> float:
        """Fraction of the references' core columns in which `seq_id` has a residue.

        A full-length sequence scores ~1.0; a fragment covering 60% of the family scores ~0.6.
        An insertion in only a few references does not make everybody look "short" (which
        dividing by the total alignment length would do).
        """
        rows = [self.index[r] for r in dict.fromkeys(reference_ids) if r in self.index and r != seq_id]
        if not rows:
            return 0.0
        core = self.core_columns(rows, gap_tolerance)
        if not core.any():
            return 0.0
        return float((self.matrix[self.index[seq_id]][core] != GAP).mean())

    # ---- fit of two groups over their shared ("core") columns ------------------

    def core_group_fits(self, group1: Iterable[str], group2: Iterable[str],
                        gap_tolerance: float = CORE_GAP_TOLERANCE) -> Tuple[float, float, int]:
        """Compare two groups of sequences on the columns that are core columns of BOTH groups.

        Returns (mean fit of group1, mean fit of group2, number of core columns),
        where fits are measured against the consensus of the whole alignment.
        """
        rows1 = [self.index[s] for s in dict.fromkeys(group1) if s in self.index]
        rows2 = [self.index[s] for s in dict.fromkeys(group2) if s in self.index]
        if not rows1 or not rows2:
            return 0.0, 0.0, 0

        core = self.core_columns(rows1, gap_tolerance) & self.core_columns(rows2, gap_tolerance)
        if not core.any():
            return 0.0, 0.0, 0

        return (float(self._fits(rows1, core).mean()),
                float(self._fits(rows2, core).mean()),
                int(core.sum()))


# =============================================================================
# Does a sequence belong to a cluster?
# =============================================================================

def normalized_fit(query_id: str, seq_db: Dict[str, str],
                   representative_ids: Sequence[str], cluster_size: int) -> Optional[float]:
    """Score how well a sequence fits a cluster (higher = better; roughly 0 to 1).

    1. Align the query together with some representative cluster members.
    2. `raw_fit`      = fraction of the query's residues that match the column consensus.
    3. `coverage`     = fraction of the cluster's core alignment columns that the query
                        spans (see `Alignment.core_coverage`). Penalises fragments.
    4. `length_ratio` = how similar the query's length is to the representatives' mean
                        length (1.0 = identical).
    5. `size_factor`  = ln(cluster_size + 1) / ln(50), capped at 1. This deliberately
                        penalises very small clusters, whose consensus is not reliable.

        normalized_fit = raw_fit * (0.5 + 0.5 * coverage)
                                 * (0.7 + 0.3 * length_ratio) * size_factor

    The query itself is never used as one of its own representatives.
    Returns None if the alignment could not be built.
    """
    query_seq = seq_db[query_id]
    reps = [r for r in representative_ids if r != query_id and seq_db.get(r)]
    if not reps or not query_seq:
        return None

    with TemporaryDirectory() as tmp:
        fasta, aligned = Path(tmp) / "fit.fa", Path(tmp) / "fit.aln.fa"
        write_fasta(fasta, [(r, seq_db[r]) for r in reps] + [(query_id, query_seq)])
        if not run_clustalo(fasta, aligned):
            return None
        alignment = Alignment.from_fasta(aligned)
    if alignment is None or query_id not in alignment.index:
        return None

    mean_rep_length = sum(len(seq_db[r]) for r in reps) / len(reps)
    length_ratio = min(len(query_seq), mean_rep_length) / max(len(query_seq), mean_rep_length)
    size_factor = min(1.0, float(np.log(cluster_size + 1) / np.log(SIZE_FACTOR_FULL_AT)))

    coverage = alignment.core_coverage(query_id, reps)

    return (alignment.fit_of(query_id)
            * ((1 - COVERAGE_WEIGHT) + COVERAGE_WEIGHT * coverage)
            * ((1 - LENGTH_RATIO_WEIGHT) + LENGTH_RATIO_WEIGHT * length_ratio)
            * size_factor)


def summarize_fits(fits: Dict[str, float], min_member_fit: float,
                   groups: Optional[Dict[str, Iterable[str]]] = None) -> Dict[str, float]:
    """Count the members whose fit is below `min_member_fit` ("bad" members).

    Returns the number and fraction of bad members overall and, for every named group of
    sequence IDs, the fraction of *that group's* members that are bad (key `bad_fraction_<name>`).
    """
    bad = {seq_id for seq_id, fit in fits.items() if fit < min_member_fit}
    summary: Dict[str, float] = {"n_bad": len(bad), "bad_fraction": len(bad) / len(fits)}
    for name, ids in (groups or {}).items():
        members = [m for m in dict.fromkeys(ids) if m in fits]
        summary[f"bad_fraction_{name}"] = sum(m in bad for m in members) / len(members) if members else 0.0
    return summary


def validate_cluster_alignment(member_ids: Sequence[str], seq_db: Dict[str, str],
                               min_member_fit: float, max_bad_fraction: float,
                               groups: Optional[Dict[str, Iterable[str]]] = None) -> Tuple[bool, Dict]:
    """Align a complete cluster and check that (almost) all members fit it.

    A member is "bad" if its fit is below `min_member_fit`. The cluster is accepted
    if the fraction of bad members is at most `max_bad_fraction`.
    `groups` (optional) only adds per-group bad fractions to the report, e.g. the two clusters
    that were merged; it does not influence the decision.
    Returns (ok, info) where info holds the metrics and a `reason` ("ok" or why not).
    """
    with TemporaryDirectory() as tmp:
        fasta, aligned = Path(tmp) / "cluster.fa", Path(tmp) / "cluster.aln.fa"
        write_fasta(fasta, [(m, seq_db[m]) for m in member_ids])
        alignment = run_clustalo(fasta, aligned) and Alignment.from_fasta(aligned)
    if not alignment:
        return False, {"reason": "clustalo_failed"}

    info = {"mean_consensus": alignment.mean_consensus,
            **summarize_fits(alignment.member_fits(), min_member_fit, groups)}
    info["reason"] = "too_many_bad_members" if info["bad_fraction"] > max_bad_fraction else "ok"
    return info["reason"] == "ok", info
