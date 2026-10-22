#!/usr/bin/env python3
"""
dedup_fit_random_sample.py
Deduplication using normalized fit metric, with random sampling (deterministic seed).
For each duplicate, compute fit against each cluster using a random subset of
representatives (size = FIT_SAMPLE_SIZE). Assign to cluster with highest fit.
"""

import os
import sys
import subprocess
import random
import zlib
from collections import defaultdict
from tempfile import TemporaryDirectory
import numpy as np
from Bio import SeqIO, AlignIO

# ===== CONFIGURATION =====
cluster_dirs = ["addcog_only_fastas2"]
dedup_output_dirs = [d + "_dedup2" for d in cluster_dirs]
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

report_file = os.path.join(log_dir, "cluster_integrity_report.txt")
dedup_log = os.path.join(log_dir, "deduplication_report.txt")

clustalo_bin = "clustalo"
FIT_SAMPLE_SIZE = 200          # Max number of representatives used in alignment (if cluster larger)

# ===== UTILITIES =====
def run_clustalo(in_fasta, out_aln):
    try:
        subprocess.run([clustalo_bin, "-i", in_fasta, "-o", out_aln, "--force", "--outfmt=fasta"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def random_sample_representatives(seq_ids, sample_size, seed):
    """
    Return a list of up to sample_size sequence IDs.
    If len(seq_ids) <= sample_size, return all (no sampling).
    Otherwise, return a random sample (deterministic given seed).
    """
    if len(seq_ids) <= sample_size:
        return seq_ids
    rng = random.Random(seed)
    return rng.sample(seq_ids, sample_size)

def compute_alignment_consensus_and_fits(aln_path):
    try:
        aln = AlignIO.read(aln_path, "fasta")
    except Exception:
        return 0.0, {}
    L = aln.get_alignment_length()
    if L == 0:
        return 0.0, {}
    seq_ids = [rec.id for rec in aln]
    seqs = [str(rec.seq) for rec in aln]
    col_scores = []
    consensus = []
    for i in range(L):
        # FIX: only gaps are ignored. 'N' is asparagine in proteins, not an unknown base; it used to
        # be left out of the consensus, so Asn residues could never match it.
        col = [s[i] for s in seqs if s[i] != '-']
        if not col:
            consensus.append(None)
            col_scores.append(0.0)
        else:
            vals, counts = np.unique(col, return_counts=True)
            idx = counts.argmax()
            consensus.append(vals[idx])
            col_scores.append(counts[idx] / float(len(col)))
    mean_consensus = float(np.mean(col_scores)) if col_scores else 0.0
    fits = {}
    for sid, s in zip(seq_ids, seqs):
        matches = 0
        comps = 0
        for i, c in enumerate(s):
            cons = consensus[i]
            if c == '-' or cons is None:
                continue
            comps += 1
            if c == cons:
                matches += 1
        fits[sid] = (matches / comps) if comps > 0 else 0.0
    return mean_consensus, fits

def compute_cluster_fit(new_id, cluster_ids, seq_db, new_seq, cluster_name, duplicate_id):
    """
    Compute normalized fit of new_id against the cluster.
    Uses random sample of representatives (size = FIT_SAMPLE_SIZE) with seed = cluster_name + duplicate_id.
    """
    if not cluster_ids:
        return None
    # Create a deterministic seed for this cluster-duplicate pair
    seed_str = f"{cluster_name}_{duplicate_id}"
    # FIX: hash() of a string is different in every Python run, so the sample was not reproducible
    seed = zlib.crc32(seed_str.encode()) % (2**32)
    # FIX: the sequence is itself a member of every cluster it is duplicated in; it must not be aligned
    # against itself (that inflated its fit and put its ID twice into the alignment file)
    reps = random_sample_representatives([s for s in cluster_ids if s != new_id], FIT_SAMPLE_SIZE, seed)
    if not reps:
        return None

    with TemporaryDirectory() as tmpd:
        fa = os.path.join(tmpd, "temp.fa")
        aln = fa + ".aln.fa"
        with open(fa, "w") as out:
            for sid in reps:
                seq = seq_db.get(sid)
                if seq:
                    out.write(f">{sid}\n{seq}\n")
            out.write(f">{new_id}\n{new_seq}\n")

        if not run_clustalo(fa, aln):
            return None

        mean_c, fits = compute_alignment_consensus_and_fits(aln)
        raw_fit = fits.get(new_id, None)
        if raw_fit is None:
            return None

        # Coverage
        alignment = AlignIO.read(aln, "fasta")
        aligned_query = None
        for record in alignment:
            if record.id == new_id:
                aligned_query = str(record.seq)
                break
        if aligned_query is None:
            return None

        # FIX: this used to be (non-gap characters / sequence length), which is always 1.0, so the
        # coverage term never did anything. Now: the fraction of the core columns (columns where fewer
        # than 60% of the reference sequences have a gap) in which the query has a residue.
        query_len = len(new_seq)
        if query_len == 0:
            return None
        ref_rows = [str(r.seq) for r in alignment if r.id != new_id]
        core_cols = [i for i in range(len(aligned_query))
                     if ref_rows and sum(s[i] == '-' for s in ref_rows) / len(ref_rows) < 0.6]
        coverage = (sum(aligned_query[i] != '-' for i in core_cols) / len(core_cols)) if core_cols else 0.0

        # Cluster average length over the sampled representatives
        lengths = [len(seq_db.get(sid, "")) for sid in reps if seq_db.get(sid)]
        if not lengths:
            return None
        cluster_avg_len = sum(lengths) / len(lengths)

        length_ratio = min(query_len, cluster_avg_len) / max(query_len, cluster_avg_len)
        size_factor = np.log(len(cluster_ids) + 1) / np.log(50)
        size_factor = min(size_factor, 1.0)

        normalized_fit = raw_fit * (0.5 + 0.5 * coverage) * (0.7 + 0.3 * length_ratio) * size_factor
        return normalized_fit

# ===== LOAD CLUSTERS =====
print("Loading clusters...")
cluster_files = []
for d in cluster_dirs:
    if not os.path.exists(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".fasta"):
            cluster_files.append(os.path.join(d, f))

print(f"Found {len(cluster_files)} cluster FASTA files.")

cluster_contents = {}
cluster_id_lists = {}
seq_db = {}

for fpath in cluster_files:
    cname = os.path.basename(fpath)
    seqs = list(SeqIO.parse(fpath, "fasta"))
    cluster_contents[cname] = seqs
    ids = [rec.id for rec in seqs]
    cluster_id_lists[cname] = ids
    for rec in seqs:
        seq_db[rec.id] = str(rec.seq)

print(f"Loaded {len(cluster_contents)} clusters, {len(seq_db)} unique sequences.")

# ===== IDENTIFY DUPLICATES =====
cluster_map = defaultdict(list)
for cname, ids in cluster_id_lists.items():
    for sid in ids:
        cluster_map[sid].append(cname)

duplicates = {sid: clusters for sid, clusters in cluster_map.items() if len(clusters) > 1}
print(f"Found {len(duplicates)} duplicate sequence IDs.")

# ===== AUDIT =====
with open(report_file, "w") as out:
    out.write("Cluster integrity report\n")
    out.write("=" * 60 + "\n\n")
    out.write(f"Total clusters: {len(cluster_contents)}\n")
    out.write(f"Total unique sequences: {len(seq_db)}\n")
    out.write(f"Duplicate IDs: {len(duplicates)}\n\n")
    if duplicates:
        out.write("Duplicate sequence IDs and their clusters:\n")
        for sid, clist in sorted(duplicates.items()):
            out.write(f"{sid} : {', '.join(clist)}\n")
print(f"Audit written to {report_file}")

# ===== DEDUPLICATION =====
removed = []
kept = []

with open(dedup_log, "w") as log:
    log.write("Deduplication report (normalized fit, random sampling, sample size = {})\n".format(FIT_SAMPLE_SIZE))
    log.write("=" * 80 + "\n\n")

    for sid, clusters in sorted(duplicates.items()):
        seq_str = seq_db.get(sid)
        if seq_str is None:
            log.write(f"WARNING: sequence {sid} not found\n")
            continue

        cluster_fits = {}
        for cl in clusters:
            cluster_ids = cluster_id_lists[cl]
            fit = compute_cluster_fit(sid, cluster_ids, seq_db, seq_str, cl, sid)
            if fit is None:
                fit = 0.0
            cluster_fits[cl] = fit
            log.write(f"  {cl} -> normalized fit = {fit:.4f}\n")

        best_cluster = max(cluster_fits.items(), key=lambda x: (x[1], x[0]))[0]
        best_fit = cluster_fits[best_cluster]
        log.write(f"{sid}: best cluster {best_cluster} (fit {best_fit:.4f})\n")
        if len(clusters) > 1:
            others = [f"{c}({cluster_fits[c]:.4f})" for c in clusters if c != best_cluster]
            log.write(f"   other clusters: {', '.join(others)}\n")

        kept.append((sid, best_cluster))
        for c in clusters:
            if c != best_cluster:
                original_len = len(cluster_contents[c])
                cluster_contents[c] = [rec for rec in cluster_contents[c] if rec.id != sid]
                if len(cluster_contents[c]) < original_len:
                    removed.append((sid, c))
                cluster_id_lists[c] = [rec.id for rec in cluster_contents[c]]

    log.write(f"\nTotal duplicate placements removed: {len(removed)}\n")
    log.write(f"Total sequences kept: {len(kept)}\n")

print(f"Deduplication done. Removed {len(removed)} duplicate entries.")

# ===== VERIFY NO SEQUENCES LOST =====
all_input_ids = set(cluster_map.keys())
all_output_ids = set()
for seqs in cluster_contents.values():
    for rec in seqs:
        all_output_ids.add(rec.id)

lost = all_input_ids - all_output_ids
if lost:
    print(f"FATAL: {len(lost)} sequences lost during dedup!")
    for sid in sorted(lost)[:20]:
        print(f"  LOST: {sid}")
    sys.exit(1)
else:
    print(f"Verification OK: all {len(all_input_ids)} input sequences present in output.")

# ===== WRITE CLEANED CLUSTERS =====
print("Writing cleaned FASTAs...")
for out_base in dedup_output_dirs:
    out_dir = out_base
    os.makedirs(out_dir, exist_ok=True)
    for cname, seqs in cluster_contents.items():
        if not seqs:
            continue
        out_path = os.path.join(out_dir, cname)
        with open(out_path, "w") as out_f:
            SeqIO.write(seqs, out_f, "fasta")

print(f"Finished. Cleaned clusters saved to {', '.join(dedup_output_dirs)}")
print(f"Logs in {log_dir}")
