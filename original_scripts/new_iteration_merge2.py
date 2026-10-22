#!/usr/bin/env python3
"""
COGonly_merge3.py (patched logging)

Same merge behavior as your original script but logs full metrics and
a human-readable explanation for every attempted merge.

Outputs:
 - addcog_only_fastas2/ (merged FASTAs same as before)
 - addcog_only_logs/merge_validation.csv  (full diagnostics per attempt)
 - addcog_only_logs/merge_history.json    (detailed objects for post-analysis)
 - addcog_only_logs/sequence_verification_report.txt (unchanged)
"""
import os
import sys
import json
import glob
import shutil
import subprocess
from collections import defaultdict
from tempfile import TemporaryDirectory
from datetime import datetime

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO

# ---------------------------
# CONFIGURATION (kept as in original)
# ---------------------------

seed_cluster_folder = "COGs"
csv_folder = "hmmsearch_combined"
combined_fasta = "combined.fasta"
unidentified_fasta = "unidentified.fasta"

output_fasta_folder = "addcog_only_fastas2"
log_folder = "addcog_only_logs"

os.makedirs(output_fasta_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

merge_validation_log = os.path.join(log_folder, "merge_validation.csv")
merge_history_file = os.path.join(log_folder, "merge_history.json")
verification_file = os.path.join(log_folder, "sequence_verification_report.txt")

expand_tol = 0.35
merge_thresh = 0.12
min_shared_seqs = 1

rep_sample_size = 50
pre_mean_consensus_min = 0.28
pre_min_cluster_fit = 0.24
pre_max_fit_diff = 0.55

post_bad_member_frac = 0.22
post_member_fit_thresh = 0.22

composition_shift_limit = 0.65

clustalo_bin = "clustalo"

# ---------------------------
# UTILITIES
# ---------------------------

def dbg(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def run_clustalo(in_fasta, out_aln):
    try:
        subprocess.run([clustalo_bin, "-i", in_fasta, "-o", out_aln, "--force", "--outfmt=fasta"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False
   
def sample_representatives(df, n=rep_sample_size):
    if 'Score' in df.columns:
        df_sorted = df.sort_values('Score', ascending=False)
    else:
        df_sorted = df.sample(frac=1, random_state=1)
    subjects = df_sorted['Subject'].tolist()
    L = len(subjects)
    if L <= n:
        return subjects
    k_top = max(1, int(n * 0.85))
    k_spread = n - k_top
    top = subjects[:k_top]
    step = max(1, L // k_spread)
    spread = subjects[::step][:k_spread]
    reps = list(dict.fromkeys(top + spread))[:n]
    return reps 
       
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

def compute_cluster_fit(new_id, cluster_ids, seq_db_local, new_db_local):
    # FIX: in this script the values of seq_db are dicts ({'sequence': ..., ...}). The dict itself used to be
    # written into the FASTA file as the sequence, and len() of the dict was used as the sequence length.
    def _seq(db, sid):
        entry = db.get(sid)
        return entry['sequence'] if isinstance(entry, dict) else entry

    REP_N_LOCAL = min(40, len(cluster_ids))
    reps = sample_representatives(cluster_ids, n=REP_N_LOCAL)
    if not reps:
        return None
    
    with TemporaryDirectory() as tmpd:
        fa = os.path.join(tmpd, "temp.fa")
        aln = fa + ".aln.fa"
        
        with open(fa, "w") as out:
            for sid in reps:
                seq = _seq(seq_db_local, sid)
                if seq:
                    out.write(f">{sid}\n{seq}\n")
            out.write(f">{new_id}\n{_seq(new_db_local, new_id)}\n")
        
        ok = run_clustalo(fa, aln)
        if not ok:
            return None
        
        mean_c, fits = compute_alignment_consensus_and_fits(aln)

        # --- FIX 1: correct variable ---
        raw_fit = fits.get(new_id, None)
        if raw_fit is None:
            return None

        from Bio import AlignIO
        alignment = AlignIO.read(aln, "fasta")

        aligned_query = None
        for record in alignment:
            if record.id == new_id:
                aligned_query = str(record.seq)
                break

        if aligned_query is None:
            return None

        # --- coverage ---
        # FIX: this used to be (non-gap characters / sequence length), which is always 1.0, so the
        # coverage term never did anything. Now: the fraction of the core columns (columns where fewer
        # than 60% of the reference sequences have a gap) in which the query has a residue.
        query_len = len(_seq(new_db_local, new_id))
        if query_len == 0:
            return None
        ref_rows = [str(r.seq) for r in alignment if r.id != new_id]
        core_cols = [i for i in range(len(aligned_query))
                     if ref_rows and sum(s[i] == '-' for s in ref_rows) / len(ref_rows) < 0.6]
        coverage = (sum(aligned_query[i] != '-' for i in core_cols) / len(core_cols)) if core_cols else 0.0

        # --- cluster average length ---
        lengths = []
        for sid in reps:
            seq = _seq(seq_db_local, sid)
            if seq:
                lengths.append(len(seq))
        if not lengths:
            return None

        cluster_avg_len = sum(lengths) / len(lengths)

        # --- length ratio ---
        length_ratio = min(query_len, cluster_avg_len) / max(query_len, cluster_avg_len)
        # --- size factor ---
        size_factor = np.log(len(cluster_ids) + 1) / np.log(50)   # normalize ~1 around 50 seqs
        size_factor = min(size_factor, 1.0)

        # --- FINAL NORMALIZED FIT ---
        normalized_fit = raw_fit * (0.5 + 0.5 * coverage) * (0.7 + 0.3 * length_ratio) * size_factor

        return normalized_fit
        
def alignment_expand_filter(candidate_ids, cluster_df,
                            min_fit=0.35,
                            consensus_drop_tol=0.65,
                            rep_n=20):

    reps = sample_representatives(cluster_df, rep_n)
    if not reps:
        return []

    accepted = []

    with TemporaryDirectory() as tmpd:
        base_fa = os.path.join(tmpd, "base.fa")
        aln = base_fa + ".aln.fa"

        with open(base_fa, "w") as out:
            for sid in reps:
                out.write(f">{sid}\n{seq_db[sid]['sequence']}\n")

        # seed-only alignment
        run_clustalo(base_fa, aln)
        _, seed_fits = compute_alignment_consensus_and_fits(aln)
        seed_consensus = np.mean(list(seed_fits.values()))

        for cand in candidate_ids:
            test_fa = os.path.join(tmpd, "test.fa")
            shutil.copy(base_fa, test_fa)

            with open(test_fa, "a") as out:
                out.write(f">{cand}\n{seq_db[cand]['sequence']}\n")

            if not run_clustalo(test_fa, aln):
                continue

            mean_consensus, fits = compute_alignment_consensus_and_fits(aln)

            # HARD FILTERS
            fit = compute_cluster_fit(cand, cluster_df, seq_db, seq_db)

            if fit is None or fit < min_fit:
                continue
            if mean_consensus < seed_consensus * consensus_drop_tol:
                continue

            accepted.append(cand)

    return accepted        
        
def write_temp_fasta_from_ids(seq_db_map, id_list, outpath):
    missing = []
    with open(outpath, "w") as out:
        for sid in id_list:
            seq = seq_db_map.get(sid, {}).get('sequence')
            if seq:
                out.write(f">{sid}\n{seq}\n")
            else:
                missing.append(sid)
    if missing:
        dbg(f"WARNING: {len(missing)} sequences missing in write_temp_fasta_from_ids: {missing[:5]}")
    return len(missing) == 0


def compute_group_core_fits(aln_path, group1_ids, group2_ids, gap_tol=0.7):
    """
    Domain-aware fit:
    Only score columns where BOTH groups have enough residues.
    Columns dominated by gaps in either group are ignored.
    """
    try:
        aln = AlignIO.read(aln_path, "fasta")
    except Exception:
        return 0.0, 0.0, 0

    ids = [rec.id for rec in aln]
    seqs = {rec.id: str(rec.seq) for rec in aln}
    L = aln.get_alignment_length()

    core_cols = []

    for i in range(L):
        g1 = [seqs[s][i] for s in group1_ids if s in seqs]
        g2 = [seqs[s][i] for s in group2_ids if s in seqs]
        if not g1 or not g2:
            continue

        g1_gap = sum(c == '-' for c in g1) / len(g1)
        g2_gap = sum(c == '-' for c in g2) / len(g2)

        if g1_gap < gap_tol and g2_gap < gap_tol:
            core_cols.append(i)

    if not core_cols:
        return 0.0, 0.0, 0

    def fit_for(ids_subset):
        fits = []
        for sid in ids_subset:
            if sid not in seqs:
                continue
            s = seqs[sid]
            matches = 0
            comps = 0
            for i in core_cols:
                c = s[i]
                if c == '-':
                    continue
                comps += 1
                # simple column consensus
                col = [seqs[x][i] for x in ids if seqs[x][i] != '-']
                if not col:
                    continue
                vals, counts = np.unique(col, return_counts=True)
                if c == vals[counts.argmax()]:
                    matches += 1
            fits.append(matches / comps if comps else 0.0)
        return float(np.mean(fits)) if fits else 0.0

    return fit_for(group1_ids), fit_for(group2_ids), len(core_cols)
    
# ---------------------------
# LOAD combined.fasta
# ---------------------------
dbg("Loading combined.fasta sequences (for merge stage)...")
seq_db = {}
if os.path.exists(combined_fasta):
    for rec in SeqIO.parse(combined_fasta, "fasta"):
        seq_db[rec.id] = {
            'sequence': str(rec.seq),
            'description': rec.description,
            'source_file': os.path.basename(combined_fasta)
        }
else:
    dbg(f"ERROR: {combined_fasta} not found.", file=sys.stderr)
    sys.exit(1)

original_addcog_sequences = set(seq_db.keys())
dbg(f"Loaded {len(seq_db)} combined (addCOG) sequences for merging.")

# ---------------------------
# LOAD COGs
# ---------------------------
dbg("Loading original cluster seed memberships...")
cluster_seeds = {}

for fa in glob.glob(os.path.join(seed_cluster_folder, "*.fasta")):
    cid = os.path.splitext(os.path.basename(fa))[0]
    ids = set(rec.id for rec in SeqIO.parse(fa, "fasta"))
    cluster_seeds[cid] = ids

dbg(f"Loaded seed memberships for {len(cluster_seeds)} clusters")


# ---------------------------
# LOAD HMM CSV TABLES
# ---------------------------
dbg("Loading HMM CSV tables from", csv_folder)
if not os.path.isdir(csv_folder):
    dbg("ERROR: csv_folder not found:", csv_folder, file=sys.stderr)
    sys.exit(1)

def normalize_hmm_df(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = list(df.columns)
    subject_col = None
    score_col = None
    query_col = None
    source_col = None
    for c in cols:
        lc = c.lower()
        if lc in ("subject", "target_seq", "target", "target_sequence", "subject_id"):
            subject_col = c
        if lc in ("score", "bitscore", "bits"):
            score_col = c
        if lc in ("query", "qname", "query_profile", "query_name"):
            query_col = c
        if lc in ("source file", "source_file", "source", "file", "target_file"):
            source_col = c
    if subject_col is None and len(cols) >= 2:
        subject_col = cols[1]
    if score_col is None and len(cols) >= 5:
        score_col = cols[4]
    if query_col is None:
        query_col = None
    nd = pd.DataFrame()
    nd['Subject'] = df[subject_col].astype(str) if subject_col in df else df.iloc[:,1].astype(str)
    if score_col and score_col in df:
        nd['Score'] = pd.to_numeric(df[score_col], errors='coerce')
    else:
        try:
            nd['Score'] = pd.to_numeric(df.iloc[:,2], errors='coerce')
        except Exception:
            nd['Score'] = np.nan
    if query_col and query_col in df:
        nd['Query'] = df[query_col].astype(str)
    else:
        nd['Query'] = os.path.splitext(os.path.basename(path))[0]
    if source_col and source_col in df:
        nd['Source file'] = df[source_col].astype(str)
    else:
        nd['Source file'] = np.nan
    return nd

cluster_tables = {}
for csvf in sorted(glob.glob(os.path.join(csv_folder, "*.csv"))):
    try:
        nd = normalize_hmm_df(csvf)
        if nd.empty:
            continue
        cid = nd['Query'].iloc[0] if 'Query' in nd.columns else os.path.splitext(os.path.basename(csvf))[0]
        cluster_tables[cid] = nd
    except Exception as e:
        dbg("Warning: failed to load", csvf, e)

dbg(f"Loaded {len(cluster_tables)} HMM tables (may contain unidentified hits).")

# ---------------------------
# PHASE 1: Build expanded clusters (addCOG-only)
# ---------------------------
dbg("Phase 1: building addCOG-only expanded clusters (no unidentified sequences included).")
expanded = {}
merge_history = {"timestamp": datetime.now().isoformat(), "expansion": {}, "merge_ops": []}

for cid, table in cluster_tables.items():

    # --- seed (original) cluster members ---
    seed_ids = cluster_seeds.get(cid, set())
    if not seed_ids:
        continue

    # restrict to addCOG universe
    current_universe = set().union(*cluster_seeds.values())
    tbl_add = table[table['Subject'].isin(current_universe)]

    if tbl_add.empty:
        continue

    # seed hits only (used to define score window)
    seed_df = tbl_add[tbl_add['Subject'].isin(seed_ids)]

    if seed_df.empty:
        # preserve cluster exactly as it is
        expanded_ids = set(seed_ids)
        exp_df = pd.DataFrame({
            "Subject": list(expanded_ids),
            "Score": [np.nan] * len(expanded_ids)
        })
        expanded[cid] = exp_df
        dbg(f"[EXPAND-PRESERVE] {cid}: no HMM seed hits — cluster kept unchanged")
        continue


    scores = seed_df['Score'].dropna()
    if scores.empty:
        continue

    mean_score = float(scores.median())
    lower, upper = mean_score * (1 - expand_tol), mean_score * (1 + expand_tol)

    # --- HMM prefilter: NON-SEED addCOGs only ---
    hmm_candidates_df = tbl_add[
        (tbl_add['Score'] >= lower) &
        (tbl_add['Score'] <= upper) &
        (~tbl_add['Subject'].isin(seed_ids))
    ]

    candidate_ids = hmm_candidates_df['Subject'].unique().tolist()

    # --- alignment validation ---
    accepted_ids = alignment_expand_filter(
        candidate_ids,
        seed_df,
        min_fit=pre_min_cluster_fit,
        rep_n=rep_sample_size
    )

    # --- final expanded cluster ---
    prev_ids = set(cluster_seeds.get(cid, set()))
    expanded_ids = prev_ids | set(accepted_ids)

    # Start from HMM hits
    exp_df = tbl_add[tbl_add['Subject'].isin(expanded_ids)].drop_duplicates(subset=['Subject'])

    # Ensure ALL seed members are present even without HMM hits
    present = set(exp_df['Subject'])
    missing_seeds = expanded_ids - present
    if missing_seeds:
        dbg(f"  [WARN] {cid}: {len(missing_seeds)} seed members had no HMM hit — force-including them")
        rescue_rows = [{"Subject": sid, "Score": np.nan} for sid in missing_seeds]
        exp_df = pd.concat([exp_df, pd.DataFrame(rescue_rows)], ignore_index=True)

    expanded[cid] = exp_df.reset_index(drop=True)

    merge_history['expansion'][cid] = {
        "seed": int(len(seed_ids)),
        "hmm_candidates": int(len(candidate_ids)),
        "alignment_accepted": int(len(accepted_ids)),
        "expanded": int(len(expanded_ids))
    }

    dbg(f"[EXPAND] {cid}: seed={len(seed_ids)} "
        f"hmm={len(candidate_ids)} align_ok={len(accepted_ids)}")


dbg(f"Expanded (addCOG-only) clusters: {len(expanded)}")

# After the expansion loop, add a safety net:
all_expanded_ids = set()
for df in expanded.values():
    all_expanded_ids.update(df['Subject'].tolist())

all_seed_ids = set()
for cid, ids in cluster_seeds.items():
    all_seed_ids.update(ids)

orphaned = all_seed_ids - all_expanded_ids
if orphaned:
    dbg(f"WARNING: {len(orphaned)} seed sequences not captured by expansion. Preserving them.")
    for cid, ids in cluster_seeds.items():
        missing_from_cid = ids & orphaned
        if missing_from_cid:
            # Create a minimal DataFrame for this cluster if it doesn't exist
            if cid not in expanded:
                rows = [{"Subject": sid, "Score": np.nan} for sid in missing_from_cid]
                expanded[cid] = pd.DataFrame(rows)
            else:
                existing = set(expanded[cid]['Subject'])
                to_add = missing_from_cid - existing
                if to_add:
                    rows = [{"Subject": sid, "Score": np.nan} for sid in to_add]
                    expanded[cid] = pd.concat([expanded[cid], pd.DataFrame(rows)], ignore_index=True)
            orphaned -= missing_from_cid
            
# -----------------------------------------------------------
# EXPORT PHASE-1 EXPANDED CLUSTERS FOR SEQUENCE LINEAGE TRACING
# -----------------------------------------------------------
expanded_out = {cid: df['Subject'].tolist() for cid, df in expanded.items()}
with open(os.path.join(log_folder, "expanded_clusters.json"), "w") as ef:
    json.dump(expanded_out, ef, indent=2)
dbg("Saved expanded_clusters.json for full sequence lineage tracing.")

# ---------------------------
# Candidate merge selection (conservative)
# ---------------------------
def should_merge_basic(df1, df2, merge_threshold=merge_thresh, min_shared=min_shared_seqs):
    s1, s2 = set(df1['Subject']), set(df2['Subject'])
    shared = s1 & s2
    shared_n = len(shared)

    # ---------- overlap metrics ----------
    if shared_n > 0:
        overlap = shared_n / min(len(s1), len(s2))
        # FIX: pair the two profiles' scores by sequence ID. Before, the two score lists were paired by row
        # position, i.e. scores of DIFFERENT sequences were correlated (both tables are sorted by their own
        # scores), so the correlation came out high whatever the real relationship was.
        s1_scores = df1[df1['Subject'].isin(shared)].set_index('Subject')['Score']
        s2_scores = df2[df2['Subject'].isin(shared)].set_index('Subject')['Score'].reindex(s1_scores.index)

        try:
            corr = np.corrcoef(s1_scores, s2_scores)[0,1] if shared_n > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        # FIX: bit scores of different profiles are on different scales, so the 0.35 / 0.55 thresholds below
        # (meant for relative scores) were always satisfied. Divide each profile's scores by the best score
        # that profile gives to any member of its own cluster (1.0 = as good as its best member).
        best1, best2 = df1['Score'].max(), df2['Score'].max()
        norm_s1 = s1_scores / best1 if best1 > 0 else s1_scores * 0.0
        norm_s2 = s2_scores / best2 if best2 > 0 else s2_scores * 0.0
        mean_cross = float((norm_s1.mean() + norm_s2.mean()) / 2.0)
        max_cross  = float(np.fmax(norm_s1.max(), norm_s2.max()))
    else:
        overlap = corr = mean_cross = max_cross = 0.0

    combined_score = overlap * 0.7 + max(corr, 0.0) * 0.3

    # ---------- channel A: overlap path ----------
    ok_overlap = (
        shared_n >= min_shared
        and overlap >= merge_threshold
        and combined_score >= merge_threshold
    )

    # ---------- channel B: score-signal path ----------
    ok_signal = (
        shared_n >= 1
        and corr >= 0.75
        and mean_cross >= 0.35
        and max_cross >= 0.55
    )

    should = ok_overlap or ok_signal

    meta = {
        "shared": shared_n,
        "overlap": float(overlap),
        "corr": float(corr),
        "combined": float(combined_score),
        "mean_cross": mean_cross,
        "max_cross": max_cross,
        "path": "signal" if ok_signal and not ok_overlap else "overlap"
    }

    return bool(should), float(combined_score), meta



expanded_items = sorted(expanded.items(), key=lambda x: len(x[1]), reverse=True)
candidate_merges = []
for i, (cid1, df1) in enumerate(expanded_items):
    for cid2, df2 in expanded_items[i+1:]:
        ok, score, meta = should_merge_basic(df1, df2)
        if ok:
            candidate_merges.append((cid1, cid2, score, meta))
dbg(f"Initial candidate merges (addCOG-only): {len(candidate_merges)}")

# ---------------------------
# Representative sampling (kept)
# ---------------------------

def pre_merge_reciprocal_test(cid1, cid2, current_clusters_local):
    df1 = current_clusters_local[cid1]
    df2 = current_clusters_local[cid2]
    reps1 = sample_representatives(df1, rep_sample_size)
    reps2 = sample_representatives(df2, rep_sample_size)
    with TemporaryDirectory() as tmpd:
        fpath = os.path.join(tmpd, f"{cid1}__{cid2}__reps.fa")
        # FIX: a sequence shared by both clusters is in reps1 and reps2; write it only once
        write_temp_fasta_from_ids(seq_db, list(dict.fromkeys(reps1 + reps2)), fpath)
        aln = fpath + ".aln.fa"
        ok = run_clustalo(fpath, aln)
        if not ok:
            return False, {"reason":"clustalo_fail"}
        mean_consensus, fits = compute_alignment_consensus_and_fits(aln)

        core_fit1, core_fit2, core_cols = compute_group_core_fits(
            aln,
            reps1,
            reps2,
            gap_tol=0.6
        )
        fit1 = core_fit1
        fit2 = core_fit2
        info = {"mean_consensus": float(mean_consensus), "fit1": float(fit1), "fit2": float(fit2),
                "reps1": reps1[:5], "reps2": reps2[:5], "n_reps1": len(reps1), "n_reps2": len(reps2)}
        info["core_fit1"] = float(core_fit1)
        info["core_fit2"] = float(core_fit2)
        info["core_columns"] = int(core_cols)
        info["size1"] = len(df1)
        info["size2"] = len(df2)

        if mean_consensus < pre_mean_consensus_min:
            info["reason"]="low_consensus"; return False, info
        # DOMAIN-AWARE: require strong fit in at least one cluster core
        if max(fit1, fit2) < pre_min_cluster_fit:
            info["reason"]="no_core_fit"; return False, info
        if abs(fit1-fit2) > pre_max_fit_diff:
            info["reason"]="fit_diff_too_large"; return False, info
        return True, info

def post_merge_validate(subject_list):
    with TemporaryDirectory() as tmpd:
        tmpfa = os.path.join(tmpd, "merged.fa")
        write_temp_fasta_from_ids(seq_db, subject_list, tmpfa)
        tmpaln = tmpfa + ".aln.fa"
        ok = run_clustalo(tmpfa, tmpaln)
        if not ok:
            return False, {"reason":"post_clustalo_fail"}
        mean_consensus, fits = compute_alignment_consensus_and_fits(tmpaln)
        bad = [s for s, fit in fits.items() if fit < post_member_fit_thresh]
        bad_frac = len(bad) / len(fits) if fits else 0.0
        info = {"mean_consensus": float(mean_consensus), "bad_fraction": float(bad_frac), "n_bad": len(bad),
                "example_bad": bad[:10]}
        if bad_frac > post_bad_member_frac:
            info["reason"]="too_many_bad_members"; info["bad_members"] = bad[:10]
            return False, info
        return True, info

# ---------------------------
# MERGE LOOP (Phase 1) with enriched logging
# ---------------------------
current_clusters = {cid: df.copy() for cid, df in expanded.items()}
validation_rows = []
merge_ops = []
audit_rows = []


# candidate_merges elements: (cid1, cid2, score, meta)
candidate_merges = sorted(candidate_merges, key=lambda x: x[2], reverse=True)

for cid1, cid2, init_score, meta in candidate_merges:
    # skip if cluster removed in previous merges
    if cid1 not in current_clusters or cid2 not in current_clusters:
        continue

    # prepare the base vrow that will be filled for either accept/reject
    vrow = {
        "cid1": cid1,
        "cid2": cid2,
        "initial_score": float(init_score),
        "candidate_meta": meta
    }
    audit_base = {
        "cid1": cid1,
        "cid2": cid2,
        "size1": len(current_clusters[cid1]),
        "size2": len(current_clusters[cid2]),
        "initial_score": float(init_score),
        "shared": meta.get("shared"),
        "overlap": meta.get("overlap"),
        "corr": meta.get("corr"),
    }

    # composition breakdown (only combined sequences)
    def composition_breakdown_local(df):
        counts = defaultdict(int)
        for s in df['Subject']:
            counts[combined_fasta] += 1
        total = sum(counts.values()) or 1
        frac = {k: v / total for k, v in counts.items()}
        return counts, frac

    counts_before1, frac_before1 = composition_breakdown_local(current_clusters[cid1])
    counts_before2, frac_before2 = composition_breakdown_local(current_clusters[cid2])
    merged_preview = pd.concat([current_clusters[cid1], current_clusters[cid2]], ignore_index=True).drop_duplicates(subset=['Subject'])
    counts_merged, frac_merged = composition_breakdown_local(merged_preview)

    # composition shift check
    keys = set(list(frac_before1.keys()) + list(frac_before2.keys()) + list(frac_merged.keys()))
    shifts = {k: abs(frac_merged.get(k, 0.0) - max(frac_before1.get(k, 0.0), frac_before2.get(k, 0.0))) for k in keys}
    max_shift = max(shifts.values()) if shifts else 0.0

    # record candidate-level stats in vrow
    vrow["composition_before_1"] = {"counts": counts_before1, "frac": frac_before1}
    vrow["composition_before_2"] = {"counts": counts_before2, "frac": frac_before2}
    vrow["composition_merged"] = {"counts": counts_merged, "frac": frac_merged}
    vrow["composition_max_shift"] = float(max_shift)

    reason_details = []

    if max_shift > composition_shift_limit:
        reason_details.append(f"composition_shift(max_shift={max_shift:.3f})")
        vrow.update({"pre_info": None, "post_info": None, "decision": "reject", "reason": ";".join(reason_details)})
        validation_rows.append(vrow)
        continue

    # PRE-merge reciprocal alignment check (on combined sequences only)
    ok_pre, pre_info = pre_merge_reciprocal_test(cid1, cid2, current_clusters)
    audit_entry = dict(audit_base)
    audit_entry["fallback"] = meta.get("fallback", False)
    audit_entry.update({
        "pre_ok": ok_pre,
        "pre_reason": pre_info.get("reason") if pre_info else None,
        "mean_consensus": pre_info.get("mean_consensus") if pre_info else None,
        "fit1": pre_info.get("fit1") if pre_info else None,
        "fit2": pre_info.get("fit2") if pre_info else None,
        "core_fit1": pre_info.get("core_fit1") if pre_info else None,
        "core_fit2": pre_info.get("core_fit2") if pre_info else None,
        "core_cols": pre_info.get("core_columns") if pre_info else None,
    })

    vrow["pre_info"] = pre_info
    if not ok_pre:
        audit_entry["post_ok"] = False
        audit_entry["post_reason"] = pre_info.get("reason")
        audit_rows.append(audit_entry)
        reason_details.append(f"pre_merge_failed: {pre_info.get('reason', 'unknown')}")
        vrow.update({"post_info": None, "decision": "reject", "reason": ";".join(reason_details)})
        validation_rows.append(vrow)
        continue
    else:
        reason_details.append(f"pre_ok(mean_consensus={pre_info.get('mean_consensus'):.3f},fit1={pre_info.get('fit1'):.3f},fit2={pre_info.get('fit2'):.3f})")

    # Tentative merged subjects (combined sequences only)
    merged_subjects = list(merged_preview['Subject'].unique())

    # POST-merge validation on combined sequences only
    ok_post, post_info = post_merge_validate(merged_subjects)
    vrow["post_info"] = post_info
    if not ok_post:
        audit_entry["post_ok"] = False
        audit_entry["post_reason"] = post_info.get("reason")
        audit_rows.append(audit_entry)
        reason_details.append(f"post_merge_failed: {post_info.get('reason', 'unknown')}")
        vrow.update({"decision": "reject", "reason": ";".join(reason_details)})
        validation_rows.append(vrow)
        continue
    else:
        reason_details.append(f"post_ok(mean_consensus={post_info.get('mean_consensus'):.3f},bad_frac={post_info.get('bad_fraction'):.3f})")

    # Candidate-level checks (overlap/corr etc) - include them in reason_details
    shared = meta.get("shared", 0)
    overlap = meta.get("overlap", 0.0)
    corr = meta.get("corr", 0.0)
    combined = meta.get("combined", 0.0)
    reason_details.append(f"candidate(shared={int(shared)},overlap={overlap:.3f},corr={corr:.3f},combined={combined:.3f})")

    # Accept merge: update clusters
    merged_df_final = merged_preview.sort_values('Score', ascending=False).drop_duplicates(subset=['Subject']).reset_index(drop=True)
    current_clusters[cid1] = merged_df_final
    if cid2 in current_clusters:
        del current_clusters[cid2]

    merge_ops.append({"clusters": [cid1, cid2], "initial_score": float(init_score), "candidate_meta": meta, "pre_info": pre_info, "post_info": post_info})
    audit_entry["post_ok"] = True
    audit_entry["post_reason"] = "merged"
    audit_rows.append(audit_entry)

    # final vrow fields for accepted merge
    vrow.update({
        "decision": "accept",
        "reason": ";".join(reason_details),
        "final_count": int(len(merged_df_final))
    })
    validation_rows.append(vrow)

    dbg(f"Merged {cid1} + {cid2} -> {len(merged_df_final)} combined seqs")

dbg(f"Phase 1 merging complete. Remaining clusters: {len(current_clusters)}")

# ---------------------------
# Build final_clusters dict (addCOG-only merged clusters)
# ---------------------------
final_clusters = {cid: df.copy() for cid, df in current_clusters.items()}

# ---------------------------
# PHASE 2: Assign unidentified sequences (unchanged logic)
# ---------------------------
# ---------------------------
# PHASE 2: DISABLED (second iteration)
# ---------------------------
### CHANGED FOR SECOND ITERATION ###
dbg("Phase 2 skipped: no unidentified sequences will be added.")


# ---------------------------
# Write outputs & verification
# ---------------------------

before_ids = set().union(*cluster_seeds.values())
after_ids  = set().union(*[set(df['Subject']) for df in final_clusters.values()])

lost = before_ids - after_ids
if lost:
    print("FATAL: sequences lost during merge stage:", len(lost))
    for x in list(lost)[:20]:
        print("  LOST:", x)
    raise RuntimeError("Merge stage lost sequences — aborting.")

dbg("Writing merged FASTAs and logs...")

written = 0
for cid, df in final_clusters.items():
    out_fa = os.path.join(output_fasta_folder, f"{cid}_merged.fasta")
    if 'Score' in df.columns:
        dfw = df.sort_values('Score', ascending=False).drop_duplicates(subset=['Subject'])
    else:
        dfw = df.drop_duplicates(subset=['Subject'])
    subjects = dfw['Subject'].tolist()
    with open(out_fa, "w") as fh:
        for sid in subjects:
            seq = seq_db.get(sid, {}).get("sequence")
            if seq:
                fh.write(f">{sid} {cid}\n{seq}\n")

    written += 1

all_final_subjects = set()
for df in final_clusters.values():
    all_final_subjects.update(df['Subject'].tolist())
missing_originals = original_addcog_sequences - all_final_subjects

merge_history["merge_ops"] = merge_ops
merge_history["verification"] = {
    "all_original_sequences_preserved": len(missing_originals) == 0,
    "missing_count": int(len(missing_originals)),
    "missing_examples": list(missing_originals)[:200]
}

# write validation csv (with JSON fields for pre/post and candidate_meta)
with open(merge_validation_log, "w", newline="") as csvf:
    import csv as _csv
    w = _csv.writer(csvf)
    w.writerow(["cid1","cid2","initial_score","candidate_meta","pre_info","post_info","decision","reason","final_count"])
    for r in validation_rows:
        w.writerow([r.get("cid1"), r.get("cid2"), r.get("initial_score"),
                    json.dumps(r.get("candidate_meta")), json.dumps(r.get("pre_info")),
                    json.dumps(r.get("post_info")), r.get("decision"), r.get("reason"),
                    r.get("final_count")])

with open(merge_history_file, "w") as mh:
    json.dump(merge_history, mh, indent=2)

with open(verification_file, "w") as vf:
    vf.write("Verification of preserved original addCOG sequences\n")
    vf.write("="*60 + "\n")
    vf.write(f"All preserved: {merge_history['verification']['all_original_sequences_preserved']}\n")
    vf.write(f"Missing count: {merge_history['verification']['missing_count']}\n")
    if merge_history['verification']['missing_count'] > 0:
        vf.write("\nMissing examples:\n")
        for s in merge_history['verification']['missing_examples'][:200]:
            vf.write(s + "\n")

dbg(f"Written {written} merged FASTA files to {output_fasta_folder}")
dbg(f"Validation log: {merge_validation_log}")
dbg(f"Merge history: {merge_history_file}")
dbg(f"Verification: {verification_file}")
dbg("✅ addCOG-only merge then expand finished.")

