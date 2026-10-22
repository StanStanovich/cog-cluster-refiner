#!/usr/bin/env python3

"""

COGonly_phase2_flexible.py

Phase 2 that can add sequences from any of your three files:
- unidentified.fasta (suffix: _No_MCogs_noCogs)
- No_MCogs_with_Cogs.fasta (suffix: No_MCogs_with_Cogs)  
- With_MCogs_tagged.fasta (suffix: _MitoCog)

Usage: python phase2.py <file_to_add>

Examples:
  python phase2.py unidentified.fasta
  python phase2.py No_MCogs_with_Cogs.fasta
  python phase2.py With_MCogs_tagged.fasta
"""

import os
import sys
import glob
import subprocess
from collections import defaultdict
from tempfile import TemporaryDirectory
from datetime import datetime

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO

# ---------------------------
# CHECK COMMAND LINE ARGUMENT
# ---------------------------
if len(sys.argv) != 2:
    print("ERROR: Please specify which file to add sequences from")
    print("Usage: python phase2.py <filename>")
    print("Options: unidentified.fasta, No_MCogs_with_Cogs.fasta, With_MCogs_tagged.fasta")
    sys.exit(1)

input_file = sys.argv[1]
valid_files = ['unidentified.fasta', 'No_MCogs_with_Cogs.fasta', 'With_MCogs_tagged.fasta']

if input_file not in valid_files:
    print(f"ERROR: Invalid filename. Must be one of: {valid_files}")
    sys.exit(1)

# Map filename to suffix
suffix_map = {
    'unidentified.fasta': '_No_MCogs_no_Cogs',
    'No_MCogs_with_Cogs.fasta': 'No_MCogs_with_Cogs',
    'With_MCogs_tagged.fasta': '_MitoCog'
}
target_suffix = suffix_map[input_file]

# ---------------------------
# CONFIGURATION
# ---------------------------

csv_folder = "hmmsearch_combined"
combined_fasta = "combined.fasta"
input_fasta_folder = "COGs"
output_fasta_folder = "addcog_only_fastas2_dedup2"
log_folder = "addcog_only_logs"

# Create output folder with suffix info
base_name = input_file.replace('.fasta', '')
debug_tsv = os.path.join(log_folder, f"phase2_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
debug_log = os.path.join(log_folder, f"phase2_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

os.makedirs(output_fasta_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Phase 2 parameters
TOP_N = 3
REP_N = 40
MIN_FIT = 0.30
TOP_RATIO = 1.35
SCORE_WINDOW = 0.30

clustalo_bin = "clustalo"

# ---------------------------
# SETUP DEBUG LOGGING
# ---------------------------
debug_log_file = open(debug_log, 'w', encoding='utf-8')

def dbg(*args, **kwargs):
    """Print to both console and debug file"""
    message = ' '.join(str(arg) for arg in args)
    print(message, **kwargs)
    print(message, file=debug_log_file, **kwargs)
    sys.stdout.flush()
    debug_log_file.flush()

# ---------------------------
# UTILITIES
# ---------------------------

def run_clustalo(in_fasta, out_aln):
    try:
        subprocess.run([clustalo_bin, "-i", in_fasta, "-o", out_aln, "--force", "--outfmt=fasta"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        dbg(f"    clustalo failed: {e}")
        return False

def sample_representatives(seq_ids, scores_dict=None, n=REP_N):
    if scores_dict:
        sorted_ids = sorted(seq_ids, key=lambda x: scores_dict.get(x, 0), reverse=True)
    else:
        sorted_ids = seq_ids.copy()
    
    L = len(sorted_ids)
    if L <= n:
        return sorted_ids
    
    k_top = max(1, int(n * 0.85))
    k_spread = n - k_top
    top = sorted_ids[:k_top]
    step = max(1, L // k_spread)
    spread = sorted_ids[::step][:k_spread]
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

def post_merge_validate(subject_list, seq_db_local, new_db_local):
    with TemporaryDirectory() as tmpd:
        tmpfa = os.path.join(tmpd, "merged.fa")
        with open(tmpfa, "w") as out:
            for sid in subject_list:
                seq = seq_db_local.get(sid)
                if seq is None and new_db_local:
                    seq = new_db_local.get(sid)
                if seq:
                    out.write(f">{sid}\n{seq}\n")
        
        tmpaln = tmpfa + ".aln.fa"
        ok = run_clustalo(tmpfa, tmpaln)
        if not ok:
            return False, {"reason": "post_clustalo_fail"}
        
        mean_consensus, fits = compute_alignment_consensus_and_fits(tmpaln)
        bad = [s for s, fit in fits.items() if fit < 0.22]
        bad_frac = len(bad) / len(fits) if fits else 0.0
        
        info = {
            "mean_consensus": float(mean_consensus), 
            "bad_fraction": float(bad_frac), 
            "n_bad": len(bad),
            "example_bad": bad[:10]
        }
        
        if bad_frac > 0.22:
            info["reason"] = "too_many_bad_members"
            info["bad_members"] = bad[:10]
            return False, info
        
        return True, info

def compute_cluster_fit(new_id, cluster_ids, seq_db_local, new_db_local):
    reps = sample_representatives(cluster_ids, n=REP_N)
    if not reps:
        return None
    
    with TemporaryDirectory() as tmpd:
        fa = os.path.join(tmpd, "temp.fa")
        aln = fa + ".aln.fa"
        
        with open(fa, "w") as out:
            for sid in reps:
                seq = seq_db_local.get(sid)
                if seq:
                    out.write(f">{sid}\n{seq}\n")
            out.write(f">{new_id}\n{new_db_local[new_id]}\n")
        
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
        query_len = len(new_db_local[new_id])
        if query_len == 0:
            return None
        ref_rows = [str(r.seq) for r in alignment if r.id != new_id]
        core_cols = [i for i in range(len(aligned_query))
                     if ref_rows and sum(s[i] == '-' for s in ref_rows) / len(ref_rows) < 0.6]
        coverage = (sum(aligned_query[i] != '-' for i in core_cols) / len(core_cols)) if core_cols else 0.0

        # --- cluster average length ---
        lengths = []
        for sid in reps:
            seq = seq_db_local.get(sid)
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

def normalize_hmm_df(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = list(df.columns)
    
    dbg(f"    Reading {os.path.basename(path)} with columns: {cols}")
    
    if 'Query' in df.columns and 'Subject' in df.columns and 'Score' in df.columns:
        dbg(f"    File already in correct format")
        nd = df.copy()
        if 'Score' in nd.columns:
            nd['Score'] = pd.to_numeric(nd['Score'], errors='coerce')
        dbg(f"    Using directly: {len(nd)} rows")
        return nd
    
    dbg(f"    Attempting to map from old column names")
    column_mapping = {
        'query_profile': 'Query',
        'target_seq': 'Subject',
        'score': 'Score'
    }
    
    nd = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            nd[new_col] = df[old_col]
            dbg(f"    Mapped {old_col} -> {new_col}")
    
    if 'Score' in nd.columns:
        nd['Score'] = pd.to_numeric(nd['Score'], errors='coerce')
    else:
        nd['Score'] = np.nan
    
    if 'Query' not in nd.columns:
        nd['Query'] = os.path.splitext(os.path.basename(path))[0]
    
    dbg(f"    Normalized: {len(nd)} rows, columns: {list(nd.columns)}")
    return nd

# ---------------------------
# LOAD DATA
# ---------------------------
dbg(f"\n{'='*60}")
dbg(f"Adding sequences from: {input_file} (suffix: '{target_suffix}')")
dbg(f"{'='*60}")

dbg("Loading combined.fasta sequences...")
seq_db = {}
if os.path.exists(combined_fasta):
    for rec in SeqIO.parse(combined_fasta, "fasta"):
        seq_db[rec.id] = str(rec.seq)
else:
    dbg(f"ERROR: {combined_fasta} not found.")
    sys.exit(1)
dbg(f"Loaded {len(seq_db)} combined sequences.")

dbg(f"Loading sequences from {input_file}...")
new_db = {}
if os.path.exists(input_file):
    for rec in SeqIO.parse(input_file, "fasta"):
        new_db[rec.id] = str(rec.seq)
    dbg(f"Loaded {len(new_db)} sequences from {input_file}")
else:
    dbg(f"ERROR: {input_file} not found!")
    sys.exit(0)

new_ids = list(new_db.keys())

dbg(f"Loading deduplicated clusters from {input_fasta_folder}...")
cluster_files = glob.glob(os.path.join(input_fasta_folder, "*_merged.fasta"))
if not cluster_files:
    dbg(f"ERROR: No *_merged.fasta files found in {input_fasta_folder}")
    sys.exit(1)

clusters = {}
cluster_seqs = {}

for fa_file in cluster_files:
    cid = os.path.basename(fa_file).replace("_merged.fasta", "")
    seq_ids = []
    seq_dict = {}
    
    for rec in SeqIO.parse(fa_file, "fasta"):
        seq_ids.append(rec.id)
        seq_dict[rec.id] = str(rec.seq)
    
    clusters[cid] = seq_ids
    cluster_seqs[cid] = seq_dict
    dbg(f"  {cid}: {len(seq_ids)} sequences")

dbg(f"Loaded {len(clusters)} deduplicated clusters.")

dbg("Loading HMM CSV tables...")
if not os.path.isdir(csv_folder):
    dbg("ERROR: csv_folder not found:", csv_folder)
    sys.exit(1)

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

dbg(f"Loaded {len(cluster_tables)} HMM tables.")

# Build score dictionaries
cluster_scores = {}
for cid in clusters:
    if cid not in cluster_tables:
        cluster_scores[cid] = {}
        continue
    
    hmm_df = cluster_tables[cid]
    score_dict = {}
    
    for seq_id in clusters[cid]:
        hits = hmm_df[hmm_df['Subject'] == seq_id]['Score']
        if len(hits) > 0:
            score_dict[seq_id] = float(hits.iloc[0])
        else:
            score_dict[seq_id] = np.nan
    
    cluster_scores[cid] = score_dict

# ---------------------------
# EXTREME DEBUGGING - SEE WHAT'S IN CSV FILES
# ---------------------------
dbg("\n" + "="*80)
dbg("EXTREME DEBUGGING - CSV CONTENTS")
dbg("="*80)

# Take the first 5 CSV files and dump their contents
for cid, table in list(cluster_tables.items())[:5]:
    dbg(f"\n{cid} - first 20 subjects:")
    subjects = table['Subject'].head(20).tolist()
    for i, subj in enumerate(subjects):
        dbg(f"  {i+1}: {subj}")
    
    # Show all unique suffixes in this table
    all_suffixes = set()
    for subj in table['Subject']:
        subj_str = str(subj)
        if '_No_MCogs_no_Cogs' in subj_str:
            all_suffixes.add('_No_MCogs_no_Cogs')
        elif 'No_MCogs_no_Cogs' in subj_str:
            all_suffixes.add('No_MCogs_no_Cogs (no underscore)')
        elif 'No_MCogs_with_Cogs' in subj_str:
            all_suffixes.add('No_MCogs_with_Cogs')
        elif '_MitoCog' in subj_str:
            all_suffixes.add('_MitoCog')
        else:
            parts = subj_str.split('_')
            if len(parts) > 1:
                all_suffixes.add(f"OTHER: ..._{parts[-1]}")
    
    dbg(f"  Suffixes found: {all_suffixes}")

# Check your new_ids (sequences from input file)
dbg(f"\nFirst 20 sequences from {input_file}:")
for i, subj in enumerate(new_ids[:20]):
    dbg(f"  {i+1}: {subj}")

# Now try a direct string match test
if len(new_ids) > 0:
    test_id = new_ids[0]
    dbg(f"\nTesting direct match for first sequence:")
    dbg(f"  Looking for: '{test_id}'")

    found_in_any = False
    for cid, table in cluster_tables.items():
        if test_id in table['Subject'].values:
            found_in_any = True
            dbg(f"  ✓ Found in {cid}")
            row = table[table['Subject'] == test_id].iloc[0]
            dbg(f"    Score: {row['Score']}")
            break

    if not found_in_any:
        dbg(f"  ✗ NOT FOUND in any table")
        
        # Try partial matching
        dbg(f"\nTrying partial matches for '{test_id[:20]}...'")
        for cid, table in list(cluster_tables.items())[:3]:
            partial = test_id.split('_')[0]
            matches = table[table['Subject'].astype(str).str.contains(partial, na=False)]
            if len(matches) > 0:
                dbg(f"  {cid}: found {len(matches)} matches containing '{partial}'")
                dbg(f"    Examples: {matches['Subject'].head(3).tolist()}")

# ---------------------------
# BUILD SUBJECT HITS WITH SUFFIX FILTERING
# ---------------------------
dbg("\n" + "="*60)
dbg(f"Building subject_hits for suffix: '{target_suffix}'")
dbg("="*60)

subject_hits = defaultdict(list)
total_matches = 0
other_suffix_counts = defaultdict(int)

for cid, table in cluster_tables.items():
    # Strip '_merged' from the cluster ID to match the keys in clusters
    base_cid = cid.replace('_merged', '')
    if base_cid not in clusters:
        continue

    # Filter to only sequences with the target suffix
    if target_suffix == '_No_MCogs_no_Cogs':
        # Match both with and without underscore
        mask = table['Subject'].astype(str).str.contains(r'_?No_MCogs_no_Cogs', na=False, regex=True)
        dbg(f"  Using flexible regex for {target_suffix}")
    elif target_suffix == 'No_MCogs_with_Cogs':
        mask = table['Subject'].astype(str).str.contains(r'No_MCogs_with_Cogs', na=False, regex=True)
    else:
        mask = table['Subject'].astype(str).str.contains(target_suffix, na=False, regex=False)

    sub_tbl = table[mask]
    matches = len(sub_tbl)
    total_matches += matches

    if matches > 0:
        dbg(f"\n{base_cid}: found {matches} matches with suffix '{target_suffix}'")
        dbg(f"  Sample: {sub_tbl['Subject'].iloc[0] if matches > 0 else 'none'}")
        for _, row in sub_tbl.iterrows():
            subj = row["Subject"]
            score = float(row["Score"])
            subject_hits[subj].append((base_cid, score))
    else:
        # Count other suffixes for debugging
        for suffix in ['_No_MCogs_no_Cogs', 'No_MCogs_with_Cogs', '_MitoCog']:
            if suffix != target_suffix:
                count = len(table[table['Subject'].astype(str).str.contains(suffix, na=False)])
                if count > 0:
                    other_suffix_counts[suffix] += count

dbg(f"\nTotal matches with suffix '{target_suffix}': {total_matches}")
if other_suffix_counts:
    dbg(f"Sequences with other suffixes (ignored): {dict(other_suffix_counts)}")

dbg(f"\nUnique sequences from {input_file} with hits: {len(subject_hits)}")
dbg(f"Total sequences in {input_file}: {len(new_ids)}")
dbg(f"Coverage: {len(subject_hits)/len(new_ids)*100:.1f}%")

# ---------------------------
# DEBUGGING PHASE 2
# ---------------------------
dbg("\n" + "="*80)
dbg("STARTING PHASE 2 ASSIGNMENT")
dbg("="*80)

# Define fieldnames for TSV
fieldnames = ['sequence', 'status', 'reason', 'top_hits', 'best_cluster', 'best_fit', 
              'post_bad_frac', 'post_consensus', 'score_ratio']
for i in range(TOP_N):
    fieldnames.extend([
        f'candidate{i+1}_cluster',
        f'candidate{i+1}_hmm_score',
        f'candidate{i+1}_score_window_pass',
        f'candidate{i+1}_cluster_median',
        f'candidate{i+1}_alignment_fit',
        f'candidate{i+1}_fit_pass'
    ])

debug_rows = []
assigned_count = 0
assignments = []
cluster_updates = defaultdict(list)

for subj in new_ids:
    dbg(f"\n{'='*60}")
    dbg(f"Processing: {subj}")
    
    # Initialize debug row
    debug_row = {
        'sequence': subj,
        'status': 'unknown',
        'reason': '',
        'top_hits': '',
        'best_cluster': '',
        'best_fit': '',
        'post_bad_frac': '',
        'post_consensus': '',
        'score_ratio': ''
    }
    for i in range(TOP_N):
        debug_row[f'candidate{i+1}_cluster'] = ''
        debug_row[f'candidate{i+1}_hmm_score'] = ''
        debug_row[f'candidate{i+1}_score_window_pass'] = ''
        debug_row[f'candidate{i+1}_cluster_median'] = ''
        debug_row[f'candidate{i+1}_alignment_fit'] = ''
        debug_row[f'candidate{i+1}_fit_pass'] = ''
    
    hits = subject_hits.get(subj, [])
    if not hits:
        dbg(f"  ❌ NO HMM HITS for this sequence")
        debug_row['status'] = 'no_hmm_hits'
        debug_row['reason'] = f'No HMM hits in any cluster for suffix {target_suffix}'
        debug_rows.append(debug_row)
        continue
    
    dbg(f"  Raw HMM hits: {[(cid, f'{score:.2f}') for cid, score in hits]}")
    
    hits_sorted = sorted(hits, key=lambda x: x[1], reverse=True)[:TOP_N]
    top_hits_str = '; '.join([f"{cid}:{score:.2f}" for cid, score in hits_sorted])
    debug_row['top_hits'] = top_hits_str
    
    # Check score ratio
    if len(hits_sorted) >= 2:
        ratio = hits_sorted[0][1] / hits_sorted[1][1]
        debug_row['score_ratio'] = f"{ratio:.3f}"
        dbg(f"  Top score ratio: {ratio:.3f} (threshold: {TOP_RATIO})")
        if ratio < TOP_RATIO:
            dbg(f"  ❌ FAILED score ratio test")
            debug_row['status'] = 'failed_score_ratio'
            debug_row['reason'] = f'Score ratio {ratio:.3f} < {TOP_RATIO}'
            debug_rows.append(debug_row)
            continue
    else:
        debug_row['score_ratio'] = 'single_hit'
        dbg(f"  Only one top hit, skipping ratio test")
    
    best_alignment_cluster = None
    best_alignment_fit = -1
    candidate_results = []
    
    for idx, (cid, hmm_score) in enumerate(hits_sorted):
        candidate_num = idx + 1
        dbg(f"\n  Testing cluster {candidate_num}: {cid} (HMM score: {hmm_score:.2f})")
        
        debug_row[f'candidate{candidate_num}_cluster'] = cid
        debug_row[f'candidate{candidate_num}_hmm_score'] = f"{hmm_score:.2f}"
        
        # Score window test
        scores = list(cluster_scores[cid].values())
        scores = [s for s in scores if not np.isnan(s)]
        
        score_window_pass = False
        
        if len(scores) > 5:
            med = np.median(scores)
            debug_row[f'candidate{candidate_num}_cluster_median'] = f"{med:.2f}"
            lower = med * (1 - SCORE_WINDOW)
            upper = med * (1 + SCORE_WINDOW)
            dbg(f"    Cluster score stats: median={med:.2f}, range=[{lower:.2f}, {upper:.2f}]")
            
            if lower <= hmm_score <= upper:
                score_window_pass = True
                dbg(f"    ✅ PASSED score window test")
            else:
                dbg(f"    ❌ FAILED score window test")
        else:
            debug_row[f'candidate{candidate_num}_cluster_median'] = 'small_cluster'
            score_window_pass = True
            dbg(f"    Cluster has <=5 sequences, skipping score window test")
        
        debug_row[f'candidate{candidate_num}_score_window_pass'] = str(score_window_pass)
        
        if not score_window_pass:
            debug_row[f'candidate{candidate_num}_alignment_fit'] = 'skipped'
            debug_row[f'candidate{candidate_num}_fit_pass'] = 'False'
            candidate_results.append({'cid': cid, 'score_window_pass': False})
            continue
        
        # Alignment fit test
        dbg(f"    Computing alignment fit...")
        fit = compute_cluster_fit(subj, clusters[cid], seq_db, new_db)
        
        if fit is None:
            dbg(f"    ❌ FAILED alignment")
            debug_row[f'candidate{candidate_num}_alignment_fit'] = 'failed'
            debug_row[f'candidate{candidate_num}_fit_pass'] = 'False'
            candidate_results.append({'cid': cid, 'score_window_pass': True, 'fit': None})
            continue
        
        debug_row[f'candidate{candidate_num}_alignment_fit'] = f"{fit:.3f}"
        dbg(f"    Alignment fit: {fit:.3f}")
        
        fit_pass = fit >= MIN_FIT
        debug_row[f'candidate{candidate_num}_fit_pass'] = str(fit_pass)
        
        candidate_results.append({'cid': cid, 'score_window_pass': True, 'fit': fit, 'fit_pass': fit_pass})
        
        if fit_pass:
            dbg(f"    ✅ PASSED minimum fit threshold")
            if fit > best_alignment_fit:
                best_alignment_fit = fit
                best_alignment_cluster = cid
                dbg(f"    → New best cluster: {cid}")
        else:
            dbg(f"    ❌ FAILED minimum fit threshold")
    
    if best_alignment_cluster is None:
        dbg(f"\n  ❌ NO cluster passed all tests")
        debug_row['status'] = 'no_valid_cluster'
        debug_row['reason'] = 'Failed all tests for all candidate clusters'
        debug_rows.append(debug_row)
        continue
    
    dbg(f"\n  Best candidate: {best_alignment_cluster} (fit: {best_alignment_fit:.3f})")
    debug_row['best_cluster'] = best_alignment_cluster
    debug_row['best_fit'] = f"{best_alignment_fit:.3f}"
    
    # Post-add validation
    dbg(f"  Running post-add validation...")
    test_subjects = clusters[best_alignment_cluster] + [subj]
    ok_post, post_info = post_merge_validate(test_subjects, seq_db, new_db)
    
    if not ok_post:
        dbg(f"  ❌ FAILED post-add validation")
        if 'bad_fraction' in post_info:
            debug_row['post_bad_frac'] = f"{post_info['bad_fraction']:.3f}"
        if 'mean_consensus' in post_info:
            debug_row['post_consensus'] = f"{post_info['mean_consensus']:.3f}"
        
        debug_row['status'] = 'failed_post_validation'
        debug_row['reason'] = post_info.get('reason', 'unknown')
        debug_rows.append(debug_row)
        continue
    
    # Success!
    dbg(f"  ✅ ASSIGNED to {best_alignment_cluster}")
    cluster_updates[best_alignment_cluster].append(subj)
    assigned_count += 1
    assignments.append((subj, best_alignment_cluster, best_alignment_fit))
    
    debug_row['status'] = 'ASSIGNED'
    debug_row['reason'] = 'Success'
    debug_row['post_bad_frac'] = f"{post_info.get('bad_fraction', '')}"
    debug_row['post_consensus'] = f"{post_info.get('mean_consensus', '')}"
    debug_rows.append(debug_row)

# ---------------------------
# WRITE OUTPUTS
# ---------------------------
import csv
with open(debug_tsv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(debug_rows)

dbg(f"\n{'='*80}")
dbg(f"SUMMARY")
dbg(f"{'='*80}")
dbg(f"File processed: {input_file}")
dbg(f"Suffix used: '{target_suffix}'")
dbg(f"Total sequences in file: {len(new_ids)}")
dbg(f"Sequences with HMM hits: {len(subject_hits)}")
dbg(f"Successfully assigned: {assigned_count}")
dbg(f"Debug TSV: {debug_tsv}")

# Write updated FASTAs
dbg("\nWriting updated FASTAs...")
written = 0
for cid in clusters:
    out_fa = os.path.join(output_fasta_folder, f"{cid}_merged.fasta")
    all_seqs = cluster_seqs[cid].copy()
    
    if cid in cluster_updates:
        for new_id in cluster_updates[cid]:
            all_seqs[new_id] = new_db[new_id]
    
    with open(out_fa, "w") as fh:
        for seq_id, sequence in all_seqs.items():
            fh.write(f">{seq_id} {cid}\n{sequence}\n")
    
    written += 1
    added = len(cluster_updates.get(cid, []))
    if added > 0:
        dbg(f"  Updated {cid}: {len(all_seqs)} sequences (+{added})")

# Write report
phase2_report = os.path.join(log_folder, f"phase2_{base_name}_report.txt")
with open(phase2_report, "w") as rf:
    rf.write(f"Phase 2: Adding sequences from {input_file}\n")
    rf.write(f"Suffix: '{target_suffix}'\n")
    rf.write("=" * 70 + "\n")
    rf.write(f"Total sequences: {len(new_ids)}\n")
    rf.write(f"Successfully assigned: {assigned_count}\n")
    rf.write(f"Remaining unassigned: {len(new_ids) - assigned_count}\n\n")
    
    if assignments:
        rf.write("\nSuccessful Assignments:\n")
        rf.write("-" * 40 + "\n")
        for subj, cid, fit in sorted(assignments, key=lambda x: x[1]):
            rf.write(f"{subj} -> {cid} (fit: {fit:.3f})\n")

dbg(f"\nReport: {phase2_report}")
dbg(f"Debug log: {debug_log}")
dbg(f"Output folder: {output_fasta_folder}")
dbg("✅ Phase 2 complete.")
debug_log_file.close()
