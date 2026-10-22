# Original scripts (with bug fixes)

These are the scripts the clusters were first produced with, kept here so that they are easy to find.
**Only definite defects were fixed, and nothing else was changed**: no new features, no refactoring, the
same configuration blocks, file names, thresholds and outputs. Every change is marked with a
`# FIX` comment that says what was wrong. The maintained, refactored version of the pipeline is in the
parent folder (see the main README). On the synthetic data used to test both, the fixed scripts here and
the refactored pipeline produced the same clusters.

## Order in which the scripts were used

| step | script | reads | writes |
|---|---|---|---|
| 1 | `hmms_from_fasta.py` | `COGs/` (one FASTA per cluster) | `alignments/`, `profiles/` |
| 2 | `hmmsearch_to_table.py` | `profiles/` and the four target FASTA files named in its configuration block | `hmmsearch_combined/` |
| 3 | `new_iteration_merge2.py` | `COGs/`, `hmmsearch_combined/`, `combined.fasta` | `addcog_only_fastas2/`, `addcog_only_logs/` |
| 4 | `dedup_working.py` | `addcog_only_fastas2/` | `addcog_only_fastas2_dedup2/`, `logs/` |
| 5 | `phase2.py <file>` (once, after the first iteration) | `COGs/` (the deduplicated clusters, named `*_merged.fasta`), `hmmsearch_combined/`, `combined.fasta`, `<file>` | updates `addcog_only_fastas2_dedup2/` |
| 6 | `dedup_working.py` again, "just in case" | | |

The second and third iteration repeat steps 1 to 4 with the deduplicated clusters as the new `COGs/`. The
parameters of each iteration are set by hand in the configuration block at the top of
`new_iteration_merge2.py`. The values of the three iterations are in
`../schedules/relaxing_3_iterations.json`, under these names:

| variable in `new_iteration_merge2.py` | key in the schedule file |
|---|---|
| `expand_tol` | `score_window` |
| `merge_thresh` | `min_overlap` |
| `min_shared_seqs` | `min_shared` |
| `rep_sample_size` | `rep_sample_size` |
| `pre_mean_consensus_min` | `pre_min_consensus` |
| `pre_min_cluster_fit` | `expand_min_fit` and `pre_min_fit` |
| `pre_max_fit_diff` | `pre_max_fit_diff` |
| `post_bad_member_frac` | `post_max_bad_fraction` |
| `post_member_fit_thresh` | `post_member_fit` |
| `composition_shift_limit` | (has no effect in this script) |

All paths are relative: run the scripts from a folder that contains the input folders and files above.

## Requirements

Python 3 with `numpy`, `pandas` and **Biopython** (`pip install biopython`; the refactored pipeline does
not need it), plus HMMER and Clustal Omega on the PATH.

## What was fixed

| script | fix | why |
|---|---|---|
| `new_iteration_merge2.py` | `compute_cluster_fit` uses the sequence string instead of the dictionary that this script stores for every sequence | The dictionary text was written into the FASTA file as the "sequence", and its length (3) was used as the sequence length, so the fit of every candidate was meaningless. |
| `new_iteration_merge2.py` | The score correlation of the shared sequences pairs the two profiles' scores by sequence ID | The two score lists were paired by row position, so scores of different sequences were correlated. Both tables are sorted by their own scores, so the correlation came out high almost regardless of the real relationship. |
| `new_iteration_merge2.py` | The scores in the score-signal path are divided by the best score each profile gives to its own cluster | The thresholds 0.35 and 0.55 are meant for relative scores; raw bit scores always satisfied them. |
| `new_iteration_merge2.py` | In the pre-merge check a sequence shared by both clusters is written once into the alignment file | It was written twice, with the same ID. |
| `new_iteration_merge2.py`, `dedup_working.py`, `phase2.py` | `N` is no longer ignored when the consensus is computed | `N` is asparagine in a protein, not an unknown base, so it could never match the consensus. |
| `new_iteration_merge2.py`, `dedup_working.py`, `phase2.py` | `coverage` is the fraction of the core columns (columns where fewer than 60% of the reference sequences have a gap) in which the query has a residue | It was computed as non-gap characters divided by sequence length, which is always 1.0. |
| `dedup_working.py` | The random sample is seeded with a CRC32 value instead of `hash()` | `hash()` of a string differs from run to run, so the sample was not reproducible. |
| `dedup_working.py` | The sequence being placed is left out of the sample it is compared with | It is a member of every cluster it is duplicated in, so it was aligned against itself. |
| `hmms_from_fasta.py` | A comment said `"clustalo"` twice; it now says `"mafft"` as the alternative | Comment only. |

`hmmsearch_to_table.py` is unchanged.

## What this means for your results

Some of these fixes change which clusters are merged, so the fixed scripts can give different results
from your earlier runs; the effect of each fix is described in `../CHANGES_FROM_ORIGINAL.md`. The largest
effect is the correlation fix: with the old pairing, nearly every pair of clusters that shared two or
more sequences became a merge candidate. To get roughly that behaviour back on purpose, set
`merge_thresh = 0.0` and `min_shared_seqs = 2` in the configuration block of `new_iteration_merge2.py`;
the alignment checks then decide, as before.

## Left as it was

All thresholds and the algorithm itself, including things that look odd but are not defects: the
"composition shift" setting that has no effect, the score window of `phase2.py` (0.30), the
dataset-specific file names and suffixes, and the hard-coded folders.
