# Differences from the original scripts

This repository is a cleaned-up version of the scripts that were first used to produce the
clusters (`hmms_from_fasta.py`, `hmmsearch_to_table.py`, `new_iteration_merge2.py`,
`dedup_working.py`, `phase2.py`). While refactoring, several defects were found and fixed.
**Because some of them influence which clusters are merged, results of this version can differ
from results obtained with the original scripts.** The original scripts themselves are kept in
`original_scripts/` with the same fixes applied (and nothing else changed); to reproduce earlier results
exactly, use your own unfixed copies of them.

## Changes that can change the results

| # | Original behaviour | Now | Effect |
|---|---|---|---|
| 1 | The correlation of the scores of shared sequences paired the two score lists by *row position*. Both tables are sorted best-first by their own profile, so the pairs were mismatched and the correlation came out high (≈ 0.9–1.0) almost regardless of the real relationship. | Scores are paired by sequence ID. | **Large.** With the old pairing, a pair of clusters sharing 2 or more sequences almost always became a merge candidate through the score-signal path. With the true correlation this no longer happens: the shared sequences are typically each cluster's own members, which score highest against their own profile, so the true correlation is often negative. Candidates now need an overlap of about 0.17 (`--min-overlap` ÷ 0.7). To approximate the old candidate selection use `--min-overlap 0 --min-shared 2`; the alignment checks then decide. |
| 2 | Signal-path conditions on mean/max score were applied to raw bit scores (tens to hundreds) and were therefore always true. | Scores are divided by the best score the profile gives to any of its own members. | Small (the signal path rarely fires, see 1). |
| 3 | In `new_iteration_merge2.py` the sequence database held dictionaries; `compute_cluster_fit` wrote the dictionary text as the "sequence" into the FASTA given to Clustal Omega. | Real sequences are used. | **Large for expansion:** candidates could not be evaluated properly before. |
| 4 | `N` (asparagine) was ignored in the consensus as if it were an unknown nucleotide. | Only gaps are ignored. | Small to moderate: Asn is ~4 % of residues. |
| 5 | `coverage` in the fit score was computed as non-gap residues / sequence length, which is always 1. | Fraction of the *core* alignment columns that the sequence spans (same definition of core as the pre-merge check). | ≈ 1 for full-length sequences, lower for fragments. |
| 6 | The sequence being tested could be part of its own reference set (dedup), and shared sequences were written twice into the same alignment file (pre-merge check). | Excluded / written once. | Small. |
| 7 | The random sample in the deduplication was seeded with Python's `hash()`, which differs from run to run. | CRC32-based seed (`--seed`). | The sample is now reproducible. |
| 8 | A cluster that had no usable profile hits was only protected by a safety net that restored *orphaned sequences*. | It is passed on unchanged. | Only differs in rare edge cases. |
| 9 | Scores of a sequence found in several target files were all used. | Each sequence keeps its best score. | Only matters with several target files. |
| 10 | Residues were compared case-sensitively. | Case-insensitively. | Only matters for inputs with lowercase letters. |

## Changes that should not change the results

* The "composition shift" check was removed: it compared a single source file with itself and could never trigger.
* Merge candidates are found through an index of sequences instead of testing all pairs (same pairs).
* Step 5 (former `phase2.py`) no longer uses dataset-specific suffixes to tell sequence sets apart;
  it takes a FASTA file of unclustered sequences. Cluster members are ranked by profile score
  when representatives are chosen. Debug dumps were removed.
* `hmmbuild` is called with `--amino --informat afa -n <cluster>`; one profile name per cluster.
* Output files are named `<cluster>.fasta` (no `_merged` suffix), logs are CSV/TSV/JSON files.
* Parameters are command-line options; the per-iteration values are in `schedules/`.

## Safeguards that did not exist before

* A work folder that already contains iteration folders is refused (results of different runs are not mixed), and
  step outputs are written only to folders without FASTA files unless `--overwrite` is given.
* All options of all iterations are validated before `run_pipeline.py` starts any work.
* Every step aborts without writing if a sequence of its input would be lost, and the driver compares the final
  clusters with the input at the end.
* Step 1 stops if more than half of the profile builds fail (usually a broken HMMER or Clustal Omega
  installation), and a failing step stops `run_pipeline.py` with a clear message instead of a traceback.
* Steps 3 and 5 stop if the hit-table folder contains no tables (usually a wrong folder name) instead of
  silently doing nothing.
* `run_pipeline.py` saves the command line, the schedule and the HMMER / Clustal Omega versions in
  `<workdir>/run_info.json`.
* If the seed alignment of a cluster cannot be built (for example for a one-member cluster) only the
  conservation test is skipped for it; the fit test still decides.

## Things that were looked at and deliberately left as they are

* The expansion conservation test (`--expand-consensus-drop`) hardly ever rejects anything
  (see the counters printed by step 3); it is kept because it is part of the method.
* The pre-merge check compares expanded clusters, which overlap strongly by construction, so its
  fit-difference rule mainly acts on partially overlapping clusters; the post-merge check on the
  complete merged cluster is the independent safeguard.
* The post-merge check limits the fraction of badly fitting members of the *whole* merged cluster. A small
  poorly fitting cluster merged into a large one can pass. `merge_decisions.csv` therefore also reports
  `post_bad_fraction_1` and `post_bad_fraction_2` for the two merged clusters.
* The score window of the expansion step is centred on the median score of the cluster's own members.
* The size factor in the fit score favours large clusters, so small clusters tend to be absorbed by
  large ones rather than to attract new sequences.
