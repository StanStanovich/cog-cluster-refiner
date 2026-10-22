"""
Regression tests for the bug fixes in original_scripts/.

The old scripts run their whole pipeline when they are imported, so the tests load only the functions
that were fixed (by reading the source with `ast`). Biopython and Clustal Omega are replaced by small
stand-ins, so the tests need nothing but numpy and pandas:

    python -m unittest discover tests -v
"""

import ast
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import common  # noqa: E402

ORIGINALS = ROOT / "original_scripts"
SCRIPTS = ["hmms_from_fasta.py", "hmmsearch_to_table.py", "new_iteration_merge2.py", "dedup_working.py", "phase2.py"]


# ---- stand-ins -------------------------------------------------------------------------------------

class FakeAlignment(list):
    def get_alignment_length(self):
        return len(self[0].seq)


class FakeAlignIO:
    """Reads aligned FASTA like Bio.AlignIO.read(path, "fasta")."""
    @staticmethod
    def read(path, fmt="fasta"):
        records = [types.SimpleNamespace(id=i, seq=s) for i, s in common.read_fasta_records(path)]
        if not records:
            raise ValueError("no records")
        return FakeAlignment(records)


def pad_aligner(in_fasta, out_aln, threads=1):
    """A 'multiple sequence alignment' that right-pads with gaps (exact for equal-length sequences)."""
    records = list(common.read_fasta_records(in_fasta))
    if len(records) < 2:
        return False
    width = max(len(s) for _, s in records)
    common.write_fasta(out_aln, [(i, s.ljust(width, "-")) for i, s in records], line_width=10 ** 6)
    return True


def load_functions(script, names, **extra):
    """Execute only the named functions of an original script and return them in a dictionary."""
    namespace = {"np": np, "pd": pd, "os": os, "subprocess": subprocess, "random": random, "zlib": zlib,
                 "TemporaryDirectory": TemporaryDirectory, "AlignIO": FakeAlignIO, "run_clustalo": pad_aligner,
                 "rep_sample_size": 50, "REP_N": 40, "FIT_SAMPLE_SIZE": 200, "merge_thresh": 0.12,
                 "min_shared_seqs": 1, "dbg": lambda *a, **k: None}
    namespace.update(extra)
    path = ORIGINALS / script
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    return namespace


AMINO = "ACDEFGHIKLMNPQRSTVWY"
RNG = random.Random(4)


def random_sequence(length=100):
    return "".join(RNG.choice(AMINO) for _ in range(length))


def mutate(seq, rate):
    return "".join(RNG.choice(AMINO) if RNG.random() < rate else c for c in seq)


class OriginalScriptsTestCase(unittest.TestCase):
    def setUp(self):
        # some functions do `from Bio import AlignIO` themselves; make that work without Biopython
        self._saved = {k: sys.modules.get(k) for k in ("Bio", "Bio.AlignIO")}
        bio = types.ModuleType("Bio")
        bio.AlignIO = FakeAlignIO
        sys.modules["Bio"], sys.modules["Bio.AlignIO"] = bio, FakeAlignIO
        self._saved_clustalo = common.run_clustalo
        common.run_clustalo = pad_aligner

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        common.run_clustalo = self._saved_clustalo


class FilesTests(OriginalScriptsTestCase):
    def test_all_original_scripts_are_present_and_valid_python(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                ast.parse((ORIGINALS / name).read_text())

    def test_every_change_is_marked(self):
        for name in ("new_iteration_merge2.py", "dedup_working.py", "phase2.py", "hmms_from_fasta.py"):
            self.assertIn("FIX", (ORIGINALS / name).read_text(), name)


class ConsensusTests(OriginalScriptsTestCase):
    def test_asparagine_is_a_normal_residue_in_all_three_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aln.fa"
            path.write_text(">a\nNNNN\n>b\nNNNN\n>c\nNNNN\n")
            for script in ("new_iteration_merge2.py", "dedup_working.py", "phase2.py"):
                with self.subTest(script=script):
                    f = load_functions(script, {"compute_alignment_consensus_and_fits"})
                    mean, fits = f["compute_alignment_consensus_and_fits"](str(path))
                    self.assertEqual(mean, 1.0)
                    self.assertEqual(fits, {"a": 1.0, "b": 1.0, "c": 1.0})

    def test_results_equal_the_refactored_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aln.fa"
            root = random_sequence(60)
            seqs = ["".join("-" if RNG.random() < 0.1 else c for c in mutate(root, 0.4)) for _ in range(12)]
            common.write_fasta(path, [(f"s{i}", s) for i, s in enumerate(seqs)], line_width=10 ** 6)
            reference = common.Alignment.from_fasta(path)
            f = load_functions("dedup_working.py", {"compute_alignment_consensus_and_fits"})
            mean, fits = f["compute_alignment_consensus_and_fits"](str(path))
            self.assertAlmostEqual(mean, reference.mean_consensus)
            for key, value in reference.member_fits().items():
                self.assertAlmostEqual(fits[key], value)


class MergeDecisionTests(OriginalScriptsTestCase):
    def setUp(self):
        super().setUp()
        # overlap path switched off (threshold 0.99, at least 5 shared), so only the score-signal path can accept
        self.should_merge = load_functions("new_iteration_merge2.py", {"should_merge_basic"},
                                           merge_thresh=0.99, min_shared_seqs=5)["should_merge_basic"]

    @staticmethod
    def table(scores, best=100.0, own="own1"):
        """Cluster table: a private best member plus shared sequences s0, s1, ..."""
        return pd.DataFrame({"Subject": [own] + [f"s{i}" for i in range(len(scores))], "Score": [best] + list(scores)})

    def test_correlation_compares_the_same_sequences(self):
        shared = [f"s{i}" for i in range(6)]
        t1 = pd.DataFrame({"Subject": shared, "Score": [100, 90, 80, 70, 60, 50]}).sort_values("Score", ascending=False)
        t2 = pd.DataFrame({"Subject": shared, "Score": [50, 60, 70, 80, 90, 100]}).sort_values("Score", ascending=False)
        _, _, meta = self.should_merge(t1, t2)
        self.assertAlmostEqual(meta["corr"], -1.0)

    def test_scores_of_different_scale_are_made_comparable(self):
        # profile 2 scores ten times lower, but both rate the shared sequences the same relative to their best
        ok, _, meta = self.should_merge(self.table([90, 85, 80, 75]), self.table([9, 8.5, 8, 7.5], best=10.0, own="own2"))
        self.assertTrue(ok)
        self.assertEqual(meta["path"], "signal")
        self.assertAlmostEqual(meta["max_cross"], 0.9)

    def test_pair_linked_only_by_weak_hits_is_rejected(self):
        ok, _, meta = self.should_merge(self.table([46, 45, 44, 43]), self.table([46, 45, 44, 43], own="own2"))
        self.assertFalse(ok)
        self.assertLess(meta["max_cross"], 0.55)

    def test_decisions_equal_the_refactored_implementation(self):
        should_merge = load_functions("new_iteration_merge2.py", {"should_merge_basic"})["should_merge_basic"]
        spec = types.ModuleType("merge03")
        source = (ROOT / "scripts" / "03_merge_clusters.py").read_text()
        sys.path.insert(0, str(ROOT / "scripts"))
        exec(compile(source, "03_merge_clusters.py", "exec"), spec.__dict__)
        cfg = types.SimpleNamespace(min_shared=1, min_overlap=0.12, signal_min_corr=0.75,
                                    signal_min_mean_score=0.35, signal_min_max_score=0.55)
        rng = np.random.default_rng(3)
        for _ in range(60):
            n1, n2 = int(rng.integers(4, 30)), int(rng.integers(4, 30))
            k = min(int(rng.integers(2, 4 + min(n1, n2))), n1, n2)
            shared = [f"x{i}" for i in range(k)]
            t1 = pd.DataFrame({"Subject": shared + [f"a{i}" for i in range(n1 - k)],
                               "Score": rng.uniform(30, 150, n1)}).sort_values("Score", ascending=False)
            t2 = pd.DataFrame({"Subject": shared + [f"b{i}" for i in range(n2 - k)],
                               "Score": rng.uniform(30, 150, n2)}).sort_values("Score", ascending=False)
            ok_old, combined_old, meta_old = should_merge(t1, t2)
            ok_new, combined_new, meta_new = spec.evaluate_pair(t1, t2, cfg)
            self.assertEqual(ok_old, ok_new)
            self.assertAlmostEqual(combined_old, combined_new)
            self.assertAlmostEqual(meta_old["corr"], meta_new["score_corr"])
            self.assertAlmostEqual(meta_old["max_cross"], meta_new["max_relative_score"])


class FitTests(OriginalScriptsTestCase):
    def setUp(self):
        super().setUp()
        root = random_sequence(100)
        self.members = {f"m{i}": mutate(root, 0.15) for i in range(12)}
        self.ids = list(self.members)
        self.full = mutate(root, 0.15)
        self.fragment = self.full[:70]

    NAMES = {"sample_representatives", "compute_alignment_consensus_and_fits", "compute_cluster_fit"}

    def test_merge_script_uses_the_sequence_not_the_dictionary(self):
        compute = load_functions("new_iteration_merge2.py", self.NAMES)["compute_cluster_fit"]
        db = {k: {"sequence": v, "description": k, "source_file": "combined.fasta"}
              for k, v in {**self.members, "Q": self.full}.items()}   # the way this script stores sequences
        df = pd.DataFrame({"Subject": self.ids, "Score": np.linspace(100, 60, len(self.ids))})
        fit = compute(("Q"), df, db, db)
        self.assertGreater(fit, 0.0)
        self.assertLess(fit, 1.0)       # the unfixed function returned values far above 1
        self.assertAlmostEqual(fit, common.normalized_fit("Q", {**self.members, "Q": self.full}, self.ids, len(self.ids)))

    def test_coverage_penalises_a_fragment_in_all_three_scripts(self):
        plain_full = {**self.members, "Q": self.full}
        plain_frag = {**self.members, "Q": self.fragment}
        expected_full = common.normalized_fit("Q", plain_full, self.ids, len(self.ids))
        expected_frag = common.normalized_fit("Q", plain_frag, self.ids, len(self.ids))
        self.assertLess(expected_frag, expected_full)

        df = pd.DataFrame({"Subject": self.ids, "Score": np.linspace(100, 60, len(self.ids))})
        merge = load_functions("new_iteration_merge2.py", self.NAMES)["compute_cluster_fit"]
        phase2 = load_functions("phase2.py", self.NAMES)["compute_cluster_fit"]
        for label, plain, expected in (("full", plain_full, expected_full), ("fragment", plain_frag, expected_frag)):
            dict_db = {k: {"sequence": v} for k, v in plain.items()}
            with self.subTest(script="merge", query=label):
                self.assertAlmostEqual(merge("Q", df, dict_db, dict_db), expected)
            with self.subTest(script="phase2", query=label):
                self.assertAlmostEqual(phase2("Q", self.ids, plain, plain), expected)

    def test_dedup_does_not_align_a_sequence_with_itself_and_has_a_stable_seed(self):
        seen = []

        def recording_aligner(in_fasta, out_aln, threads=1):
            seen.append([i for i, _ in common.read_fasta_records(in_fasta)])
            return pad_aligner(in_fasta, out_aln)

        funcs = load_functions("dedup_working.py", {"random_sample_representatives", "compute_alignment_consensus_and_fits",
                                                    "compute_cluster_fit"}, run_clustalo=recording_aligner)
        plain = {**self.members, "Q": self.full}
        cluster = self.ids + ["Q"]               # as in the real run, the sequence is a member of the cluster
        fit = funcs["compute_cluster_fit"]("Q", cluster, plain, self.full, "clusterX", "Q")
        self.assertEqual(seen[0].count("Q"), 1)
        self.assertAlmostEqual(fit, common.normalized_fit("Q", plain, self.ids, cluster_size=len(cluster)))

    def test_dedup_script_never_calls_hash(self):
        tree = ast.parse((ORIGINALS / "dedup_working.py").read_text())
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn("hash", calls)


if __name__ == "__main__":
    unittest.main()
