"""
Unit tests for the pure-Python parts of the pipeline.

They need only numpy and pandas (no Clustal Omega / HMMER) and run in a second:

    python -m unittest discover tests -v
"""

import importlib.util
import random
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402


def load_script(filename: str):
    """Import a script whose file name starts with a digit (so a normal `import` is impossible)."""
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_from(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_script("03_merge_clusters.py")
dedup = load_script("04_deduplicate_clusters.py")
add_unclustered = load_script("05_add_unclustered.py")


def make_alignment(**named_seqs) -> common.Alignment:
    return common.Alignment(list(named_seqs), list(named_seqs.values()))


class FastaTests(unittest.TestCase):
    def test_round_trip_and_id_is_first_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.fasta"
            path.write_text(">seq1 some description\nACDE\nFGHI\n\n>seq2\nKLMN\n")
            self.assertEqual(list(common.read_fasta_records(path)),
                             [("seq1", "ACDEFGHI"), ("seq2", "KLMN")])

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y.fasta"
            long_seq = "ACDEFGHIKLMNPQRSTVWY" * 10
            common.write_fasta(path, [("a clusterX", long_seq), ("b clusterX", "MK")])
            self.assertEqual(list(common.read_fasta_records(path)), [("a", long_seq), ("b", "MK")])


class AlignmentTests(unittest.TestCase):
    def test_consensus_and_fits(self):
        aln = make_alignment(a="ACDE", b="ACDF", c="A-DE")
        self.assertEqual("".join(aln.consensus), "ACDE")
        np.testing.assert_allclose(aln.consensus_strength, [1, 1, 1, 2 / 3])
        self.assertAlmostEqual(aln.mean_consensus, (3 + 2 / 3) / 4)
        fits = aln.member_fits()
        self.assertAlmostEqual(fits["a"], 1.0)
        self.assertAlmostEqual(fits["b"], 3 / 4)   # last residue differs from the consensus
        self.assertAlmostEqual(fits["c"], 1.0)     # the gap is not counted

    def test_ties_resolve_alphabetically(self):
        aln = make_alignment(a="A", b="C")
        self.assertEqual(aln.consensus[0], "A")

    def test_asparagine_is_a_normal_residue(self):
        # Regression test: 'N' (asparagine) used to be treated like an unknown base and ignored.
        aln = make_alignment(a="NNNN", b="NNNN", c="NNNN")
        self.assertEqual(aln.mean_consensus, 1.0)
        self.assertEqual(aln.member_fits(), {"a": 1.0, "b": 1.0, "c": 1.0})

    def test_core_columns_ignore_domain_only_present_in_one_group(self):
        #            |--- shared domain ---|--- domain only in group 1 ---|
        aln = make_alignment(
            g1a="ACDEF" + "WWWWW", g1b="ACDEF" + "WWWWW",
            g2a="ACDEF" + "-----", g2b="ACDEG" + "-----")
        fit1, fit2, n_core = aln.core_group_fits(["g1a", "g1b"], ["g2a", "g2b"], gap_tolerance=0.6)
        self.assertEqual(n_core, 5)       # the 5 columns where group 2 is all gaps are not core
        self.assertAlmostEqual(fit1, 1.0)
        self.assertAlmostEqual(fit2, 0.9)  # g2b differs in one of five core columns

    def test_unequal_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            common.Alignment(["a", "b"], ["ACD", "AC"])


class MergeCandidateTests(unittest.TestCase):
    cfg = SimpleNamespace(min_shared=1, min_overlap=0.12, signal_min_corr=0.75,
                          signal_min_mean_score=0.35, signal_min_max_score=0.55)

    @staticmethod
    def table(**scores):
        return pd.DataFrame({"Subject": list(scores), "Score": list(scores.values())})

    def test_score_correlation_compares_the_same_sequences(self):
        # Regression test: scores used to be paired by row position instead of by sequence ID.
        # Here the two profiles rank the six shared sequences in exactly opposite order, so the
        # correlation must be -1, even though both tables are sorted best-first.
        t1 = self.table(s0=100, s1=90, s2=80, s3=70, s4=60, s5=50).sort_values("Score", ascending=False)
        t2 = self.table(s0=50, s1=60, s2=70, s3=80, s4=90, s5=100).sort_values("Score", ascending=False)
        _, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertAlmostEqual(metrics["score_corr"], -1.0)

    def test_no_shared_sequences_is_never_a_candidate(self):
        ok, score, _ = merge.evaluate_pair(self.table(a=10, b=9), self.table(c=10, d=9), self.cfg)
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_enough_overlap_is_a_candidate(self):
        t1 = self.table(a=10, b=9, c=8, d=7)
        t2 = self.table(c=9, d=8, e=7, f=6)
        ok, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertTrue(ok)
        self.assertEqual(metrics["shared"], 2)
        self.assertAlmostEqual(metrics["overlap"], 0.5)

    def test_pair_search_matches_brute_force(self):
        rng = random.Random(3)
        universe = [f"s{i}" for i in range(60)]
        clusters = {}
        for k in range(15):
            members = rng.sample(universe, rng.randint(2, 12))
            clusters[f"c{k}"] = pd.DataFrame({"Subject": members, "Score": [rng.uniform(20, 100) for _ in members]})

        fast = {(a, b) for a, b, _, _ in merge.find_candidate_merges(clusters, self.cfg)}

        order = sorted(clusters, key=lambda c: len(clusters[c]), reverse=True)
        slow = set()
        for cid1, cid2 in combinations(order, 2):
            ok, _, _ = merge.evaluate_pair(clusters[cid1], clusters[cid2], self.cfg)
            if ok:
                slow.add((cid1, cid2))
        self.assertEqual(fast, slow)


class RepresentativeTests(unittest.TestCase):
    def test_small_cluster_returns_everything_in_the_given_order(self):
        self.assertEqual(common.pick_representatives(["b", "c", "a"], 10), ["b", "c", "a"])

    def test_large_cluster_is_capped_unique_and_contains_the_best(self):
        ranked = [f"s{i}" for i in range(500)]
        reps = common.pick_representatives(ranked, 50)
        self.assertLessEqual(len(reps), 50)
        self.assertEqual(len(reps), len(set(reps)))
        self.assertEqual(reps[:42], ranked[:42])  # top 85% of the slots

    def test_one_slot_does_not_crash(self):
        self.assertEqual(common.pick_representatives(["a", "b"], 1), ["a"])

    def test_ranking_of_a_cluster_table(self):
        table = pd.DataFrame({"Subject": ["a", "b", "c"], "Score": [1.0, 3.0, 2.0]})
        self.assertEqual(merge.ranked_ids(table), ["b", "c", "a"])


class DedupTests(unittest.TestCase):
    def test_seed_is_stable_across_runs(self):
        # Fixed expected values: CRC32 is defined by a standard, whereas Python's built-in
        # hash() of a string differs from one run to the next.
        self.assertEqual(dedup.stable_seed(1, "COG0001", "seqA"), 1240817081)
        self.assertEqual(dedup.stable_seed("a", "b"), 2523025035)
        self.assertNotEqual(dedup.stable_seed("a", "b"), dedup.stable_seed("b", "a"))


class CoverageTests(unittest.TestCase):
    CORE = "ACDEFGHIKL" * 10

    def alignment_with_one_insertion(self, query: str) -> common.Alignment:
        gap = "-" * 50
        refs = {f"ref{i}": self.CORE[:50] + ("W" * 50 if i == 0 else gap) + self.CORE[50:] for i in range(8)}
        return common.Alignment(list(refs) + ["query"], list(refs.values()) + [query])

    def test_insertion_in_one_reference_does_not_penalise_a_full_length_query(self):
        aln = self.alignment_with_one_insertion(self.CORE[:50] + "-" * 50 + self.CORE[50:])
        refs = [f"ref{i}" for i in range(8)]
        self.assertEqual(aln.core_coverage("query", refs), 1.0)
        # ...whereas "non-gap columns / all columns" would have said 100/150
        self.assertAlmostEqual((aln.matrix[aln.index["query"]] != "-").mean(), 2 / 3)

    def test_fragment_is_penalised(self):
        aln = self.alignment_with_one_insertion(self.CORE[:50] + "-" * 50 + self.CORE[50:60] + "-" * 40)
        refs = [f"ref{i}" for i in range(8)]
        self.assertAlmostEqual(aln.core_coverage("query", refs), 0.6)


class SignalScoreTests(unittest.TestCase):
    # overlap path switched off (min_overlap 0.99), so only the score-signal path can accept a pair
    cfg = SimpleNamespace(min_shared=5, min_overlap=0.99, signal_min_corr=0.75,
                          signal_min_mean_score=0.35, signal_min_max_score=0.55)

    @staticmethod
    def pair(shared1, shared2, top1=100.0, top2=100.0):
        """Two cluster tables that share sequences s0..sN. Each also has one private best member."""
        shared = [f"s{i}" for i in range(len(shared1))]
        t1 = pd.DataFrame({"Subject": ["own1"] + shared, "Score": [top1] + list(shared1)})
        t2 = pd.DataFrame({"Subject": ["own2"] + shared, "Score": [top2] + list(shared2)})
        return t1, t2

    def test_scores_on_different_scales_are_comparable(self):
        # Profile 2 produces scores ten times lower than profile 1 (e.g. a shorter profile),
        # but both rate the shared sequences as equally good relative to their own best member.
        t1, t2 = self.pair([90.0, 85.0, 80.0, 75.0], [9.0, 8.5, 8.0, 7.5], top1=100.0, top2=10.0)
        ok, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertTrue(ok)
        self.assertEqual(metrics["path"], "signal")
        self.assertAlmostEqual(metrics["max_relative_score"], 0.9)

    def test_max_condition_rejects_a_pair_linked_only_by_weak_hits(self):
        # Correlated scores, mean passes (0.45 >= 0.35), but no shared sequence reaches 55% of the
        # profile's best member. Only the max condition can reject this pair.
        t1, t2 = self.pair([46.0, 45.0, 44.0, 43.0], [46.0, 45.0, 44.0, 43.0])
        ok, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertFalse(ok)
        self.assertGreaterEqual(metrics["mean_relative_score"], 0.35)
        self.assertLess(metrics["max_relative_score"], 0.55)

    def test_one_strong_shared_hit_is_enough_for_the_max_condition(self):
        t1, t2 = self.pair([70.0, 45.0, 44.0, 43.0], [70.0, 45.0, 44.0, 43.0])
        ok, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertTrue(ok)
        self.assertAlmostEqual(metrics["max_relative_score"], 0.7)

    def test_mean_condition_rejects_uniformly_low_scores(self):
        t1, t2 = self.pair([60.0, 10.0, 9.0, 8.0], [60.0, 11.0, 10.0, 9.0])
        ok, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertFalse(ok)
        self.assertLess(metrics["mean_relative_score"], 0.35)

    def test_one_profile_without_scores_does_not_hide_the_other(self):
        t1, t2 = self.pair([np.nan] * 4, [80.0, 70.0, 60.0, 50.0])
        _, _, metrics = merge.evaluate_pair(t1, t2, self.cfg)
        self.assertAlmostEqual(metrics["max_relative_score"], 0.8)


class FitSummaryTests(unittest.TestCase):
    def test_pooled_fraction_can_hide_a_bad_small_part(self):
        # 80 good members of cluster 1 and 20 members of cluster 2 that all fit badly:
        # the pooled bad fraction is only 20%, but cluster 2 on its own is 100% bad.
        fits = {f"a{i}": 0.8 for i in range(80)} | {f"b{i}": 0.05 for i in range(20)}
        summary = common.summarize_fits(fits, 0.22, groups={"1": [f"a{i}" for i in range(80)],
                                                              "2": [f"b{i}" for i in range(20)]})
        self.assertAlmostEqual(summary["bad_fraction"], 0.20)
        self.assertEqual(summary["bad_fraction_1"], 0.0)
        self.assertEqual(summary["bad_fraction_2"], 1.0)

    def test_shared_members_count_in_both_groups(self):
        summary = common.summarize_fits({"x": 0.1, "y": 0.9}, 0.22, groups={"1": ["x", "y"], "2": ["x"]})
        self.assertEqual((summary["bad_fraction_1"], summary["bad_fraction_2"]), (0.5, 1.0))


class CoreColumnTests(unittest.TestCase):
    def test_coverage_and_pre_merge_use_the_same_definition_of_core(self):
        # Column 4 is occupied by 9 of 20 reference sequences (45%): fewer than 60% gaps -> core.
        # Column 5 by 7 of 20 (35%): more than 60% gaps -> not core.
        refs = {}
        for i in range(20):
            seq = list("ACDE") + ["F" if i < 9 else "-"] + ["G" if i < 7 else "-"]
            refs[f"r{i}"] = "".join(seq)
        aln = common.Alignment(list(refs) + ["q"], list(refs.values()) + ["ACDE-G"])  # q misses column 4
        ids = list(refs)

        core = aln.core_columns([aln.index[r] for r in ids])
        self.assertEqual(core.tolist(), [True, True, True, True, True, False])

        # coverage: q has a residue in 4 of the 5 core columns
        self.assertAlmostEqual(aln.core_coverage("q", ids), 4 / 5)
        # pre-merge comparison: the same five columns
        _, _, n_core = aln.core_group_fits(ids[:10], ids[10:], gap_tolerance=0.6)
        self.assertEqual(n_core, 4)   # ...of which column 4 is not core in the half that has few residues there


class AddUnclusteredTests(unittest.TestCase):
    def test_score_ratio(self):
        self.assertIsNone(add_unclustered.score_ratio([("A", 50.0)]))
        self.assertAlmostEqual(add_unclustered.score_ratio([("A", 90.0), ("B", 60.0)]), 1.5)
        self.assertEqual(add_unclustered.score_ratio([("A", 90.0), ("B", 0.0)]), float("inf"))

    def test_score_window(self):
        member_scores = [100.0] * 6
        self.assertEqual(add_unclustered.in_score_window(75.0, member_scores, 0.3, 6), (True, 100.0))
        self.assertEqual(add_unclustered.in_score_window(60.0, member_scores, 0.3, 6), (False, 100.0))
        self.assertEqual(add_unclustered.in_score_window(130.0, member_scores, 0.3, 6), (True, 100.0))

    def test_small_clusters_skip_the_window(self):
        self.assertEqual(add_unclustered.in_score_window(1.0, [100.0] * 5, 0.3, 6), (True, None))


class DriverTests(unittest.TestCase):
    def test_schedule_keys_become_command_line_flags(self):
        sys.path.insert(0, str(SCRIPTS.parent))
        driver = load_script_from(SCRIPTS.parent / "run_pipeline.py")
        self.assertEqual(driver.flags({"score_window": 0.35, "min_shared": 2, "_note": "ignored"}),
                         ["--score-window", "0.35", "--min-shared", "2"])
        self.assertEqual(driver.flags({"a": 1}, "--b 2"), ["--a", "1", "--b", "2"])

    def test_shipped_schedules_only_use_options_the_scripts_understand(self):
        import json
        driver = load_script_from(SCRIPTS.parent / "run_pipeline.py")
        required = {"03": ["--clusters", "x", "--hits", "x"],
                    "05": ["--clusters", "x", "--hits", "x", "--unclustered", "x"]}
        files = sorted((SCRIPTS.parent / "schedules").glob("*.json"))
        self.assertGreaterEqual(len(files), 2)
        for path in files:
            schedule = json.loads(path.read_text())
            cases = [(it["merge"], merge, required["03"]) for it in schedule["iterations"]]
            cases.append((schedule["add_unclustered"], add_unclustered, required["05"]))
            for params, module, base_args in cases:
                saved = sys.argv
                sys.argv = ["script", *base_args, *driver.flags(params)]
                try:
                    module.parse_args()   # exits with an error on any unknown option or bad value
                finally:
                    sys.argv = saved


class ScheduleSelectionTests(unittest.TestCase):
    def setUp(self):
        self.driver = load_script_from(SCRIPTS.parent / "run_pipeline.py")

    def test_default_is_relaxing_3_iterations(self):
        self.assertEqual(self.driver.DEFAULT_SCHEDULE, "relaxing_3_iterations")
        self.assertEqual(self.driver.build_parser().parse_args(["--input", "x"]).schedule, "relaxing_3_iterations")

    def test_schedule_can_be_chosen_by_name_by_path_or_switched_off(self):
        resolve = self.driver.resolve_schedule
        by_name = resolve("relaxing_3_iterations_any_two_shared")
        self.assertEqual(by_name.name, "relaxing_3_iterations_any_two_shared.json")
        self.assertEqual(resolve("relaxing_3_iterations_any_two_shared.json"), by_name)
        self.assertEqual(resolve(str(by_name)), by_name)
        self.assertIsNone(resolve("none"))
        self.assertIsNone(resolve("NONE"))
        with self.assertRaises(SystemExit):
            resolve("no_such_schedule")

    def test_own_schedule_file_can_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            mine = Path(tmp) / "mine.json"
            mine.write_text('{"iterations": [{"merge": {"score_window": 0.5}}]}')
            self.assertEqual(self.driver.resolve_schedule(str(mine)), mine)
            self.assertEqual(self.driver.load_schedule(str(mine))["iterations"][0]["merge"]["score_window"], 0.5)

    def test_every_shipped_schedule_is_described(self):
        # --list-schedules prints these descriptions
        import json
        for name, path in self.driver.available_schedules().items():
            with self.subTest(schedule=name):
                self.assertGreater(len(json.loads(path.read_text()).get("description", "").split()), 10)

    def test_run_info_records_the_schedule_and_survives_missing_programs(self):
        import json
        args = self.driver.build_parser().parse_args(["--input", "x"])
        with tempfile.TemporaryDirectory() as tmp:
            schedule_path = self.driver.resolve_schedule("relaxing_3_iterations")
            self.driver.write_run_info(Path(tmp) / "work", args, schedule_path, json.loads(schedule_path.read_text()))
            info = json.loads((Path(tmp) / "work" / "run_info.json").read_text())
        self.assertTrue(info["schedule_file"].endswith("relaxing_3_iterations.json"))
        self.assertEqual(len(info["schedule"]["iterations"]), 3)
        self.assertIn("hmmsearch", info)
        self.assertEqual(self.driver.tool_version(["definitely-not-a-program"], "x"), "unknown")
        # a program that runs but fails must not have its error text recorded as a "version"
        self.assertEqual(self.driver.tool_version([sys.executable, "-c", "import sys; print('oops'); sys.exit(1)"], ""),
                         "unknown")


class DocumentationTests(unittest.TestCase):
    PROGRAMS = ["run_pipeline.py", "scripts/01_build_hmms.py", "scripts/02_search_hmms.py",
                "scripts/03_merge_clusters.py", "scripts/04_deduplicate_clusters.py",
                "scripts/05_add_unclustered.py"]

    def test_every_option_has_a_real_description(self):
        for relative in self.PROGRAMS:
            parser = load_script_from(SCRIPTS.parent / relative).build_parser()
            for action in parser._actions:
                if action.option_strings and action.dest != "help":
                    with self.subTest(script=relative, option=action.option_strings[0]):
                        self.assertGreaterEqual(len((action.help or "").split()), 8,
                                                "description is missing or too short")

    def test_parameter_reference_is_up_to_date(self):
        tool = load_script_from(SCRIPTS.parent / "tools" / "make_parameter_docs.py")
        self.assertEqual((SCRIPTS.parent / "docs" / "PARAMETERS.md").read_text(), tool.render(),
                         "docs/PARAMETERS.md is out of date: run python tools/make_parameter_docs.py")


class RegressionTests(unittest.TestCase):
    def test_residues_are_compared_case_insensitively(self):
        aln = common.Alignment(["a", "b"], ["acde", "ACDE"])
        self.assertEqual(aln.mean_consensus, 1.0)

    def test_failed_seed_alignment_does_not_cancel_the_expansion(self):
        # A one-member cluster cannot be aligned. The conservation test then cannot be applied,
        # but a good candidate must still be judged by its fit (and accepted).
        def fake_clustalo(in_fasta, out_aln, threads=1):
            records = list(common.read_fasta_records(in_fasta))
            if len(records) < 2:
                return False
            width = max(len(seq) for _, seq in records)
            common.write_fasta(out_aln, [(i, seq.ljust(width, "-")) for i, seq in records], line_width=10 ** 6)
            return True

        seq = "ACDEFGHIKLMNPQRSTVWY" * 6
        cfg = SimpleNamespace(rep_sample_size=50, expand_consensus_drop=0.65, expand_min_fit=0.15)
        seed_hits = pd.DataFrame({"Subject": ["S"], "Score": [100.0]})
        saved = merge.run_clustalo, common.run_clustalo
        merge.run_clustalo = common.run_clustalo = fake_clustalo
        try:
            accepted, counts = merge.filter_candidates_by_alignment(["C"], seed_hits, {"S": seq, "C": seq}, cfg)
        finally:
            merge.run_clustalo, common.run_clustalo = saved
        self.assertEqual(accepted, ["C"])
        self.assertEqual(counts["alignment_failed"], 0)


class FitSummaryTests(unittest.TestCase):
    def test_pooled_fraction_can_hide_a_bad_small_part(self):
        # 80 good members of cluster 1 and 20 members of cluster 2 that all fit badly:
        # the pooled bad fraction is only 20%, but cluster 2 on its own is 100% bad.
        fits = {f"a{i}": 0.8 for i in range(80)} | {f"b{i}": 0.05 for i in range(20)}
        summary = common.summarize_fits(fits, 0.22, groups={"1": [f"a{i}" for i in range(80)],
                                                              "2": [f"b{i}" for i in range(20)]})
        self.assertAlmostEqual(summary["bad_fraction"], 0.20)
        self.assertEqual(summary["bad_fraction_1"], 0.0)
        self.assertEqual(summary["bad_fraction_2"], 1.0)

    def test_shared_members_count_in_both_groups(self):
        summary = common.summarize_fits({"x": 0.1, "y": 0.9}, 0.22, groups={"1": ["x", "y"], "2": ["x"]})
        self.assertEqual((summary["bad_fraction_1"], summary["bad_fraction_2"]), (0.5, 1.0))


class CoreColumnTests(unittest.TestCase):
    def test_coverage_and_pre_merge_use_the_same_definition_of_core(self):
        # Column 4 is occupied by 9 of 20 reference sequences (45%): fewer than 60% gaps -> core.
        # Column 5 by 7 of 20 (35%): more than 60% gaps -> not core.
        refs = {}
        for i in range(20):
            seq = list("ACDE") + ["F" if i < 9 else "-"] + ["G" if i < 7 else "-"]
            refs[f"r{i}"] = "".join(seq)
        aln = common.Alignment(list(refs) + ["q"], list(refs.values()) + ["ACDE-G"])  # q misses column 4
        ids = list(refs)

        core = aln.core_columns([aln.index[r] for r in ids])
        self.assertEqual(core.tolist(), [True, True, True, True, True, False])

        # coverage: q has a residue in 4 of the 5 core columns
        self.assertAlmostEqual(aln.core_coverage("q", ids), 4 / 5)
        # pre-merge comparison: the same five columns
        _, _, n_core = aln.core_group_fits(ids[:10], ids[10:], gap_tolerance=0.6)
        self.assertEqual(n_core, 4)   # ...of which column 4 is not core in the half that has few residues there


class AddUnclusteredTests(unittest.TestCase):
    def test_score_ratio(self):
        self.assertIsNone(add_unclustered.score_ratio([("A", 50.0)]))
        self.assertAlmostEqual(add_unclustered.score_ratio([("A", 90.0), ("B", 60.0)]), 1.5)
        self.assertEqual(add_unclustered.score_ratio([("A", 90.0), ("B", 0.0)]), float("inf"))

    def test_score_window(self):
        member_scores = [100.0] * 6
        self.assertEqual(add_unclustered.in_score_window(75.0, member_scores, 0.3, 6), (True, 100.0))
        self.assertEqual(add_unclustered.in_score_window(60.0, member_scores, 0.3, 6), (False, 100.0))
        self.assertEqual(add_unclustered.in_score_window(130.0, member_scores, 0.3, 6), (True, 100.0))

    def test_small_clusters_skip_the_window(self):
        self.assertEqual(add_unclustered.in_score_window(1.0, [100.0] * 5, 0.3, 6), (True, None))


class DriverTests(unittest.TestCase):
    def test_schedule_keys_become_command_line_flags(self):
        sys.path.insert(0, str(SCRIPTS.parent))
        driver = load_script_from(SCRIPTS.parent / "run_pipeline.py")
        self.assertEqual(driver.flags({"score_window": 0.35, "min_shared": 2, "_note": "ignored"}),
                         ["--score-window", "0.35", "--min-shared", "2"])
        self.assertEqual(driver.flags({"a": 1}, "--b 2"), ["--a", "1", "--b", "2"])

    def test_shipped_schedule_only_uses_options_the_scripts_understand(self):
        import json
        driver = load_script_from(SCRIPTS.parent / "run_pipeline.py")
        schedule = json.loads((SCRIPTS.parent / "schedules" / "relaxing_3_iterations.json").read_text())
        required = {"03": ["--clusters", "x", "--hits", "x"],
                    "05": ["--clusters", "x", "--hits", "x", "--unclustered", "x"]}
        cases = [(it["merge"], merge, required["03"]) for it in schedule["iterations"]]
        cases.append((schedule["add_unclustered"], add_unclustered, required["05"]))

        for params, module, base_args in cases:
            saved = sys.argv
            sys.argv = ["script", *base_args, *driver.flags(params)]
            try:
                module.parse_args()   # exits with an error on any unknown option or bad value
            finally:
                sys.argv = saved


class ClusterFolderTests(unittest.TestCase):
    def test_missing_hit_tables_are_an_error_not_a_silent_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                common.load_hit_tables(tmp, universe={"a"})
            with self.assertRaises(SystemExit):
                common.load_hit_tables(Path(tmp) / "does_not_exist", universe={"a"})

    def test_upper_case_file_extensions_are_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "A.FASTA").write_text(">s1\nMKV\n")
            self.assertEqual(common.load_clusters(tmp)[0], {"A": ["s1"]})

    def test_load_and_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            (src / "A.fasta").write_text(">s1 x\nMKV\n>s2\nMKL\n")
            (src / "B.fa").write_text(">s2\nMKL\n>s3\nGGG\n")
            clusters, seq_db = common.load_clusters(src)
            self.assertEqual(clusters, {"A": ["s1", "s2"], "B": ["s2", "s3"]})
            self.assertEqual(seq_db, {"s1": "MKV", "s2": "MKL", "s3": "GGG"})

            common.prepare_output_dir(dst)
            self.assertEqual(common.write_clusters(dst, {"A": ["s1"], "B": []}, seq_db), 1)  # empty -> skipped
            self.assertEqual(list(common.read_fasta_records(dst / "A.fasta")), [("s1", "MKV")])

    def test_refuses_to_mix_with_old_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old.fasta").write_text(">x\nA\n")
            with self.assertRaises(SystemExit):
                common.prepare_output_dir(tmp)
            common.prepare_output_dir(tmp, overwrite=True)
            self.assertFalse((Path(tmp) / "old.fasta").exists())


if __name__ == "__main__":
    unittest.main()
