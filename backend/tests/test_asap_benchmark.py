"""Tests for the ASAP benchmark helpers.

These must pass on a clean clone. The dataset is licensed and gitignored, so
nothing here may read it -- every test builds its own inputs. The one test
that needs the real split file skips when it is absent.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_benchmark():
    """The script lives in scripts/, which isn't a package."""
    path = os.path.join(BACKEND_DIR, "scripts", "asap_benchmark.py")
    spec = importlib.util.spec_from_file_location("asap_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()


class TestPlaceholderCleaning:
    """ASAP redacts entities as @TYPE<n>. Leaving them in feeds the
    hand-crafted features junk -- '@NUM1' carries a digit, so digit_count
    would reward an essay for having been anonymised."""

    def test_replaces_each_placeholder_type_with_a_neutral_word(self):
        text, count = benchmark.clean_text(
            "@PERSON1 works at @ORGANIZATION2 in @LOCATION1 since @DATE3."
        )
        assert count == 4
        assert "@" not in text
        assert "someone" in text and "an organization" in text and "a place" in text

    def test_reports_how_many_were_replaced(self):
        _, count = benchmark.clean_text("@CAPS1 @CAPS2 @NUM1 plain words here")
        assert count == 3

    def test_no_placeholder_leaves_text_alone_and_counts_zero(self):
        original = "A perfectly ordinary sentence with no redactions."
        text, count = benchmark.clean_text(original)
        assert count == 0
        assert text == original

    def test_digits_from_placeholders_do_not_survive_into_features(self):
        """The whole point: an anonymised essay must not look numerate."""
        text, _ = benchmark.clean_text("In @DATE1 the @ORGANIZATION1 said @PERCENT1 of @NUM1 people.")
        assert not any(character.isdigit() for character in text)
        digit_count = benchmark.extract_features([text])[0][benchmark.FEATURE_NAMES.index("digit_count")]
        assert digit_count == 0

    def test_real_numbers_written_by_the_student_are_kept(self):
        text, _ = benchmark.clean_text("In 2019 about 45% of students agreed.")
        assert "2019" in text and "45%" in text

    def test_normalises_mojibake_and_smart_quotes(self):
        text, _ = benchmark.clean_text("it’s “quoted” — really")
        assert "’" not in text and "“" not in text
        assert "it's" in text and '"quoted"' in text

    def test_collapses_whitespace_and_space_before_punctuation(self):
        text, _ = benchmark.clean_text("Hello   world .  Next  sentence !")
        assert text == "Hello world. Next sentence!"


class TestQwkWithLabels:
    """QWK must be computed over the full 2-12 range. Without explicit labels
    a system that never predicts an extreme score is scored on a smaller
    grid, which quietly changes the number."""

    LABELS = list(range(2, 13))

    def test_perfect_agreement_is_one(self):
        y = [2, 5, 8, 12, 7]
        assert benchmark.qwk(y, y, self.LABELS) == pytest.approx(1.0)

    def test_constant_predictions_have_no_agreement_to_measure(self):
        # Expected disagreement is zero, so kappa is undefined rather than perfect.
        assert np.isnan(benchmark.qwk([4, 4, 4, 4], [4, 4, 4, 4], self.LABELS))

    def test_near_misses_cost_less_than_far_misses(self):
        # The truth has to vary: against a constant truth every kappa is 0,
        # because there is no spread for chance agreement to be measured against.
        truth = [4, 6, 8, 10, 12, 4, 6, 8]
        near = benchmark.qwk(truth, [5, 5, 9, 9, 11, 3, 7, 7], self.LABELS)
        far = benchmark.qwk(truth, [12, 12, 2, 2, 4, 12, 12, 2], self.LABELS)
        assert near > far
        assert near > 0.9 and far < 0

    def test_matches_sklearn_when_given_the_same_label_range(self):
        from sklearn.metrics import cohen_kappa_score
        rng = np.random.default_rng(0)
        truth = rng.integers(2, 13, size=200)
        pred = np.clip(truth + rng.integers(-2, 3, size=200), 2, 12)
        mine = benchmark.qwk(truth, pred, self.LABELS)
        theirs = cohen_kappa_score(truth, pred, weights="quadratic", labels=self.LABELS)
        assert mine == pytest.approx(theirs)

    def test_gaps_in_observed_scores_must_not_collapse(self):
        """Why the full label range has to be passed explicitly.

        If the scores actually seen are only {2, 3, 11, 12}, inferring the
        grid from the data makes 3 and 11 adjacent, so an eight-point
        disagreement is weighted like a one-point one. On the real 2-12 grid
        the same predictions are near-perfect; on the collapsed grid they look
        mediocre.
        """
        truth = [2, 3, 2, 3, 11, 12, 11, 12]
        pred = [3, 2, 3, 3, 12, 11, 12, 11]
        observed_only = sorted(set(truth) | set(pred))

        full = benchmark.qwk(truth, pred, self.LABELS)
        collapsed = benchmark.qwk(truth, pred, observed_only)
        assert full > 0.95
        assert collapsed < 0.7
        assert full - collapsed > 0.3

    def test_sklearn_without_labels_reproduces_the_collapsed_mistake(self):
        """Pinning the reason we never call it without labels."""
        from sklearn.metrics import cohen_kappa_score
        truth = [2, 3, 2, 3, 11, 12, 11, 12]
        pred = [3, 2, 3, 3, 12, 11, 12, 11]
        inferred = cohen_kappa_score(truth, pred, weights="quadratic")
        explicit = cohen_kappa_score(truth, pred, weights="quadratic", labels=self.LABELS)
        assert inferred < explicit
        assert benchmark.qwk(truth, pred, self.LABELS) == pytest.approx(explicit)

    def test_labels_outside_the_data_are_tolerated(self):
        # No essay scores 2 or 12 here; the label range still covers them.
        truth, pred = [7, 8, 9, 8], [8, 8, 9, 7]
        assert -1.0 <= benchmark.qwk(truth, pred, self.LABELS) <= 1.0


class TestScoreClamping:
    def test_rounds_and_clamps_into_the_scale(self):
        got = benchmark.to_scores([1.2, 2.4, 7.5, 12.9, 40.0, -5.0])
        assert got.min() >= benchmark.SCORE_MIN
        assert got.max() <= benchmark.SCORE_MAX
        assert list(got) == [2.0, 2.0, 8.0, 12.0, 12.0, 2.0]


class TestSplitDeterminism:
    """The split is committed before any model is fit, so it has to be
    reproducible from the seed alone -- otherwise the locked test set means
    nothing."""

    @staticmethod
    def _frame(n=400, seed=1, rare=8):
        """Mostly common scores plus a handful of rare ones, mirroring set 1
        (which has one score with a single essay)."""
        rng = np.random.default_rng(seed)
        scores = np.concatenate([rng.integers(6, 13, size=n - rare - 1),
                                 np.full(rare, 3), [4]])
        return pd.Series(range(1, n + 1)), pd.Series(scores)

    def test_same_seed_gives_identical_partitions(self):
        ids, scores = self._frame()
        first = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        second = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        assert first == second

    def test_different_seed_gives_a_different_split(self):
        ids, scores = self._frame()
        a = benchmark.make_split(ids, scores, seed=1)
        b = benchmark.make_split(ids, scores, seed=2)
        assert a != b

    def test_partitions_are_disjoint_and_complete(self):
        ids, scores = self._frame()
        split = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        train, dev, test = set(split["train"]), set(split["dev"]), set(split["test"])
        assert not (train & dev) and not (train & test) and not (dev & test)
        assert train | dev | test == set(ids)

    def test_roughly_seventy_fifteen_fifteen(self):
        ids, scores = self._frame(n=1000)
        split = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        assert len(split["train"]) == pytest.approx(700, abs=5)
        assert len(split["dev"]) == pytest.approx(150, abs=5)
        assert len(split["test"]) == pytest.approx(150, abs=5)

    def test_a_score_with_a_single_essay_does_not_break_stratification(self):
        """Set 1 really does contain a score with exactly one essay."""
        ids, scores = self._frame()
        assert (scores == 4).sum() == 1
        split = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        assert sum(len(v) for v in split.values()) == len(ids)

    def test_rare_bucket_too_small_to_split_falls_back_instead_of_crashing(self):
        """With only two rare essays the pooled bucket cannot survive both
        splits, so it is folded into the most common score."""
        ids, scores = self._frame(rare=1)  # one '3' and one '4' => 2 rare essays
        split = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        assert sum(len(v) for v in split.values()) == len(ids)

    def test_fingerprint_changes_if_an_id_moves(self):
        ids, scores = self._frame()
        split = benchmark.make_split(ids, scores, seed=benchmark.SPLIT_SEED)
        before = benchmark.split_fingerprint(split)
        moved = {**split, "test": split["test"][:-1], "dev": sorted(split["dev"] + split["test"][-1:])}
        assert benchmark.split_fingerprint(moved) != before

    def test_committed_split_file_matches_its_own_fingerprint(self):
        """Guards the committed file against silent edits. Skips when the file
        is absent so a clean clone without the dataset still passes."""
        if not os.path.exists(benchmark.SPLIT_PATH):
            pytest.skip("asap_split.json not present")
        with open(benchmark.SPLIT_PATH) as f:
            split = json.load(f)
        assert benchmark.split_fingerprint(split) == split["fingerprint"]


class TestFeaturisationMatchesProduction:
    """The benchmark loads embedder.py and features.py by path to avoid
    importing Flask and SQLAlchemy. That bypass must not let it drift from
    the function the live grader uses."""

    @pytest.mark.slow
    def test_build_matrix_is_identical_to_the_shipped_one(self):
        from app.grading.engine import build_matrix as production_build_matrix

        texts = ["A short answer.", "In 2019 the FTC fined Facebook $5 billion, which shows regulation works."]
        assert np.array_equal(benchmark.build_matrix(texts), production_build_matrix(texts))

    def test_feature_names_are_the_shipped_ones(self):
        from app.grading.features import FEATURE_NAMES as production_names

        assert benchmark.FEATURE_NAMES == production_names
