"""Tests for the pure logic in scripts/asap_finetune.py.

Nothing here trains, loads a model or reads the dataset: the dataset is
licensed and gitignored, and a clean clone must still pass. What is tested is
the arithmetic that turns a model output into a rubric score, where the
script looks for its inputs, which device it insists on, and that the
experiment's inputs and design are the ones it declared in advance.
"""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The split as locked in f3c5d17, before anything was fit. Pinned here rather
# than read from the file, so regenerating the split -- even consistently,
# with a matching fingerprint -- fails loudly instead of passing.
LOCKED_FINGERPRINT = "828dbd347ea2cf81"
LOCKED_COUNTS = {"train": 1248, "dev": 267, "test": 268}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BACKEND_DIR, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finetune = _load("asap_finetune", "scripts/asap_finetune.py")


class TestTargetScaling:
    """The model is trained on scores squashed to 0-1 and its output is
    stretched back. Getting either direction wrong shifts every prediction."""

    def test_the_ends_of_the_scale_map_to_zero_and_one(self):
        assert list(finetune.to_unit([2, 12])) == [0.0, 1.0]
        assert finetune.to_unit([7])[0] == pytest.approx(0.5)

    def test_rescale_inverts_to_unit_for_every_score(self):
        scores = np.arange(2, 13)
        assert np.allclose(finetune.rescale(finetune.to_unit(scores)), scores)

    def test_from_unit_round_trips_every_whole_score(self):
        scores = np.arange(2, 13)
        assert list(finetune.from_unit(finetune.to_unit(scores))) == list(scores.astype(float))

    def test_from_unit_rounds_to_the_nearest_whole_score(self):
        # 0.44 -> 6.4 -> 6;  0.46 -> 6.6 -> 7
        assert list(finetune.from_unit([0.44, 0.46])) == [6.0, 7.0]

    def test_from_unit_clamps_outputs_outside_the_scale(self):
        # A regression head is unbounded; -0.3 and 1.4 are real possibilities.
        assert list(finetune.from_unit([-0.3, 1.4])) == [2.0, 12.0]

    def test_from_unit_only_ever_returns_whole_scores_in_range(self):
        raw = np.random.default_rng(0).normal(0.5, 0.6, size=500)
        scores = finetune.from_unit(raw)
        assert scores.min() >= 2 and scores.max() <= 12
        assert np.all(scores == np.round(scores))

    def test_rescale_does_not_round(self):
        """Candidate B blends continuous predictions; rounding here would
        make it round twice."""
        assert finetune.rescale([0.43])[0] == pytest.approx(6.3)

    def test_blending_rounds_once_not_twice(self):
        """Why rescale exists separately from from_unit. With a fine-tuned
        6.6 and a Ridge 6.2 at equal weight, the true blend is 6.4, which is
        a 6. Rounding the fine-tuned half first (6.6 -> 7) gives 6.6, a 7."""
        finetuned_unit, ridge = 0.46, 6.2  # rescale(0.46) == 6.6
        once = finetune.bench.to_scores(0.5 * finetune.rescale([finetuned_unit]) + 0.5 * ridge)
        twice = finetune.bench.to_scores(0.5 * finetune.from_unit([finetuned_unit]) + 0.5 * ridge)
        assert once[0] == 6.0
        assert twice[0] == 7.0

    def test_scale_matches_the_benchmark(self):
        """Every system in the comparison must use the same score range."""
        assert (finetune.SCORE_MIN, finetune.SCORE_MAX) == (finetune.bench.SCORE_MIN, finetune.bench.SCORE_MAX)


class TestDataDirLayout:
    """On Colab every input comes from one flat Google Drive folder; locally
    the repo layout is used. Both must resolve to the right files."""

    def test_data_dir_is_one_flat_folder(self):
        assert finetune.resolve_paths("/drive/asap") == {
            "essays": "/drive/asap/essays.xlsx",
            "split": "/drive/asap/asap_split.json",
            "published": "/drive/asap/asap.json",
            "cache": "/drive/asap/embedding_cache",
        }

    def test_without_data_dir_the_repo_layout_is_used(self):
        paths = finetune.resolve_paths(None)
        assert paths["essays"] == os.path.join(BACKEND_DIR, "data", "asap", "essays.xlsx")
        assert paths["split"] == os.path.join(BACKEND_DIR, "training_data", "asap_split.json")
        assert paths["published"] == os.path.join(BACKEND_DIR, "training_data", "results", "asap.json")
        assert paths["cache"] == os.path.join(BACKEND_DIR, "data", "asap", ".cache")

    def test_apply_paths_points_the_benchmark_loaders_at_them(self, monkeypatch):
        for name in ("DATA_PATH", "SPLIT_PATH", "CACHE_DIR"):
            monkeypatch.setattr(finetune.bench, name, getattr(finetune.bench, name))  # restored afterwards
        finetune.apply_paths(finetune.resolve_paths("/drive/asap"))
        assert finetune.bench.DATA_PATH == "/drive/asap/essays.xlsx"
        assert finetune.bench.SPLIT_PATH == "/drive/asap/asap_split.json"
        assert finetune.bench.CACHE_DIR == "/drive/asap/embedding_cache"

    def test_repo_default_survives_apply_paths(self, monkeypatch):
        for name in ("DATA_PATH", "SPLIT_PATH", "CACHE_DIR"):
            monkeypatch.setattr(finetune.bench, name, getattr(finetune.bench, name))
        finetune.apply_paths(finetune.resolve_paths("/drive/asap"))
        assert finetune.resolve_paths(None)["split"] == os.path.join(BACKEND_DIR, "training_data", "asap_split.json")


class TestEmbeddingCachePath:
    """The pre-flight looks for the benchmark's cached embeddings by name. If
    its idea of the name drifted from the benchmark's, a present cache would
    look missing, or a recomputed one would slip through."""

    def test_matches_the_file_the_benchmark_writes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(finetune.bench, "CACHE_DIR", str(tmp_path))
        texts = ["first essay", "second essay"]
        finetune.bench.cached_embeddings("single_train", texts, lambda: np.zeros((2, 3)))
        assert os.path.exists(finetune.expected_cache_path("single_train", texts))

    def test_different_texts_or_names_give_different_files(self):
        base = finetune.expected_cache_path("single_train", ["a"])
        assert finetune.expected_cache_path("single_train", ["b"]) != base
        assert finetune.expected_cache_path("single_test", ["a"]) != base


class TestDevicePolicy:
    """The full grid froze an 8 GB Apple M1, so it runs on CUDA or not at all
    unless explicitly overridden."""

    def test_prefers_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert finetune.pick_device().type == "cuda"

    def test_falls_back_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert finetune.pick_device().type in ("mps", "cpu")

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_full_run_refuses_without_cuda(self, device):
        with pytest.raises(SystemExit):
            finetune.require_gpu(torch.device(device))

    def test_full_run_proceeds_on_cuda(self):
        finetune.require_gpu(torch.device("cuda"))  # a device descriptor; needs no GPU to construct

    def test_light_runs_and_an_explicit_override_are_exempt(self):
        finetune.require_gpu(torch.device("mps"), light=True)
        finetune.require_gpu(torch.device("cpu"), allow_no_gpu=True)


class TestCommandLine:
    def test_takes_a_data_dir_and_an_out_dir(self):
        args = finetune.build_parser().parse_args(["--data-dir", "/d", "--out-dir", "/o"])
        assert (args.data_dir, args.out_dir) == ("/d", "/o")

    def test_defaults_to_the_repo_layout(self):
        args = finetune.build_parser().parse_args([])
        assert args.data_dir is None and args.out_dir is None

    def test_has_no_time_cap(self):
        """The grid runs to completion on a GPU; a cap would silently cut the
        pre-declared grid short."""
        options = {o for action in finetune.build_parser()._actions for o in action.option_strings}
        assert "--max-minutes" not in options


class TestLockedSplitUnchanged:
    """The fine-tuning run must use the exact split the benchmark locked
    before anything was fit, or its test numbers aren't comparable."""

    @staticmethod
    def _split():
        with open(finetune.resolve_paths(None)["split"]) as f:
            return json.load(f)

    def test_the_script_pins_the_locked_fingerprint(self):
        assert finetune.LOCKED_SPLIT_FINGERPRINT == LOCKED_FINGERPRINT

    def test_the_fine_tune_reads_the_benchmarks_split_file(self):
        benchmark = _load("asap_benchmark_for_split", "scripts/asap_benchmark.py")
        assert finetune.resolve_paths(None)["split"] == benchmark.SPLIT_PATH

    def test_fingerprint_is_the_one_locked_before_fitting(self):
        assert self._split()["fingerprint"] == LOCKED_FINGERPRINT

    def test_ids_still_match_the_locked_fingerprint(self):
        split = self._split()
        payload = json.dumps({k: split[k] for k in ("train", "dev", "test")}, sort_keys=True)
        assert hashlib.sha256(payload.encode()).hexdigest()[:16] == LOCKED_FINGERPRINT

    def test_partition_sizes_are_unchanged(self):
        split = self._split()
        assert {k: len(split[k]) for k in ("train", "dev", "test")} == LOCKED_COUNTS

    def test_partitions_do_not_overlap(self):
        split = self._split()
        train, dev, test = set(split["train"]), set(split["dev"]), set(split["test"])
        assert not (train & dev) and not (train & test) and not (dev & test)


class TestDesignMatchesThePreDeclaration:
    """The design was committed to asap_results.md before training. These
    pin the script to it, so the declared design and the run can't drift."""

    def test_model_and_length(self):
        assert finetune.MODEL_NAME == "distilbert-base-uncased"
        assert finetune.MAX_LENGTH == 512

    def test_grid(self):
        assert finetune.LEARNING_RATES == (2e-5, 3e-5)
        assert finetune.MAX_EPOCHS == 4

    def test_blend_weights(self):
        assert finetune.BLEND_WEIGHTS == (0.25, 0.5, 0.75)

    def test_seed_and_batch(self):
        assert finetune.SEED == 20260928
        assert finetune.BATCH_SIZE == 8

    def test_outputs_default_to_the_gitignored_experiment_dir(self):
        assert finetune.OUT_DIR == os.path.join(BACKEND_DIR, "ml_experiments")
        with open(os.path.join(BACKEND_DIR, "..", ".gitignore")) as f:
            assert "backend/ml_experiments/" in f.read()


class TestColabRequirements:
    """One pip cell on Colab. It must not touch torch -- a pip torch would
    replace Colab's CUDA build -- and must cover what the script imports."""

    @staticmethod
    def _packages():
        with open(os.path.join(BACKEND_DIR, "requirements-colab.txt")) as f:
            lines = [line.split("#")[0].strip() for line in f]
        return [line for line in lines if line]

    def test_does_not_install_torch(self):
        assert not any(p.lower().startswith("torch") for p in self._packages())

    @pytest.mark.parametrize("package", ["transformers", "scikit-learn", "pandas", "numpy", "openpyxl"])
    def test_lists_what_the_script_needs(self, package):
        assert any(p.lower().startswith(package) for p in self._packages())
