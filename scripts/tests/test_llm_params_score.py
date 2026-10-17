"""
Sim-free tests for damagesim/llm_params/score.py (EVAL.md sections 6 and 7) on hand-computed
values and synthetic run directories. No network.

Run from the repo root, for example:
    uv run --no-project --python 3.10 --with pytest --with numpy --with 'scipy<1.15' --with pyyaml \
        --with 'openai>=1.40' --with 'pydantic>=2' --with tyro \
        python -m pytest scripts/tests/test_llm_params_score.py -q
(scipy 1.15 wheels fail to load on macOS 27 with Python 3.10, hence the pin; lab machines need none.)
"""

import csv
import json
import random

import numpy as np
import pytest

from damagesim.llm_params import ground_truth as gt
from damagesim.llm_params import score
from damagesim.llm_params.schema import PARAM_NAMES

A = np.array


# ── Metrics on hand-computed values ─────────────────────────────────────

def test_metric_values_by_hand():
    assert score.pairwise_accuracy(A([1, 1, 4]), A([1, 2, 3])) == pytest.approx(2.5 / 3)
    assert score.log10_mae(A([10, 100]), A([100, 100])) == pytest.approx(0.5)
    assert score.within(A([10, 100, 250]), A([100, 100, 100]), 3) == pytest.approx(2 / 3)
    assert score.within(A([10, 100, 250]), A([100, 100, 100]), 10) == 1.0
    assert score.mae(A([0.5, 1.0]), A([1.0, 1.0])) == pytest.approx(0.25)
    assert score.zero_agreement(A([0.001, 0.5, 0.0, 0.2]), A([0, 0, 0, 0.2])) == 0.75


def test_constant_prediction_has_no_ranking():
    p, t = A([5.0, 5.0, 5.0]), A([1.0, 2.0, 3.0])
    assert score.spearman(p, t) == 0.0 and score.kendall(p, t) == 0.0 and score.pairwise_accuracy(p, t) == 0.5


def test_consistency_rule():
    assert score.is_consistent([100, 103, 97, 104, 100], 5)
    assert not score.is_consistent([100, 106, 100, 100, 100], 5)
    assert score.is_consistent([0.001, 0.004, 0.0, 0.002, 0.001], 5)
    assert not score.is_consistent([0.001, 0.001, 0.001, 0.001, 0.02], 5)
    assert not score.is_consistent([100, 100, 100, 100], 5)


def test_closest_name_tie_is_alphabetical():
    pool = [{"display_name": "ac", "entry_key": "ac"}, {"display_name": "aa", "entry_key": "aa"}]
    assert score.closest_name("ab", pool)["display_name"] == "aa"


def test_human_check_draw_matches_eval_md():
    rows = gt.export("robocasa")
    keys = sorted(r["entry_key"] for r in rows if r["is_target"] and not r["lockbox"])
    assert score.human_check_keys(rows) == sorted(random.Random(0).sample(keys, 10))


# ── End to end on a synthetic run ───────────────────────────────────────

def _write_run(run_dir, backend, targets, samples):
    """samples: entry_key -> list of dicts (the four numbers) or None for a failed sample."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"config": {"backend": backend, "n": 5, "k": 2}, "model_spec": {}, "git_commit": None, "targets": targets}))
    with open(run_dir / "context.jsonl", "w") as f:
        for key in targets:
            f.write(json.dumps({"entry_key": key, "exemplars": [], "response_id": None, "error": None}) + "\n")
    with open(run_dir / "generations.jsonl", "w") as f:
        for key, ss in samples.items():
            for i, s in enumerate(ss, start=1):
                rec = {"entry_key": key, "sample": i, "error": None if s else "invalid",
                       **{p: (s[p] if s else None) for p in PARAM_NAMES}}
                f.write(json.dumps(rec) + "\n")
    (run_dir / "requests.jsonl").write_text("")


def test_perfect_run_with_one_failure(tmp_path):
    rows = {r["entry_key"]: r for r in gt.export("robocasa")}
    others = [k for k, r in rows.items() if r["is_target"] and not r["lockbox"] and k not in ("plate", "bowl")][:7]
    targets = others + ["plate"]
    truth = {k: {p: rows[k][p] for p in PARAM_NAMES} for k in targets}
    samples = {k: [truth[k]] * 5 for k in others}
    samples["plate"] = [truth["plate"], truth["plate"], None, None, None]
    run_dir = tmp_path / "run"
    _write_run(run_dir, "robocasa", targets, samples)

    out = score.score(run_dir, n_boot=200)
    assert (run_dir / "scores.csv").exists()
    cons = [r for r in out if r["method"] == "model" and r["metric"] == "consistency"]
    assert len(cons) == 4 and all(r["n_failed"] == 1 and r["value"] == pytest.approx(7 / 8) for r in cons)

    with open(run_dir / "per_target.csv", newline="") as f:
        plate = next(r for r in csv.DictReader(f) if r["entry_key"] == "plate")
    assert plate["failed"] == "true"
    assert plate["damage_threshold_median"] == plate["damage_threshold_baseline_median"]
    assert plate["accept"] == "" and plate["reason"] == ""

    assert score.score(run_dir, n_boot=200) == out
    lines = score.gate_summary(out, "robocasa")
    assert any(l.startswith("check 1: ") for l in lines)
    assert any(l.startswith("check 3: pending") for l in lines)
    assert score.main([str(run_dir), "--n-boot", "50"]) == 0
