"""
Score a generation run against the hand-tuned tables (EVAL.md sections 6 and 7).

    python -m damagesim.llm_params.score runs/<run_id> [--n-boot 2000] [--seed 0]

Writes scores.csv and per_target.csv into the run directory and prints the Gate 1 checks.
Where this file and EVAL.md differ, EVAL.md wins.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from damagesim.llm_params import ground_truth as gt
from damagesim.llm_params.context import visible_pool
from damagesim.llm_params.schema import PARAM_NAMES

# Values from EVAL.md sections 6-7; the ones marked (proposed) there wait for Arpit.
CONSISTENCY_MARGIN = 0.05  # (proposed) relative, around the median of the samples
ZERO_EPS = 0.005           # a prediction below this counts as zero
CONSISTENCY_MIN = 0.8      # (proposed) Gate 1 check 1
SPEARMAN_MIN = 0.5         # (proposed) Gate 1 check 2
HUMAN_N = 10               # Gate 1 check 3
N_BOOT = 2000
SEED = 0
MIN_VALID = 3              # fewer valid samples than this (of n) = failed target

LOG_PARAMS = ("damage_threshold", "damage_scale")
METHODS = ("model", "baseline_median", "baseline_closest_name")
SCORE_COLUMNS = ["backend", "method", "param", "metric", "value", "ci_low", "ci_high", "n_targets", "n_failed"]


# ── Metrics ─────────────────────────────────────────────────────────────

def _corr(fn, pred, true) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fn(pred, true)[0]
    return 0.0 if np.isnan(r) else float(r)


def spearman(pred, true) -> float:
    return _corr(stats.spearmanr, pred, true)


def kendall(pred, true) -> float:
    return _corr(lambda p, t: stats.kendalltau(p, t, variant="b"), pred, true)


def pairwise_accuracy(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    dp, dt = pred[:, None] - pred[None, :], true[:, None] - true[None, :]
    mask = np.triu(np.ones_like(dt, dtype=bool), k=1) & (dt != 0)
    if not mask.any():
        return float("nan")
    s = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
    return float(s[mask].mean())


def log10_mae(pred, true) -> float:
    return float(np.mean(np.abs(np.log10(pred) - np.log10(true))))


def within(pred, true, factor: float) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.maximum(pred / true, true / pred) <= factor))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def zero_agreement(pred, true) -> float:
    return float(np.mean((np.asarray(pred) < ZERO_EPS) == (np.asarray(true) == 0)))


def is_consistent(samples: list[float], n: int) -> bool:
    if len(samples) < n:
        return False
    s = np.asarray(samples, float)
    med = float(np.median(s))
    if med < ZERO_EPS:
        return bool(np.all(s < ZERO_EPS))
    return bool(np.max(np.abs(s / med - 1)) <= CONSISTENCY_MARGIN)


def _metrics(backend: str, param: str) -> dict:
    out = {"spearman": spearman, "kendall": kendall, "pairwise": pairwise_accuracy}
    if param in LOG_PARAMS:
        out.update(log10_mae=log10_mae, within_3x=lambda p, t: within(p, t, 3), within_10x=lambda p, t: within(p, t, 10))
    else:
        out["mae"] = mae
        if backend == "robocasa":
            out["zero_agreement"] = zero_agreement
    return out


# ── Baselines and the human-check draw ──────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def closest_name(target_name: str, pool: list[dict]) -> dict:
    """Smallest normalized edit distance on display_name; ties alphabetically (EVAL.md section 7)."""
    def key(r):
        name = r["display_name"]
        return levenshtein(target_name, name) / max(len(target_name), len(name), 1), name
    return min(pool, key=key)


def median_baseline(pool: list[dict]) -> dict[str, float]:
    return {p: float(np.median([r[p] for r in pool])) for p in PARAM_NAMES}


def human_check_keys(rows: list[dict]) -> list[str]:
    keys = sorted(r["entry_key"] for r in rows if r["is_target"] and not r["lockbox"])
    return sorted(random.Random(0).sample(keys, HUMAN_N))


def bootstrap_idx(n_targets: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_targets, (n_boot, n_targets))


# ── Scoring a run ───────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    """Records in ``path``, ignoring a truncated last line; never rewrites the file."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}:{i + 1}: not JSON") from None
    return out


def _fmt(v) -> str:
    return gt.format_cell(float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)


def score(run_dir: Path, n_boot: int = N_BOOT, seed: int = SEED) -> list[dict]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    backend, n = meta["config"]["backend"], meta["config"]["n"]
    targets = meta["targets"]
    rows = gt.export(backend)
    by_key = {r["entry_key"]: r for r in rows}
    contexts = {r["entry_key"]: r for r in _read_jsonl(run_dir / "context.jsonl")}
    gens: dict[str, dict[int, dict]] = {}
    for g in _read_jsonl(run_dir / "generations.jsonl"):
        gens.setdefault(g["entry_key"], {})[g["sample"]] = g
    human = set(human_check_keys(rows))

    per_target = []
    for key in targets:
        target, pool = by_key[key], visible_pool(rows, key)
        ctx = contexts.get(key)
        valid = [g for g in gens.get(key, {}).values()
                 if g.get("error") is None and all(g.get(p) is not None for p in PARAM_NAMES)]
        failed = ctx is None or bool(ctx.get("error")) or len(valid) < MIN_VALID
        mb, cn = median_baseline(pool), closest_name(target["display_name"], pool)
        rec = {"entry_key": key, "display_name": target["display_name"], "in_human_check": key in human,
               "failed": failed, "exemplars": ";".join((ctx or {}).get("exemplars") or []),
               "closest_name": cn["entry_key"]}
        for p in PARAM_NAMES:
            samples = [float(g[p]) for g in valid]
            rec[f"{p}_true"] = float(target[p])
            rec[f"{p}_samples"] = ";".join(repr(s) for s in samples)
            rec[f"{p}_median"] = mb[p] if failed else float(np.median(samples))
            rec[f"{p}_consistent"] = (not failed) and is_consistent(samples, n)
            rec[f"{p}_baseline_median"] = mb[p]
            rec[f"{p}_baseline_closest_name"] = float(cn[p])
        rec["accept"] = rec["reason"] = None
        per_target.append(rec)

    n_targets, n_failed = len(per_target), sum(r["failed"] for r in per_target)
    idx = bootstrap_idx(n_targets, n_boot, seed)
    out = []

    def add(method, param, metric, fn, *arrays):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = [fn(*(a[i] for a in arrays)) for i in idx]
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out.append({"backend": backend, "method": method, "param": param, "metric": metric,
                    "value": fn(*arrays), "ci_low": float(lo), "ci_high": float(hi),
                    "n_targets": n_targets, "n_failed": n_failed})

    col = lambda name: np.array([r[name] for r in per_target], dtype=float)
    for p in PARAM_NAMES:
        true = col(f"{p}_true")
        preds = {"model": col(f"{p}_median"), "baseline_median": col(f"{p}_baseline_median"),
                 "baseline_closest_name": col(f"{p}_baseline_closest_name")}
        for metric, fn in _metrics(backend, p).items():
            for method in METHODS:
                add(method, p, metric, fn, preds[method], true)
        add("model", p, "consistency", lambda c: float(np.mean(c)), col(f"{p}_consistent"))
        if p == "damage_threshold":
            for metric in ("log10_mae", "within_10x"):
                fn = _metrics(backend, p)[metric]
                for b in ("baseline_median", "baseline_closest_name"):
                    add(f"model_minus_{b}", p, metric, lambda m, bb, t, fn=fn: fn(m, t) - fn(bb, t),
                        preds["model"], preds[b], true)

    with open(run_dir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows({c: _fmt(r[c]) for c in SCORE_COLUMNS} for r in out)
    with open(run_dir / "per_target.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_target[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({c: _fmt(v) for c, v in r.items()} for r in per_target)
    return out


# ── Gate 1 ──────────────────────────────────────────────────────────────

def _get(rows, method, param, metric):
    return next(r for r in rows if r["method"] == method and r["param"] == param and r["metric"] == metric)


def gate_summary(score_rows: list[dict], backend: str) -> list[str]:
    rows = [r for r in score_rows if r["backend"] == backend]
    cons = {p: _get(rows, "model", p, "consistency")["value"] for p in PARAM_NAMES}
    ok1 = all(v >= CONSISTENCY_MIN for v in cons.values())
    lines = [f"check 1: {'PASS' if ok1 else 'FAIL'} consistency >= {CONSISTENCY_MIN}: "
             + ", ".join(f"{p} {v:.2f}" for p, v in cons.items())]

    p = "damage_threshold"
    parts, ok2 = [], True
    for b in ("baseline_median", "baseline_closest_name"):
        d_mae, d_w = _get(rows, f"model_minus_{b}", p, "log10_mae"), _get(rows, f"model_minus_{b}", p, "within_10x")
        ok2 &= d_mae["ci_high"] < 0 and d_w["ci_low"] > 0
        parts.append(f"vs {b}: log10-MAE diff {d_mae['value']:+.3f} [{d_mae['ci_low']:+.3f}, {d_mae['ci_high']:+.3f}], "
                     f"within-10x diff {d_w['value']:+.3f} [{d_w['ci_low']:+.3f}, {d_w['ci_high']:+.3f}]")
    rho = _get(rows, "model", p, "spearman")["value"]
    ok2 &= rho >= SPEARMAN_MIN
    lines.append(f"check 2: {'PASS' if ok2 else 'FAIL'} threshold Spearman {rho:.2f} (>= {SPEARMAN_MIN}); " + "; ".join(parts))
    lines.append("check 3: pending: fill accept/reason in per_target.csv (rows with in_human_check=true)")
    if backend != "robocasa":
        lines.append("note: BEHAVIOR-1K is reported but does not decide Gate 1 (EVAL.md section 7)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rows = score(args.run_dir, args.n_boot, args.seed)
    print(f"{'method':<36} {'metric':<12} {'value':>8}  95% CI")
    for r in rows:
        if r["param"] == "damage_threshold":
            print(f"{r['method']:<36} {r['metric']:<12} {r['value']:>8.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print(f"targets {rows[0]['n_targets']}, failed {rows[0]['n_failed']}; all rows in {args.run_dir / 'scores.csv'}")
    for line in gate_summary(rows, rows[0]["backend"]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
