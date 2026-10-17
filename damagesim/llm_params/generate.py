"""
Leave-one-out generation: for each target, the model picks k similar visible objects once,
then gives the four mechanical numbers n times from their hand-tuned values.

    python -m damagesim.llm_params.generate --model gpt-6.1-sol --backend robocasa

Writes runs/<run_id>/: config.json, context.jsonl, generations.jsonl, requests.jsonl.
Rerunning the same command resumes: finished work is skipped.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from damagesim.llm_params import ground_truth as gt
from damagesim.llm_params.context import pick_similar, visible_pool
from damagesim.llm_params.models import ask_json, get_spec, make_client
from damagesim.llm_params.prompts import generation_messages
from damagesim.llm_params.schema import PARAM_NAMES, Generation

FILES = ("context.jsonl", "generations.jsonl", "requests.jsonl")
_NOT_COMPARED = {"workers", "run_id", "runs_dir"}  # may differ when resuming


@dataclass
class Config:
    model: str = "gpt-6.1-sol"
    backend: Literal["robocasa", "omnigibson"] = "robocasa"
    n: int = 5                     # generations per target
    k: int = 5                     # exemplars per target
    targets: tuple[str, ...] = ()  # empty = every non-lockbox target
    lockbox: bool = False          # allow naming lockbox targets (EVAL.md section 5)
    workers: int = 4
    reasoning_effort: str | None = None
    run_id: str | None = None      # default: <YYYY-MM-DD>_<model>_<backend>_k<k>_n<n>
    runs_dir: Path = Path("runs")


def _config_dict(cfg: Config) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=gt.REPO_ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def _resolve_targets(rows: list[dict], cfg: Config) -> list[str]:
    targets = {r["entry_key"]: r for r in rows if r["is_target"]}
    if not cfg.targets:
        return [k for k, r in targets.items() if not r["lockbox"]]
    for key in cfg.targets:
        if key not in targets:
            raise SystemExit(f"{key!r} is not a {cfg.backend} target")
        if targets[key]["lockbox"] and not cfg.lockbox:
            raise SystemExit(f"{key!r} is in the lockbox; pass --lockbox to generate it (EVAL.md section 5)")
    return list(cfg.targets)


def _read_jsonl(path: Path) -> list[dict]:
    """Records in ``path``; a truncated last line (killed mid-write) is dropped from the file."""
    lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}:{i + 1}: not JSON") from None
            path.write_text("".join(l + "\n" for l in lines[:-1]))
    return out


def run(cfg: Config, client=None) -> Path:
    spec = get_spec(cfg.model)
    rows = gt.export(cfg.backend)
    by_key = {r["entry_key"]: r for r in rows}
    targets = _resolve_targets(rows, cfg)
    run_id = cfg.run_id or f"{date.today():%Y-%m-%d}_{cfg.model}_{cfg.backend}_k{cfg.k}_n{cfg.n}"
    run_dir = Path(cfg.runs_dir) / run_id

    print(f"run directory: {run_dir}")

    conf = {**_config_dict(cfg), "run_id": run_id}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())["config"]
        diff = sorted(k for k in set(old) | set(conf) if k not in _NOT_COMPARED and old.get(k) != conf.get(k))
        if diff:
            raise SystemExit(f"config differs from {cfg_path}: {', '.join(diff)}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps({"config": conf, "model_spec": dataclasses.asdict(spec),
                                        "git_commit": _git_commit(), "targets": targets}, indent=2) + "\n")
    for name in FILES:
        (run_dir / name).touch()

    # Calls the provider refused (bad key, outage, rate limit) are redone; answers that failed
    # validation 3 times are results and stay.
    contexts = {r["entry_key"]: r for r in _read_jsonl(run_dir / "context.jsonl") if not r.get("api_error")}
    done = {(g["entry_key"], g["sample"]) for g in _read_jsonl(run_dir / "generations.jsonl")
            if not g.get("api_error")}
    _read_jsonl(run_dir / "requests.jsonl")
    if client is None:
        client = make_client(spec)
    extra = {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else None

    lock = threading.Lock()
    stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "failed_selections": 0, "failed_generations": 0}

    def write(name: str, record: dict) -> None:
        with lock, open(run_dir / name, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def logger(key: str, stage: str, sample: int | None):
        def log(record: dict) -> None:
            with lock:
                stats["calls"] += 1
                stats["input_tokens"] += record["usage"]["input_tokens"]
                stats["output_tokens"] += record["usage"]["output_tokens"]
            write("requests.jsonl", {"entry_key": key, "stage": stage, "sample": sample, **record})
        return log

    def do_target(key: str) -> None:
        target = by_key[key]
        ctx = contexts.get(key)
        if ctx is None:
            keys, summ = pick_similar(client, spec, cfg.backend, target, visible_pool(rows, key), cfg.k,
                                      logger(key, "select", None))
            ctx = {"entry_key": key, "exemplars": keys, "response_id": summ["response_id"], "error": summ["error"],
                   "api_error": summ["api_error"]}
            write("context.jsonl", ctx)
        if ctx["error"]:
            with lock:
                stats["failed_selections"] += 1
            return
        messages = generation_messages(cfg.backend, target["display_name"], [by_key[k] for k in ctx["exemplars"]])
        for sample in range(1, cfg.n + 1):
            if (key, sample) in done:
                continue
            parsed, summ = ask_json(client, spec, messages, Generation, logger(key, "generate", sample), extra)
            numbers = {p: getattr(parsed, p) if parsed else None for p in PARAM_NAMES}
            write("generations.jsonl", {
                "entry_key": key, "sample": sample, **numbers,
                "reasoning": parsed.reasoning if parsed else None,
                "response_id": summ["response_id"], "attempts": summ["attempts"],
                "input_tokens": summ["input_tokens"], "output_tokens": summ["output_tokens"], "error": summ["error"],
                "api_error": summ["api_error"],
            })
            if parsed is None:
                with lock:
                    stats["failed_generations"] += 1

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        list(pool.map(do_target, targets))

    cost = ""
    if spec.price_in is not None and spec.price_out is not None:
        usd = (stats["input_tokens"] * spec.price_in + stats["output_tokens"] * spec.price_out) / 1e6
        cost = f", USD {usd:.4f}"
    print(f"{run_dir}: {stats['calls']} calls, {stats['failed_selections']} failed selections, "
          f"{stats['failed_generations']} failed generations, tokens in {stats['input_tokens']} "
          f"out {stats['output_tokens']}{cost}")
    return run_dir


def main() -> None:
    import tyro

    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
