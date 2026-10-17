"""
The black box: what the model may see for one target (EVAL.md section 4), and the
model's own pick of the k most similar visible objects.
"""

from __future__ import annotations

from damagesim.llm_params import prompts
from damagesim.llm_params.models import ask_json
from damagesim.llm_params.schema import selection_model


def visible_pool(rows: list[dict], target_key: str) -> list[dict]:
    """Same-backend, non-lockbox targets outside the target's block group."""
    target = next((r for r in rows if r["entry_key"] == target_key and r["is_target"]), None)
    if target is None:
        raise ValueError(f"{target_key!r} is not a target row")
    return [r for r in rows
            if r["backend"] == target["backend"] and r["is_target"] and not r["lockbox"]
            and r["block_group"] != target["block_group"]]


def pick_similar(client, spec, backend: str, target: dict, pool: list[dict], k: int,
                 log) -> tuple[list[str], dict]:
    """The model's k most similar pool objects, as entry_keys in its order ([] on failure)."""
    names = [r["display_name"] for r in pool]
    by_name = {r["display_name"]: r["entry_key"] for r in pool}
    messages = prompts.selection_messages(backend, target["display_name"], names, k)
    parsed, summary = ask_json(client, spec, messages, selection_model(names, k), log)
    if parsed is None:
        return [], summary
    return [by_name[n] for n in parsed.similar], summary
