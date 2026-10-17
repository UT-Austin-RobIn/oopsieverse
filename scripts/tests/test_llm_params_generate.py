"""
Sim-free tests for the LLM generation side of damagesim/llm_params/ (schema, models, prompts,
context, generate). No network and no API key: every model call goes through FakeClient.

Run from the repo root, for example:
    uv run --no-project --python 3.10 --with pytest --with numpy --with 'scipy<1.15' --with pyyaml \
        --with 'openai>=1.40' --with 'pydantic>=2' --with tyro \
        python -m pytest scripts/tests/test_llm_params_*.py -q
(scipy 1.15 wheels fail to load on macOS 27 with Python 3.10, hence the pin; lab machines need none.)
"""

import ast
import copy
import dataclasses
import json
import re
from types import SimpleNamespace

import openai
import pytest
from pydantic import ValidationError

from damagesim.llm_params import ground_truth as gt
from damagesim.llm_params import context, generate, models, prompts, schema


def _row(rows, key):
    found = [r for r in rows if r["entry_key"] == key]
    assert len(found) == 1, key
    return found[0]


def _init_params(rel_path, class_name):
    tree = ast.parse((gt.REPO_ROOT / rel_path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return {a.arg for a in init.args.args + init.args.kwonlyargs} - {"self", "entity"}


# ── Output schema ───────────────────────────────────────────────────────

def test_param_names_are_mechanical_evaluator_kwargs():
    core = _init_params("damagesim/core/evaluators/mechanical.py", "MechanicalDamageEvaluator")
    assert set(schema.PARAM_NAMES) <= core
    assert list(schema.MechanicalParams.model_fields) == list(schema.PARAM_NAMES)


def test_threshold_must_be_positive_sensitivity_may_be_zero():
    ok = dict(damage_threshold=50, damage_scale=1, impact_damage_sensitivity=0, qs_damage_sensitivity=0)
    assert schema.MechanicalParams(**ok).impact_damage_sensitivity == 0
    for bad in ({"damage_threshold": 0}, {"damage_scale": -1}, {"qs_damage_sensitivity": -0.1}):
        with pytest.raises(ValidationError):
            schema.MechanicalParams(**{**ok, **bad})
    with pytest.raises(ValidationError):
        schema.MechanicalParams(**ok, color="red")


def test_selection_model_enum_and_count():
    Sel = schema.selection_model(["mug", "bowl", "wine glass"], k=2)
    assert Sel.__name__ == "Selection"
    assert Sel.model_json_schema()["properties"]["similar"]["items"]["enum"] == ["mug", "bowl", "wine glass"]
    assert Sel(similar=["mug", "bowl"]).similar == ["mug", "bowl"]
    for bad in (["mug"], ["mug", "mug"], ["mug", "plate"], ["mug", "bowl", "wine glass"]):
        with pytest.raises(ValidationError):
            Sel(similar=bad)
    assert schema.selection_model(["mug"], k=5)(similar=["mug"]).similar == ["mug"]


# ── Fake client and the validated call ──────────────────────────────────

class FakeClient:
    """SDK-shaped stand-in: respond(kwargs) returns reply text, None, or an Exception to raise."""

    def __init__(self, respond):
        self.respond, self.calls = respond, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        out = self.respond(kw)
        if isinstance(out, Exception):
            raise out
        msg = SimpleNamespace(content=out, refusal=None)
        return SimpleNamespace(id=f"resp{len(self.calls)}", choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


GOOD = json.dumps(dict(damage_threshold=50.0, damage_scale=1.0, impact_damage_sensitivity=1.0,
                       qs_damage_sensitivity=0.3, reasoning="glass-like"))
SOL = models.MODELS["gpt-6.1-sol"]
USER = [{"role": "user", "content": "x"}]


def test_model_table():
    assert set(models.MODELS) == {"gpt-6.1-sol", "gpt-6-luna", "muse-spark-1.3", "gemini-4-argon", "qwen3.8-27b"}
    assert models.MODELS["qwen3.8-27b"].base_url == "https://openrouter.ai/api/v1"
    assert models.MODELS["qwen3.8-27b"].key_env == "OPENROUTER_API_KEY"
    assert SOL.base_url is None and SOL.key_env == "OPENAI_API_KEY"
    with pytest.raises(SystemExit):
        models.get_spec("gpt-4")


def test_make_client_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        models.make_client(SOL)


def test_json_schema_is_strict_and_portable():
    s = models.json_schema_for(schema.Generation)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"]) == {*schema.PARAM_NAMES, "reasoning"}
    assert "exclusiveMinimum" not in json.dumps(s) and "title" not in json.dumps(s)


def test_ask_json_success_logs_one_attempt():
    log, fake = [], FakeClient(lambda kw: GOOD)
    out, summ = models.ask_json(fake, SOL, USER, schema.Generation, log.append)
    assert out.damage_threshold == 50.0 and summ["attempts"] == 1 and summ["error"] is None
    assert summ["input_tokens"] == 10 and len(log) == 1 and log[0]["reply"] == GOOD
    kw = fake.calls[0]
    assert kw["response_format"]["json_schema"]["strict"] is True and "temperature" not in kw


def test_ask_json_retries_with_error_then_gives_up():
    log, fake = [], FakeClient(lambda kw: '{"damage_threshold": -1}')
    out, summ = models.ask_json(fake, SOL, USER, schema.Generation, log.append)
    assert out is None and summ["attempts"] == 3 and len(fake.calls) == 3 and len(log) == 3
    assert "invalid" in fake.calls[1]["messages"][-1]["content"]
    assert all(r["error"] for r in log) and summ["input_tokens"] == 30


def test_ask_json_fences_strings_and_empty_reply():
    replies = iter([None, "```json\n" + GOOD.replace("50.0", '"50"') + "\n```"])
    out, summ = models.ask_json(FakeClient(lambda kw: next(replies)), SOL, USER,
                                schema.Generation, lambda r: None)
    assert out.damage_threshold == 50.0 and summ["attempts"] == 2


def test_ask_json_api_error_is_logged_not_raised():
    err = openai.APIConnectionError(request=None)   # openai 1.x uses httpx, 3.x httpx2; neither is needed here
    log = []
    out, summ = models.ask_json(FakeClient(lambda kw: err), SOL, USER, schema.Generation, log.append)
    assert out is None and summ["attempts"] == 1 and summ["error"] and log[0]["error"]


def test_json_object_mode_puts_schema_in_prompt():
    spec = dataclasses.replace(SOL, json_schema=False)
    fake = FakeClient(lambda kw: GOOD)
    models.ask_json(fake, spec, USER, schema.Generation, lambda r: None)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "damage_threshold" in fake.calls[0]["messages"][0]["content"]


# ── Black box: visible pool, exemplar picking, prompts ──────────────────

RS = gt.export("robocasa")


def _keys(pool):
    return {r["entry_key"] for r in pool}


def test_visible_pool_hides_group_lockbox_robots_and_other_backend():
    pool = context.visible_pool(RS, "plate")
    assert not {"plate", "bowl"} & _keys(pool)
    assert not {"banana", "jam", "milk"} & _keys(pool)
    assert not {"default", "fixture_default", "Panda", "PandaOmron"} & _keys(pool)
    assert {r["backend"] for r in pool} == {"robocasa"}
    assert len(pool) == 30 - 2
    with pytest.raises(ValueError):
        context.visible_pool(RS, "Panda")


def test_pick_similar_enum_is_pool_and_returns_entry_keys():
    pool = context.visible_pool(RS, "plate")
    names = [r["display_name"] for r in pool]
    fake = FakeClient(lambda kw: json.dumps({"similar": names[:5]}))
    keys, summ = context.pick_similar(fake, SOL, "robocasa", _row(RS, "plate"), pool, 5, lambda r: None)
    enum = fake.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["similar"]["items"]["enum"]
    assert enum == names and keys == [r["entry_key"] for r in pool[:5]] and summ["error"] is None


def test_pick_similar_retries_bad_count_then_fails():
    pool = context.visible_pool(RS, "plate")
    fake = FakeClient(lambda kw: json.dumps({"similar": [pool[0]["display_name"]] * 2}))
    keys, summ = context.pick_similar(fake, SOL, "robocasa", _row(RS, "plate"), pool, 5, lambda r: None)
    assert keys == [] and summ["error"] and len(fake.calls) == 3


def test_generation_prompt_lists_exemplars_and_target():
    ex = [_row(RS, "mug"), _row(RS, "wine")]
    text = json.dumps(prompts.generation_messages("robocasa", "plate", ex))
    assert "plate" in text and "mug: damage_threshold=" in text and "wine: damage_threshold=" in text


# ── Leave-one-out runner ────────────────────────────────────────────────

def _responder(kw):
    js = kw["response_format"]["json_schema"]
    if js["name"] == "Selection":
        return json.dumps({"similar": js["schema"]["properties"]["similar"]["items"]["enum"][:2]})
    return GOOD


def _cfg(tmp_path, **kw):
    base = dict(backend="robocasa", n=3, k=2, targets=("plate", "mug"), workers=1, run_id="t", runs_dir=tmp_path)
    return generate.Config(**{**base, **kw})


def _lines(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_run_writes_context_and_n_generations(tmp_path):
    fake = FakeClient(_responder)
    d = generate.run(_cfg(tmp_path), client=fake)
    assert sorted(r["entry_key"] for r in _lines(d / "context.jsonl")) == ["mug", "plate"]
    gens = _lines(d / "generations.jsonl")
    assert sorted((g["entry_key"], g["sample"]) for g in gens) == [(t, s) for t in ("mug", "plate") for s in (1, 2, 3)]
    assert all(g["damage_threshold"] == 50.0 and g["error"] is None for g in gens)
    assert len(fake.calls) == 2 + 6 == len(_lines(d / "requests.jsonl"))
    assert json.loads((d / "config.json").read_text())["targets"] == ["plate", "mug"]


def test_resume_skips_finished_work_and_truncated_tail(tmp_path):
    d = generate.run(_cfg(tmp_path), client=FakeClient(_responder))
    g = d / "generations.jsonl"
    g.write_text("\n".join(g.read_text().splitlines()[:-1]) + '\n{"entry_key": "mu')
    fake = FakeClient(_responder)
    generate.run(_cfg(tmp_path, workers=3), client=fake)
    assert len(fake.calls) == 1 and len(_lines(g)) == 6


def test_resume_with_different_config_stops(tmp_path):
    generate.run(_cfg(tmp_path), client=FakeClient(_responder))
    with pytest.raises(SystemExit, match="config differs"):
        generate.run(_cfg(tmp_path, k=3), client=FakeClient(_responder))


def test_lockbox_target_refused_without_flag(tmp_path):
    with pytest.raises(SystemExit, match="banana"):
        generate.run(_cfg(tmp_path, targets=("banana",)), client=FakeClient(_responder))


def test_failed_selection_gives_no_generations(tmp_path):
    def respond(kw):
        return '{"similar": []}' if kw["response_format"]["json_schema"]["name"] == "Selection" else GOOD
    d = generate.run(_cfg(tmp_path, targets=("plate",)), client=FakeClient(respond))
    assert _lines(d / "context.jsonl")[0]["error"] and not (d / "generations.jsonl").read_text().strip()


def test_prompts_never_name_hidden_objects(tmp_path):
    d = generate.run(_cfg(tmp_path, targets=("plate",)), client=FakeClient(_responder))
    hidden = ["bowl", "banana", "jam", "milk"]
    for rec in _lines(d / "requests.jsonl"):
        text = " ".join(m["content"] for m in rec["messages"])
        assert not [h for h in hidden if re.search(rf"\b{h}\b", text)]


# ── Review fixes: resume after API errors, resume on another day, malformed replies ──

def test_resume_retries_api_errors(tmp_path):
    down = FakeClient(lambda kw: openai.APIConnectionError(request=None))
    d = generate.run(_cfg(tmp_path), client=down)
    assert all(c["error"] for c in _lines(d / "context.jsonl"))
    fake = FakeClient(_responder)
    generate.run(_cfg(tmp_path), client=fake)
    assert len(fake.calls) == 2 + 6
    gens = [g for g in _lines(d / "generations.jsonl") if g["error"] is None]
    assert len(gens) == 6


def test_resume_default_run_id_by_name(tmp_path):
    d = generate.run(_cfg(tmp_path, run_id=None), client=FakeClient(_responder))
    assert json.loads((d / "config.json").read_text())["config"]["run_id"] == d.name
    fake = FakeClient(_responder)
    assert generate.run(_cfg(tmp_path, run_id=d.name), client=fake) == d
    assert fake.calls == []


def test_ask_json_malformed_200_reply_is_retried_not_raised():
    bad = SimpleNamespace(id="x", choices=None, usage=None)
    fake = FakeClient(lambda kw: GOOD)
    fake.chat.completions.create = lambda **kw: bad
    log = []
    out, summ = models.ask_json(fake, SOL, USER, schema.Generation, log.append)
    assert out is None and summ["attempts"] == 3 and len(log) == 3 and "malformed" in log[0]["error"]
