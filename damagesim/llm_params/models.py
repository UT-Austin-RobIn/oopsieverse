"""
The models we call and the one function that calls them.

Every provider goes through the OpenAI SDK's Chat Completions endpoint, switched by
``base_url``. Replies are validated with pydantic; an invalid reply is retried with the
error fed back. Every attempt is passed to ``log`` so the run keeps the raw replies.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ModelSpec:
    name: str             # what --model takes
    model_id: str         # what the provider expects
    base_url: str | None  # None = OpenAI default
    key_env: str          # environment variable holding the API key
    json_schema: bool = True        # strict json_schema; False = json_object + schema in the prompt
    price_in: float | None = None   # USD per 1M input tokens; None = unknown
    price_out: float | None = None


MODELS: dict[str, ModelSpec] = {s.name: s for s in (
    ModelSpec("gpt-6.1-sol", "gpt-6.1-sol", None, "OPENAI_API_KEY"),
    ModelSpec("gpt-6-luna", "gpt-6-luna", None, "OPENAI_API_KEY"),
    ModelSpec("muse-spark-1.3", "muse-spark-1.3",
              "https://api.meta.ai/v1",  # unverified
              "META_API_KEY"),
    ModelSpec("gemini-4-argon", "gemini-4-argon",
              "https://generativelanguage.googleapis.com/v1beta/openai/",  # unverified for this model
              "GEMINI_API_KEY"),
    ModelSpec("qwen3.8-27b",
              "qwen/qwen3.8-27b",  # unverified OpenRouter id
              "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
)}

MAX_ATTEMPTS = 3
_DROP_KEYS = {"title", "default", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def get_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]


def make_client(spec: ModelSpec):
    import openai

    key = os.environ.get(spec.key_env)
    if not key:
        raise SystemExit(f"set {spec.key_env} to call {spec.name}")
    return openai.OpenAI(base_url=spec.base_url, api_key=key, max_retries=5)


def json_schema_for(model_cls: type[BaseModel]) -> dict:
    """The model's JSON schema in the strict subset every provider accepts.

    Numeric bounds are dropped (pydantic still enforces them on the reply).
    """
    def walk(node):
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in _DROP_KEYS}
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(model_cls.model_json_schema())


def response_format(spec: ModelSpec, model_cls: type[BaseModel]) -> dict:
    if not spec.json_schema:
        return {"type": "json_object"}
    return {"type": "json_schema",
            "json_schema": {"name": model_cls.__name__, "strict": True, "schema": json_schema_for(model_cls)}}


def _strip_fence(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text


def ask_json(client, spec: ModelSpec, messages: list[dict], model_cls: type[BaseModel],
             log: Callable[[dict], None], extra: dict | None = None) -> tuple[BaseModel | None, dict]:
    """One validated call: up to MAX_ATTEMPTS, each logged. Returns (parsed or None, summary)."""
    import openai

    msgs = list(messages)
    if not spec.json_schema:
        msgs.insert(0, {"role": "system", "content": "Reply with one JSON object matching this JSON schema:\n"
                                                     + json.dumps(json_schema_for(model_cls))})
    summary = {"response_id": None, "attempts": 0, "input_tokens": 0, "output_tokens": 0, "error": None,
               "api_error": False}  # api_error: the provider refused the call; a resume retries it
    fmt = response_format(spec, model_cls)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        summary["attempts"] = attempt
        record = {"attempt": attempt, "model_id": spec.model_id, "messages": list(msgs), "reply": None,
                  "response_id": None, "usage": {"input_tokens": 0, "output_tokens": 0}, "error": None}
        try:
            resp = client.chat.completions.create(model=spec.model_id, messages=msgs,
                                                  response_format=fmt, **(extra or {}))
        except openai.APIError as exc:
            record["error"] = summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["api_error"] = True
            log(record)
            return None, summary

        try:  # some compatible endpoints answer 200 with no choices (an upstream error body)
            text, malformed = resp.choices[0].message.content, False
        except (TypeError, IndexError, AttributeError):
            text, malformed = None, True
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        summary["input_tokens"] += tin
        summary["output_tokens"] += tout
        summary["response_id"] = record["response_id"] = getattr(resp, "id", None)
        record.update(reply=text, usage={"input_tokens": tin, "output_tokens": tout})

        try:
            if malformed:
                raise ValueError("malformed reply: no choices")
            if text is None:
                raise ValueError("empty reply")
            parsed = model_cls.model_validate_json(_strip_fence(text))
        except (ValidationError, ValueError) as exc:
            record["error"] = summary["error"] = str(exc)
            log(record)
            msgs = msgs + [
                {"role": "assistant", "content": text or ""},
                {"role": "user", "content": f"That answer was invalid: {exc}. Reply again with JSON matching the schema."},
            ]
            continue
        summary["error"] = None
        log(record)
        return parsed, summary

    return None, summary
