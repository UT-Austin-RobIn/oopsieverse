"""
LLM/VLM prediction of DamageSim parameters.

- ``ground_truth``: export the hand-tuned tables to ``data/ground_truth_params.csv``.
- ``schema``: the pydantic output types (the four mechanical numbers).
- ``models``: the model table and one validated Chat Completions call for every provider.
- ``prompts``: the selection and generation prompts.
- ``context``: the leave-one-out visible pool and the model-picked exemplars.
- ``generate``: the leave-one-out runner (``python -m damagesim.llm_params.generate``).
- ``score``: metrics, baselines and Gate 1 checks (``python -m damagesim.llm_params.score``).

Prediction modules in this package are simulator-free; only ``replay/`` and
``render/`` (added later) import a simulator and run inside a backend env.
"""
