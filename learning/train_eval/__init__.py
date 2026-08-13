from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learning.train_eval.cfm_trainer import CFMTrainer

__all__ = ["CFMTrainer"]


def __getattr__(name: str):
    if name == "CFMTrainer":
        from learning.train_eval.cfm_trainer import CFMTrainer

        return CFMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
