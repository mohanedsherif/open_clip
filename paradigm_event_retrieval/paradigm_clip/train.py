from __future__ import annotations
from pathlib import Path
from .config import load_yaml


def validate_training_config(path: str | Path) -> dict:
    config = load_yaml(path)
    if config.get("full_model_finetuning"):
        raise ValueError("Full-model fine-tuning is deliberately disabled by default; explicitly review and override this safety check.")
    if config.get("stage") == "b" and not config.get("requires_stage_a_improvement"):
        raise ValueError("Stage B must require demonstrated Stage A validation improvement.")
    return config
