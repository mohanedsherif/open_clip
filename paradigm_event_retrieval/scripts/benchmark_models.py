from __future__ import annotations
"""Lists selectable checkpoints only; benchmark after user approves any download."""
from paradigm_event_retrieval.paradigm_clip.model_factory import list_pretrained_models

if __name__ == "__main__":
    for model, tag in list_pretrained_models(): print(f"{model}\t{tag}")
