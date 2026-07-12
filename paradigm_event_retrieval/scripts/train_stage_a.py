from __future__ import annotations
from paradigm_event_retrieval.paradigm_clip.train import validate_training_config

if __name__ == "__main__":
    config = validate_training_config("configs/train_stage_a.yaml")
    raise SystemExit("Stage A configuration is validated. Training is intentionally not started in the first iteration; review model/checkpoint download and use the current OpenCLIP CLI.")
