from __future__ import annotations
from paradigm_event_retrieval.paradigm_clip.train import validate_training_config

if __name__ == "__main__":
    validate_training_config("configs/train_stage_b.yaml")
    raise SystemExit("Stage B is gated on a measured Stage A validation improvement and is not started automatically.")
