from __future__ import annotations
import json
from paradigm_event_retrieval.paradigm_clip.model_factory import list_pretrained_models

if __name__ == "__main__": print(json.dumps([{"model_name": name, "pretrained": tag} for name, tag in list_pretrained_models()], indent=2))
