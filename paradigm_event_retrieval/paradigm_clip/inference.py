from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .model_factory import require_open_clip


class ParadigmEmbeddingModel:
    """One-checkpoint embedding service for offline retrieval and evaluation."""
    def __init__(self, model_name: str, pretrained: str | None, checkpoint: str | None = None, device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.model_name, self.pretrained, self.checkpoint = model_name, pretrained, checkpoint
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        open_clip = require_open_clip()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("state_dict", state))
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @property
    def metadata(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "pretrained": self.pretrained, "checkpoint": self.checkpoint,
                "device": str(self.device), "embedding_dimension": getattr(self.model, "embed_dim", None)}

    def _autocast(self):
        return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16) if self.device.type == "cuda" and self.torch.cuda.is_bf16_supported() else __import__("contextlib").nullcontext()

    def embed_images(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        from PIL import Image
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = []
                for raw_path in paths[start:start + batch_size]:
                    path = Path(raw_path)
                    if not path.is_file(): raise FileNotFoundError(f"Image does not exist: {path}")
                    try:
                        with Image.open(path) as image: images.append(self.preprocess(image.convert("RGB")))
                    except OSError as exc: raise ValueError(f"Cannot decode image {path}: {exc}") from exc
                with self._autocast(): features = self.model.encode_image(self.torch.stack(images).to(self.device))
                batches.append(self.torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, int(self.model.embed_dim)), dtype=np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                with self._autocast(): features = self.model.encode_text(self.tokenizer(texts[start:start + batch_size]).to(self.device))
                batches.append(self.torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, int(self.model.embed_dim)), dtype=np.float32)

    @staticmethod
    def similarity(text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
        if text_embeddings.ndim != 2 or image_embeddings.ndim != 2 or text_embeddings.shape[1] != image_embeddings.shape[1]:
            raise ValueError("Embeddings must be 2-D and have the same dimension")
        return text_embeddings @ image_embeddings.T

    def retrieve(self, query: str, image_embeddings: np.ndarray, top_k: int = 10) -> list[dict[str, float | int]]:
        scores = self.similarity(self.embed_texts([query]), image_embeddings)[0]
        return [{"index": int(i), "score": float(scores[i])} for i in np.argsort(-scores)[:top_k]]
