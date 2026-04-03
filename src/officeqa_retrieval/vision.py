from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import batched

DEFAULT_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for multimodal retrieval.") from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for multimodal retrieval.") from exc
    return AutoModel, AutoProcessor


def resolve_device(preferred_device: str | None = None) -> str:
    if preferred_device:
        return preferred_device
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisionTextEncoder:
    def __init__(
        self,
        model_key: str,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        if model_key not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported model key: {model_key}")

        torch = _import_torch()
        auto_model, auto_processor = _import_transformers()

        self.model_key = model_key
        self.model_name = model_name or DEFAULT_MODELS[model_key]
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.torch = torch
        self.processor = self._load_pretrained(auto_processor, self.model_name, use_fast=False)
        self.model = self._load_pretrained(auto_model, self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_pretrained(loader, model_name: str, **kwargs):
        try:
            return loader.from_pretrained(model_name, **kwargs)
        except Exception as first_error:
            try:
                return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
            except Exception:
                raise first_error

    def _normalize(self, tensor):
        return self.torch.nn.functional.normalize(tensor, dim=-1)

    def _ensure_feature_tensor(self, maybe_tensor, preferred_embed_attr: str):
        if hasattr(maybe_tensor, "shape") and hasattr(maybe_tensor, "dtype"):
            return maybe_tensor
        return self._extract_feature_tensor(maybe_tensor, preferred_embed_attr=preferred_embed_attr)

    def _extract_feature_tensor(self, outputs, preferred_embed_attr: str):
        if hasattr(outputs, preferred_embed_attr):
            tensor = getattr(outputs, preferred_embed_attr)
            if tensor is not None:
                return tensor
        if hasattr(outputs, "pooler_output"):
            tensor = outputs.pooler_output
            if tensor is not None:
                return tensor
        if hasattr(outputs, "last_hidden_state"):
            tensor = outputs.last_hidden_state
            if tensor is not None:
                return tensor.mean(dim=1)
        raise ValueError(f"Could not extract features from model outputs using attribute {preferred_embed_attr}.")

    def _text_features(self, texts: Sequence[str]):
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            if hasattr(self.model, "get_text_features") and callable(getattr(self.model, "get_text_features")):
                features = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = self._extract_feature_tensor(outputs, preferred_embed_attr="text_embeds")
        features = self._ensure_feature_tensor(features, preferred_embed_attr="text_embeds")
        return self._normalize(features)

    def _load_images(self, image_paths: Iterable[Path]):
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _image_features(self, image_paths: list[Path]):
        images = self._load_images(image_paths)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features") and callable(getattr(self.model, "get_image_features")):
                features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = self._extract_feature_tensor(outputs, preferred_embed_attr="image_embeds")
        features = self._ensure_feature_tensor(features, preferred_embed_attr="image_embeds")
        return self._normalize(features)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        batches = []
        for text_batch in batched(list(texts), self.batch_size):
            batches.append(self._text_features(list(text_batch)).detach().cpu().numpy().astype("float32"))
        if not batches:
            raise ValueError("texts cannot be empty")
        return np.concatenate(batches, axis=0)

    def embed_image_paths(self, image_paths: Sequence[Path]) -> np.ndarray:
        batches = []
        for image_batch in batched(list(image_paths), self.batch_size):
            batches.append(self._image_features(list(image_batch)).detach().cpu().numpy().astype("float32"))
        if not batches:
            raise ValueError("image_paths cannot be empty")
        return np.concatenate(batches, axis=0)
