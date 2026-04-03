from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .render import PageRenderer
from .schemas import PageRecord, RankedPage
from .utils import batched

DEFAULT_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for vision reranking.") from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for vision reranking.") from exc
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


class VisionReranker:
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
        self.processor = self._load_pretrained(auto_processor, self.model_name)
        self.model = self._load_pretrained(auto_model, self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_pretrained(loader, model_name: str):
        try:
            return loader.from_pretrained(model_name)
        except Exception as first_error:
            try:
                return loader.from_pretrained(model_name, local_files_only=True)
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

    def _text_features(self, query: str):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
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

    def score_image_paths(self, query: str, image_paths: list[Path]) -> list[float]:
        if not image_paths:
            return []
        query_features = self._text_features(query)
        scores: list[float] = []
        for image_batch in batched(image_paths, self.batch_size):
            image_features = self._image_features(list(image_batch))
            batch_scores = query_features @ image_features.T
            scores.extend(batch_scores[0].detach().cpu().tolist())
        return [float(score) for score in scores]

    def rerank_candidates(
        self,
        uid: str,
        query: str,
        candidates: list[RankedPage],
        page_lookup: dict[tuple[str, int], PageRecord],
        renderer: PageRenderer,
        crop_mode: str = "full",
        top_k: int | None = None,
    ) -> list[RankedPage]:
        if not candidates:
            return []

        flat_image_paths: list[Path] = []
        flat_candidate_indices: list[int] = []
        flat_crop_names: list[str | None] = []

        for candidate_index, candidate in enumerate(candidates):
            page_record = page_lookup[(candidate.doc_id, candidate.page_num)]
            image_paths = renderer.get_image_paths(page_record, crop_mode=crop_mode)
            for image_path in image_paths:
                flat_image_paths.append(image_path)
                flat_candidate_indices.append(candidate_index)
                flat_crop_names.append(image_path.name if crop_mode != "full" else None)

        vision_scores = self.score_image_paths(query, flat_image_paths)
        best_scores: dict[int, tuple[float, str | None]] = {}
        for candidate_index, score, crop_name in zip(flat_candidate_indices, vision_scores, flat_crop_names, strict=True):
            existing = best_scores.get(candidate_index)
            if existing is None or score > existing[0]:
                best_scores[candidate_index] = (score, crop_name)

        reranked: list[RankedPage] = []
        for candidate_index, candidate in enumerate(candidates):
            best_score, crop_name = best_scores[candidate_index]
            reranked.append(
                replace(
                    candidate,
                    uid=uid,
                    score=best_score,
                    rank=0,
                    method=self.model_key,
                    crop_mode=crop_mode,
                    matched_crop=crop_name,
                    component_scores={
                        "vision_score": best_score,
                        "bm25_score": candidate.bm25_score if candidate.bm25_score is not None else candidate.score,
                    },
                )
            )

        reranked.sort(key=lambda candidate: (-candidate.score, -(candidate.bm25_score or 0.0), candidate.doc_id, candidate.page_num))
        if top_k is not None:
            reranked = reranked[:top_k]
        for rank, candidate in enumerate(reranked, start=1):
            candidate.rank = rank
        return reranked
