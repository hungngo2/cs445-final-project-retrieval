from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from .bm25 import PageBm25Index
from .dataset import load_questions
from .render import PageRenderer
from .schemas import PageRecord, RankedPage
from .vision import resolve_device

DEFAULT_COLQWEN_MODEL = "vidore/colqwen2-v1.0"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for ColQwen2 retrieval.") from exc
    return torch


def _import_colqwen():
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
    except ImportError as exc:
        raise ImportError(
            "colpali-engine is required for ColQwen2 retrieval. Install colpali-engine[interpretability]."
        ) from exc
    return ColQwen2, ColQwen2Processor


def _import_flash_attention_checker():
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
    except ImportError:
        return lambda: False
    return is_flash_attn_2_available


def _import_interpretability():
    try:
        from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_all_similarity_maps
    except ImportError as exc:
        raise ImportError(
            "colpali-engine[interpretability] is required for ColQwen2 similarity-map visualizations."
        ) from exc
    return get_similarity_maps_from_embeddings, plot_all_similarity_maps


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


class ColQwen2Bm25Reranker:
    def __init__(
        self,
        render_cache: str | Path,
        embedding_cache_dir: str | Path,
        model_name: str = DEFAULT_COLQWEN_MODEL,
        device: str | None = None,
        batch_size: int = 2,
        dpi: int = 150,
    ) -> None:
        torch = _import_torch()
        colqwen_cls, processor_cls = _import_colqwen()
        flash_attention_checker = _import_flash_attention_checker()

        self.torch = torch
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_cache_key = _sanitize_model_name(model_name)
        self.renderer = PageRenderer(render_cache, dpi=dpi)
        self.embedding_cache_dir = Path(embedding_cache_dir) / self.model_cache_key
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_map = "cuda:0" if self.device == "cuda" else self.device
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }
        if self.device == "cuda" and flash_attention_checker():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = colqwen_cls.from_pretrained(model_name, **model_kwargs).eval()
        self.processor = processor_cls.from_pretrained(model_name)

    def _embedding_cache_path(self, page_record: PageRecord) -> Path:
        return self.embedding_cache_dir / page_record.doc_id / f"page_{page_record.page_num:04d}.pt"

    def _move_batch_to_device(self, batch):
        if hasattr(batch, "to"):
            return batch.to(self.device_map)
        return {key: value.to(self.device_map) for key, value in batch.items()}

    def _extract_embeddings(self, outputs):
        if hasattr(outputs, "embeddings"):
            return outputs.embeddings
        return outputs

    def _split_batch_embeddings(self, embeddings) -> list:
        if isinstance(embeddings, (list, tuple)):
            return [embedding.detach().to("cpu") for embedding in embeddings]
        if hasattr(self.torch, "is_tensor") and self.torch.is_tensor(embeddings):
            if embeddings.ndim == 2:
                return [embeddings.detach().to("cpu")]
            if embeddings.ndim >= 3:
                return [embedding.detach().to("cpu") for embedding in self.torch.unbind(embeddings, dim=0)]
        raise TypeError(f"Unsupported ColQwen2 embedding container: {type(embeddings)!r}")

    def _load_images(self, image_paths: Sequence[Path]):
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        return images

    def embed_queries(self, queries: Sequence[str]) -> list:
        embeddings = []
        for start in range(0, len(queries), self.batch_size):
            batch_queries = list(queries[start : start + self.batch_size])
            batch = self._move_batch_to_device(self.processor.process_queries(batch_queries))
            with self.torch.no_grad():
                outputs = self.model(**batch)
            embeddings.extend(self._split_batch_embeddings(self._extract_embeddings(outputs)))
        return embeddings

    def embed_page_records(self, page_records: Sequence[PageRecord]) -> list:
        cache_paths = [self._embedding_cache_path(page_record) for page_record in page_records]
        missing_positions = [index for index, cache_path in enumerate(cache_paths) if not cache_path.exists()]

        if missing_positions:
            for start in range(0, len(missing_positions), self.batch_size):
                batch_positions = missing_positions[start : start + self.batch_size]
                batch_records = [page_records[position] for position in batch_positions]
                image_paths = [self.renderer.render_page(page_record) for page_record in batch_records]
                images = self._load_images(image_paths)
                batch = self._move_batch_to_device(self.processor.process_images(images))
                with self.torch.no_grad():
                    outputs = self.model(**batch)
                batch_embeddings = self._split_batch_embeddings(self._extract_embeddings(outputs))
                for position, embedding in zip(batch_positions, batch_embeddings, strict=True):
                    cache_path = cache_paths[position]
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.torch.save(embedding, cache_path)

        return [self.torch.load(cache_path, map_location="cpu") for cache_path in cache_paths]

    def score_query_against_pages(self, query: str, page_records: Sequence[PageRecord]) -> list[float]:
        query_embeddings = self.embed_queries([query])
        page_embeddings = self.embed_page_records(page_records)
        score_fn = getattr(self.processor, "score_multi_vector", None) or getattr(self.processor, "score_retrieval", None)
        if score_fn is None:
            raise AttributeError("ColQwen2 processor does not expose score_multi_vector() or score_retrieval().")

        try:
            scores = score_fn(query_embeddings, page_embeddings, batch_size=32)
        except TypeError:
            scores = score_fn(query_embeddings, page_embeddings)

        if hasattr(scores, "detach"):
            scores = scores.detach().to("cpu")
        return [float(score) for score in scores[0].tolist()]

    def rerank_candidates(
        self,
        query: str,
        uid: str,
        candidates: Sequence[RankedPage],
        page_records: Sequence[PageRecord],
        top_k: int = 50,
    ) -> list[RankedPage]:
        scores = self.score_query_against_pages(query, page_records)
        reranked_rows = []
        for candidate, score in zip(candidates, scores, strict=True):
            reranked_rows.append(
                RankedPage(
                    uid=uid,
                    doc_id=candidate.doc_id,
                    page_num=candidate.page_num,
                    score=score,
                    rank=0,
                    method="colqwen2_rerank",
                    bm25_score=candidate.bm25_score or candidate.score,
                    crop_mode="full",
                    matched_crop=None,
                    component_scores={
                        "colqwen2_score": score,
                        "bm25_score": candidate.bm25_score or candidate.score,
                    },
                )
            )

        reranked_rows.sort(
            key=lambda row: (
                -row.score,
                -(row.bm25_score or 0.0),
                row.doc_id,
                row.page_num,
            )
        )
        reranked_rows = reranked_rows[:top_k]
        for rank, row in enumerate(reranked_rows, start=1):
            row.rank = rank
        return reranked_rows

    def get_similarity_map_artifacts(self, query: str, page_record: PageRecord):
        get_similarity_maps_from_embeddings, _ = _import_interpretability()
        image_path = self.renderer.render_page(page_record)

        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("RGB").copy()

        batch_images = self._move_batch_to_device(self.processor.process_images([image]))
        batch_queries = self._move_batch_to_device(self.processor.process_queries([query]))

        with self.torch.no_grad():
            image_embeddings = self._extract_embeddings(self.model(**batch_images))
            query_embeddings = self._extract_embeddings(self.model(**batch_queries))

        n_patches = self.processor.get_n_patches(image_size=image.size, patch_size=self.model.patch_size)
        image_mask = self.processor.get_image_mask(batch_images)
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        similarity_maps = batched_similarity_maps[0]
        query_tokens = self.processor.tokenizer.tokenize(query)
        return {
            "image": image,
            "image_path": image_path,
            "query_tokens": query_tokens,
            "similarity_maps": similarity_maps,
        }

    def plot_similarity_maps(self, query: str, page_record: PageRecord, max_tokens: int = 12):
        _, plot_all_similarity_maps = _import_interpretability()
        artifacts = self.get_similarity_map_artifacts(query, page_record)
        similarity_maps = artifacts["similarity_maps"]
        query_tokens = artifacts["query_tokens"][:max_tokens]
        similarity_maps = similarity_maps[: len(query_tokens)]
        return plot_all_similarity_maps(
            image=artifacts["image"],
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
        )


def run_colqwen2_rerank_experiment(
    questions_csv: str | Path,
    candidate_index_path: str | Path,
    render_cache: str | Path,
    embedding_cache_dir: str | Path,
    top_k: int = 50,
    candidate_top_k: int = 50,
    model_name: str = DEFAULT_COLQWEN_MODEL,
    device: str | None = None,
    batch_size: int = 2,
    dpi: int = 150,
):
    questions = load_questions(questions_csv)
    candidate_index = PageBm25Index.load(candidate_index_path)
    page_record_by_key = {page_record.key(): page_record for page_record in candidate_index.page_records}
    reranker = ColQwen2Bm25Reranker(
        render_cache=render_cache,
        embedding_cache_dir=embedding_cache_dir,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        dpi=dpi,
    )

    predictions = {}
    for question in tqdm(questions, desc="ColQwen2 rerank", leave=False):
        candidates = candidate_index.search(question.question, uid=question.uid, top_k=candidate_top_k)
        page_records = [page_record_by_key[candidate.key()] for candidate in candidates]
        predictions[question.uid] = reranker.rerank_candidates(
            question.question,
            uid=question.uid,
            candidates=candidates,
            page_records=page_records,
            top_k=top_k,
        )

    return questions, predictions
