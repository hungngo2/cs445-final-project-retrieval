from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .render import PageRenderer
from .schemas import PageRecord, QuestionRecord, RankedPage
from .utils import dump_json, dump_jsonl, ensure_dir, load_json, load_jsonl
from .vision import DEFAULT_MODELS, VisionTextEncoder


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss is required for multimodal FAISS retrieval. Install faiss-cpu in Colab.") from exc
    return faiss


@dataclass(frozen=True)
class EmbeddingMetadata:
    doc_id: str
    page_num: int
    crop_mode: str
    crop_name: str | None

    def page_key(self) -> tuple[str, int]:
        return (self.doc_id, self.page_num)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "crop_mode": self.crop_mode,
            "crop_name": self.crop_name,
        }


def aggregate_embedding_hits(
    scores: list[float],
    embedding_indices: list[int],
    metadata: list[EmbeddingMetadata],
) -> list[tuple[tuple[str, int], float, str | None]]:
    page_scores: dict[tuple[str, int], tuple[float, str | None]] = {}
    for score, embedding_index in zip(scores, embedding_indices, strict=True):
        if embedding_index < 0:
            continue
        entry = metadata[embedding_index]
        page_key = entry.page_key()
        existing = page_scores.get(page_key)
        if existing is None or score > existing[0]:
            page_scores[page_key] = (float(score), entry.crop_name)
    return sorted(
        ((page_key, score, crop_name) for page_key, (score, crop_name) in page_scores.items()),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )


class MultimodalFaissIndex:
    def __init__(
        self,
        index,
        metadata: list[EmbeddingMetadata],
        model_key: str,
        crop_mode: str,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.model_key = model_key
        self.crop_mode = crop_mode
        self.model_name = model_name or DEFAULT_MODELS[model_key]
        self.device = device
        self.batch_size = batch_size
        self.encoder = VisionTextEncoder(
            model_key=model_key,
            model_name=self.model_name,
            device=device,
            batch_size=batch_size,
        )

    @classmethod
    def build(
        cls,
        page_records: list[PageRecord],
        index_dir: str | Path,
        model_key: str,
        crop_mode: str,
        render_cache: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        page_batch_size: int = 32,
        dpi: int = 150,
    ) -> "MultimodalFaissIndex":
        if not page_records:
            raise ValueError("page_records cannot be empty")

        encoder = VisionTextEncoder(model_key=model_key, model_name=model_name, device=device, batch_size=batch_size)
        renderer = PageRenderer(render_cache, dpi=dpi)
        faiss = _import_faiss()

        metadata: list[EmbeddingMetadata] = []
        index = None

        for start in tqdm(range(0, len(page_records), page_batch_size), desc=f"Building {model_key} FAISS index"):
            page_batch = page_records[start : start + page_batch_size]
            image_paths = []
            batch_metadata: list[EmbeddingMetadata] = []
            for page_record in page_batch:
                page_image_paths = renderer.get_image_paths(page_record, crop_mode=crop_mode)
                for image_path in page_image_paths:
                    image_paths.append(image_path)
                    batch_metadata.append(
                        EmbeddingMetadata(
                            doc_id=page_record.doc_id,
                            page_num=page_record.page_num,
                            crop_mode=crop_mode,
                            crop_name=image_path.name if crop_mode != "full" else None,
                        )
                    )

            embeddings = encoder.embed_image_paths(image_paths)
            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.asarray(embeddings, dtype="float32"))
            metadata.extend(batch_metadata)

        if index is None:
            raise ValueError("No embeddings were generated for the FAISS index.")

        built = cls(
            index=index,
            metadata=metadata,
            model_key=model_key,
            crop_mode=crop_mode,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        built.save(index_dir)
        return built

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        device: str | None = None,
        batch_size: int = 8,
    ) -> "MultimodalFaissIndex":
        faiss = _import_faiss()
        root = Path(index_dir)
        config = load_json(root / "config.json")
        index = faiss.read_index(str(root / "index.faiss"))
        metadata = [EmbeddingMetadata(**row) for row in load_jsonl(root / "metadata.jsonl")]
        return cls(
            index=index,
            metadata=metadata,
            model_key=config["model_key"],
            crop_mode=config["crop_mode"],
            model_name=config.get("model_name"),
            device=device,
            batch_size=batch_size,
        )

    def save(self, index_dir: str | Path) -> None:
        faiss = _import_faiss()
        root = ensure_dir(index_dir)
        faiss.write_index(self.index, str(root / "index.faiss"))
        dump_jsonl([row.to_dict() for row in self.metadata], root / "metadata.jsonl")
        dump_json(
            {
                "model_key": self.model_key,
                "model_name": self.model_name,
                "crop_mode": self.crop_mode,
            },
            root / "config.json",
        )

    def search(
        self,
        query: str,
        uid: str,
        top_k: int = 50,
        search_k_multiplier: int = 8,
    ) -> list[RankedPage]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.encoder.embed_texts([query])
        search_k = min(self.index.ntotal, max(top_k, top_k * search_k_multiplier))

        aggregated_hits: list[tuple[tuple[str, int], float, str | None]] = []
        while True:
            scores, indices = self.index.search(np.asarray(query_embedding, dtype="float32"), search_k)
            aggregated_hits = aggregate_embedding_hits(
                scores=scores[0].tolist(),
                embedding_indices=indices[0].tolist(),
                metadata=self.metadata,
            )
            if len(aggregated_hits) >= top_k or search_k >= self.index.ntotal:
                break
            search_k = min(self.index.ntotal, search_k * 2)

        method_name = f"{self.model_key}_faiss"
        results: list[RankedPage] = []
        for rank, (page_key, score, crop_name) in enumerate(aggregated_hits[:top_k], start=1):
            doc_id, page_num = page_key
            results.append(
                RankedPage(
                    uid=uid,
                    doc_id=doc_id,
                    page_num=page_num,
                    score=score,
                    rank=rank,
                    method=method_name,
                    crop_mode=self.crop_mode,
                    matched_crop=crop_name,
                    component_scores={"faiss_score": score},
                )
            )
        return results

    def batch_search(
        self,
        questions: list[QuestionRecord],
        top_k: int = 50,
        search_k_multiplier: int = 8,
    ) -> dict[str, list[RankedPage]]:
        return {
            question.uid: self.search(
                question.question,
                uid=question.uid,
                top_k=top_k,
                search_k_multiplier=search_k_multiplier,
            )
            for question in tqdm(questions, desc=f"{self.model_key} FAISS retrieval ({self.crop_mode})", leave=False)
        }
