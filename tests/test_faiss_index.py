from __future__ import annotations

from officeqa_retrieval.faiss_index import EmbeddingMetadata, aggregate_embedding_hits


def test_aggregate_embedding_hits_uses_best_crop_per_page() -> None:
    metadata = [
        EmbeddingMetadata(doc_id="doc_a", page_num=1, crop_mode="fixed_2x2", crop_name="crop_1"),
        EmbeddingMetadata(doc_id="doc_a", page_num=1, crop_mode="fixed_2x2", crop_name="crop_2"),
        EmbeddingMetadata(doc_id="doc_b", page_num=2, crop_mode="fixed_2x2", crop_name="crop_1"),
    ]

    aggregated = aggregate_embedding_hits(
        scores=[0.1, 0.9, 0.5],
        embedding_indices=[0, 1, 2],
        metadata=metadata,
    )

    assert aggregated == [
        (("doc_a", 1), 0.9, "crop_2"),
        (("doc_b", 2), 0.5, "crop_1"),
    ]


def test_aggregate_embedding_hits_skips_invalid_indices() -> None:
    metadata = [
        EmbeddingMetadata(doc_id="doc_a", page_num=1, crop_mode="full", crop_name=None),
    ]

    aggregated = aggregate_embedding_hits(
        scores=[0.7, 0.1],
        embedding_indices=[0, -1],
        metadata=metadata,
    )

    assert aggregated == [(("doc_a", 1), 0.7, None)]
