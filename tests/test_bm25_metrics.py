from __future__ import annotations

import pytest

pytest.importorskip("rank_bm25")

from officeqa_retrieval.bm25 import PageBm25Index
from officeqa_retrieval.metrics import evaluate_predictions
from officeqa_retrieval.schemas import GoldReference, PageRecord, QuestionRecord


def test_bm25_ranks_matching_page_first() -> None:
    page_records = [
        PageRecord(doc_id="doc_a", pdf_path="/tmp/doc_a.pdf", page_num=1, page_text="alpha beta gamma"),
        PageRecord(doc_id="doc_a", pdf_path="/tmp/doc_a.pdf", page_num=2, page_text="delta epsilon"),
        PageRecord(doc_id="doc_b", pdf_path="/tmp/doc_b.pdf", page_num=1, page_text="budget receipts defense"),
    ]
    index = PageBm25Index.build(page_records)
    results = index.search("defense receipts", uid="UID0001", top_k=3)

    assert results[0].doc_id == "doc_b"
    assert results[0].page_num == 1


def test_metrics_compute_page_recall_and_mrr() -> None:
    page_records = [
        PageRecord(doc_id="doc_a", pdf_path="/tmp/doc_a.pdf", page_num=1, page_text="alpha beta gamma"),
        PageRecord(doc_id="doc_b", pdf_path="/tmp/doc_b.pdf", page_num=1, page_text="budget receipts defense"),
    ]
    index = PageBm25Index.build(page_records)
    question = QuestionRecord(
        uid="UID0001",
        question="defense receipts",
        answer=None,
        difficulty="easy",
        source_docs=("https://example.com/a?page=1",),
        source_files=("doc_b.txt",),
        gold_references=(GoldReference(source_file="doc_b.txt", doc_id="doc_b", page_num=1),),
    )
    predictions = {question.uid: index.search(question.question, uid=question.uid, top_k=2)}

    metrics = evaluate_predictions([question], predictions)
    assert metrics["summary"]["page_recall_at_1"] == 1.0
    assert metrics["summary"]["page_mrr"] == 1.0
