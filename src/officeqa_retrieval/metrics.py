from __future__ import annotations

from collections import defaultdict

from .schemas import GoldReference, QuestionRecord, RankedPage


def gold_page_keys(question: QuestionRecord) -> set[tuple[str, int]]:
    keys = set()
    for reference in question.gold_references:
        if reference.doc_id and reference.page_num is not None:
            keys.add((reference.doc_id, reference.page_num))
    return keys


def gold_doc_ids(question: QuestionRecord) -> set[str]:
    return {reference.doc_id for reference in question.gold_references if reference.doc_id}


def collapse_to_unique_docs(predictions: list[RankedPage]) -> list[str]:
    seen: set[str] = set()
    ordered_doc_ids: list[str] = []
    for prediction in predictions:
        if prediction.doc_id in seen:
            continue
        seen.add(prediction.doc_id)
        ordered_doc_ids.append(prediction.doc_id)
    return ordered_doc_ids


def recall_at_k(predicted_items: list, gold_items: set, k: int) -> float:
    if not gold_items:
        return 0.0
    return 1.0 if any(item in gold_items for item in predicted_items[:k]) else 0.0


def reciprocal_rank(predicted_items: list, gold_items: set) -> float:
    if not gold_items:
        return 0.0
    for index, item in enumerate(predicted_items, start=1):
        if item in gold_items:
            return 1.0 / index
    return 0.0


def ndcg_at_k(predicted_items: list, gold_items: set, k: int) -> float:
    if not gold_items:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(predicted_items[:k], start=1):
        if item in gold_items:
            dcg += 1.0 / _log2(index + 1)
    ideal_hits = min(len(gold_items), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / _log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _log2(value: float) -> float:
    import math

    return math.log2(value)


def evaluate_predictions(
    questions: list[QuestionRecord],
    predictions_by_uid: dict[str, list[RankedPage]],
    ks: tuple[int, ...] = (1, 5, 10),
    include_doc_metrics: bool = True,
) -> dict:
    page_recalls = {k: [] for k in ks}
    doc_recalls = {k: [] for k in ks}
    page_mrrs: list[float] = []
    doc_mrrs: list[float] = []
    ndcgs: list[float] = []
    per_query_rows: list[dict] = []

    for question in questions:
        predictions = predictions_by_uid.get(question.uid, [])
        page_keys = [prediction.key() for prediction in predictions]
        doc_ids = collapse_to_unique_docs(predictions)
        gold_pages = gold_page_keys(question)
        gold_docs = gold_doc_ids(question)

        for k in ks:
            page_recalls[k].append(recall_at_k(page_keys, gold_pages, k))
            if include_doc_metrics:
                doc_recalls[k].append(recall_at_k(doc_ids, gold_docs, k))

        query_page_mrr = reciprocal_rank(page_keys, gold_pages)
        page_mrrs.append(query_page_mrr)
        ndcgs.append(ndcg_at_k(page_keys, gold_pages, max(ks)))

        query_row = {
            "uid": question.uid,
            "difficulty": question.difficulty,
            "gold_page_count": len(gold_pages),
            "page_mrr": query_page_mrr,
            "ndcg_at_max_k": ndcgs[-1],
        }
        if include_doc_metrics:
            query_doc_mrr = reciprocal_rank(doc_ids, gold_docs)
            doc_mrrs.append(query_doc_mrr)
            query_row["doc_mrr"] = query_doc_mrr

        for k in ks:
            query_row[f"page_recall_at_{k}"] = page_recalls[k][-1]
            if include_doc_metrics:
                query_row[f"doc_recall_at_{k}"] = doc_recalls[k][-1]
        per_query_rows.append(query_row)

    summary = {
        "query_count": len(questions),
        "page_mrr": _mean(page_mrrs),
        f"ndcg_at_{max(ks)}": _mean(ndcgs),
    }
    for k in ks:
        summary[f"page_recall_at_{k}"] = _mean(page_recalls[k])
        if include_doc_metrics:
            summary[f"doc_recall_at_{k}"] = _mean(doc_recalls[k])
    if include_doc_metrics:
        summary["doc_mrr"] = _mean(doc_mrrs)

    return {
        "summary": summary,
        "per_query": per_query_rows,
    }


def summarize_by_difficulty(
    questions: list[QuestionRecord],
    predictions_by_uid: dict[str, list[RankedPage]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict]:
    grouped_questions: dict[str, list[QuestionRecord]] = defaultdict(list)
    for question in questions:
        grouped_questions[question.difficulty or "unknown"].append(question)
    return {
        difficulty: evaluate_predictions(group, predictions_by_uid, ks=ks)["summary"]
        for difficulty, group in grouped_questions.items()
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
