from __future__ import annotations

from pathlib import Path

from .bm25 import PageBm25Index
from .dataset import load_questions
from .faiss_index import MultimodalFaissIndex
from .metrics import evaluate_predictions, summarize_by_difficulty
from .schemas import RankedPage
from .utils import dump_json, dump_jsonl, ensure_dir


def run_bm25_experiment(
    questions_csv: str | Path,
    index_path: str | Path,
    top_k: int = 50,
) -> tuple[list, dict[str, list[RankedPage]]]:
    questions = load_questions(questions_csv)
    index = PageBm25Index.load(index_path)
    predictions = index.batch_search(questions, top_k=top_k)
    return questions, predictions


def run_multimodal_faiss_experiment(
    questions_csv: str | Path,
    index_path: str | Path,
    top_k: int = 50,
    search_k_multiplier: int = 8,
    device: str | None = None,
    batch_size: int = 8,
) -> tuple[list, dict[str, list[RankedPage]]]:
    questions = load_questions(questions_csv)
    index = MultimodalFaissIndex.load(
        index_path,
        device=device,
        batch_size=batch_size,
    )
    predictions = index.batch_search(questions, top_k=top_k, search_k_multiplier=search_k_multiplier)
    return questions, predictions


def save_run_artifacts(
    out_dir: str | Path,
    questions,
    predictions_by_uid: dict[str, list[RankedPage]],
    method: str,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    output_dir = ensure_dir(out_dir)
    metrics = evaluate_predictions(questions, predictions_by_uid, ks=ks)
    metrics["difficulty_breakdown"] = summarize_by_difficulty(questions, predictions_by_uid, ks=ks)
    metrics["method"] = method

    ranked_rows = []
    for question in questions:
        for prediction in predictions_by_uid.get(question.uid, []):
            row = prediction.to_dict()
            row["question"] = question.question
            row["difficulty"] = question.difficulty
            ranked_rows.append(row)

    dump_json(metrics, output_dir / "metrics.json")
    dump_jsonl(ranked_rows, output_dir / "ranked_pages.jsonl")
    return metrics
