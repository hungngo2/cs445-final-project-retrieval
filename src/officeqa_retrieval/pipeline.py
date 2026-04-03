from __future__ import annotations

from pathlib import Path

from .bm25 import PageBm25Index
from .dataset import load_questions
from .manifest import build_page_lookup, load_page_manifest
from .metrics import evaluate_predictions, summarize_by_difficulty
from .render import PageRenderer
from .rerank import VisionReranker
from .schemas import RankedPage
from .utils import dump_json, dump_jsonl, ensure_dir


def run_bm25_experiment(
    questions_csv: str | Path,
    manifest_path: str | Path,
    index_path: str | Path,
    shortlist_k: int = 50,
) -> tuple[list, dict[str, list[RankedPage]]]:
    questions = load_questions(questions_csv)
    _ = load_page_manifest(manifest_path)
    index = PageBm25Index.load(index_path)
    predictions = index.batch_search(questions, top_k=shortlist_k)
    return questions, predictions


def run_vision_experiment(
    questions_csv: str | Path,
    manifest_path: str | Path,
    index_path: str | Path,
    render_cache: str | Path,
    model_key: str,
    crop_mode: str,
    shortlist_k: int = 50,
    final_top_k: int | None = None,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = 8,
) -> tuple[list, dict[str, list[RankedPage]]]:
    questions = load_questions(questions_csv)
    page_records = load_page_manifest(manifest_path)
    page_lookup = build_page_lookup(page_records)
    index = PageBm25Index.load(index_path)
    bm25_predictions = index.batch_search(questions, top_k=shortlist_k)
    renderer = PageRenderer(render_cache)
    reranker = VisionReranker(
        model_key=model_key,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    predictions: dict[str, list[RankedPage]] = {}
    for question in questions:
        candidates = bm25_predictions[question.uid]
        predictions[question.uid] = reranker.rerank_candidates(
            uid=question.uid,
            query=question.question,
            candidates=candidates,
            page_lookup=page_lookup,
            renderer=renderer,
            crop_mode=crop_mode,
            top_k=final_top_k,
        )
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
