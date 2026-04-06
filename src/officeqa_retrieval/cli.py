from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .bm25 import PageBm25Index
from .colqwen import DEFAULT_COLQWEN_MODEL
from .dataset import load_questions, save_sanity_subset
from .faiss_index import MultimodalFaissIndex
from .manifest import build_page_manifest, load_page_manifest
from .ocr_manifest import build_ocr_page_manifest
from .pipeline import run_bm25_experiment, run_colqwen2_experiment, run_multimodal_faiss_experiment, save_run_artifacts
from .utils import ensure_parent_dir


def prepare_data_main() -> None:
    parser = argparse.ArgumentParser(description="Build an OfficeQA page manifest and sanity subset.")
    parser.add_argument("--questions-csv", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--sanity-out", required=True)
    parser.add_argument("--sanity-size", type=int, default=10)
    args = parser.parse_args()

    questions = load_questions(args.questions_csv)
    build_page_manifest(args.pdf_dir, questions_csv=args.questions_csv, output_path=args.manifest_out)
    save_sanity_subset(questions, args.sanity_out, size=args.sanity_size)


def build_page_index_main() -> None:
    parser = argparse.ArgumentParser(description="Build a BM25 page index from a page manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-out", required=True)
    args = parser.parse_args()

    page_records = load_page_manifest(args.manifest)
    index = PageBm25Index.build(page_records)
    index.save(args.index_out)


def build_ocr_manifest_main() -> None:
    parser = argparse.ArgumentParser(description="Build an OCR-derived page manifest from a native page manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--render-cache", required=True)
    parser.add_argument("--ocr-cache-dir", required=True)
    parser.add_argument("--text-source", choices=["ocr", "hybrid"], default="ocr")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--progress-out")
    args = parser.parse_args()

    build_ocr_page_manifest(
        input_manifest_path=args.manifest,
        output_path=args.manifest_out,
        render_cache=args.render_cache,
        ocr_cache_dir=args.ocr_cache_dir,
        text_source=args.text_source,
        dpi=args.dpi,
        tesseract_cmd=args.tesseract_cmd,
        lang=args.lang,
        progress_path=args.progress_out,
    )


def build_multimodal_index_main() -> None:
    parser = argparse.ArgumentParser(description="Build a multimodal FAISS page index from a page manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--model", choices=["clip", "siglip"], required=True)
    parser.add_argument("--crop-mode", choices=["full", "fixed_2x2", "layout_aware"], default="full")
    parser.add_argument("--render-cache", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--page-batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    page_records = load_page_manifest(args.manifest)
    MultimodalFaissIndex.build(
        page_records=page_records,
        index_dir=args.index_dir,
        model_key=args.model,
        crop_mode=args.crop_mode,
        render_cache=args.render_cache,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        page_batch_size=args.page_batch_size,
        dpi=args.dpi,
    )


def run_retrieval_eval_main() -> None:
    parser = argparse.ArgumentParser(description="Run OfficeQA retrieval experiments.")
    parser.add_argument("--questions-csv", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=["bm25", "clip_faiss", "siglip_faiss", "colqwen2_rerank"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--search-k-multiplier", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--render-cache")
    parser.add_argument("--embedding-cache-dir")
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--model-name", default=DEFAULT_COLQWEN_MODEL)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.model == "bm25":
        questions, predictions = run_bm25_experiment(
            questions_csv=args.questions_csv,
            index_path=args.index,
            top_k=args.top_k,
        )
        method = "bm25"
    elif args.model == "colqwen2_rerank":
        if not args.render_cache:
            raise ValueError("--render-cache is required for colqwen2_rerank runs.")
        if not args.embedding_cache_dir:
            raise ValueError("--embedding-cache-dir is required for colqwen2_rerank runs.")
        questions, predictions = run_colqwen2_experiment(
            questions_csv=args.questions_csv,
            index_path=args.index,
            render_cache=args.render_cache,
            embedding_cache_dir=args.embedding_cache_dir,
            top_k=args.top_k,
            candidate_top_k=args.candidate_top_k,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            dpi=args.dpi,
        )
        method = "colqwen2_rerank"
    else:
        questions, predictions = run_multimodal_faiss_experiment(
            questions_csv=args.questions_csv,
            index_path=args.index,
            top_k=args.top_k,
            search_k_multiplier=args.search_k_multiplier,
            device=args.device,
            batch_size=args.batch_size,
        )
        method = args.model

    metrics = save_run_artifacts(
        out_dir=args.out_dir,
        questions=questions,
        predictions_by_uid=predictions,
        method=method,
    )
    print(json.dumps(metrics["summary"], indent=2))


def make_report_tables_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize retrieval runs into a CSV report table.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    result_root = Path(args.results_dir)
    metric_paths = sorted(result_root.glob("*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics.json files found under {result_root}")

    rows = []
    for metric_path in metric_paths:
        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {"run_name": metric_path.parent.name, "method": metrics.get("method", metric_path.parent.name)}
        row.update(metrics["summary"])
        rows.append(row)

    summary_path = ensure_parent_dir(args.summary_out)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(
        "Run one of the named entrypoints instead: officeqa-prepare-data, officeqa-build-page-index, "
        "officeqa-build-ocr-manifest, officeqa-build-multimodal-index, officeqa-run-retrieval-eval, "
        "or officeqa-make-report-tables."
    )
