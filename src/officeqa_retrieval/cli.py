from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .bm25 import PageBm25Index
from .dataset import load_questions, save_sanity_subset
from .manifest import build_page_manifest, load_page_manifest
from .pipeline import run_bm25_experiment, run_vision_experiment, save_run_artifacts
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


def run_retrieval_eval_main() -> None:
    parser = argparse.ArgumentParser(description="Run OfficeQA retrieval experiments.")
    parser.add_argument("--questions-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=["bm25", "clip", "siglip"], required=True)
    parser.add_argument("--crop-mode", choices=["full", "fixed_2x2"], default="full")
    parser.add_argument("--render-cache")
    parser.add_argument("--shortlist-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.model == "bm25":
        questions, predictions = run_bm25_experiment(
            questions_csv=args.questions_csv,
            manifest_path=args.manifest,
            index_path=args.index,
            shortlist_k=args.shortlist_k,
        )
        method = "bm25"
    else:
        if not args.render_cache:
            parser.error("--render-cache is required for vision reranking")
        questions, predictions = run_vision_experiment(
            questions_csv=args.questions_csv,
            manifest_path=args.manifest,
            index_path=args.index,
            render_cache=args.render_cache,
            model_key=args.model,
            crop_mode=args.crop_mode,
            shortlist_k=args.shortlist_k,
            final_top_k=args.final_top_k,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        method = f"{args.model}_{args.crop_mode}"

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
        "officeqa-run-retrieval-eval, or officeqa-make-report-tables."
    )
