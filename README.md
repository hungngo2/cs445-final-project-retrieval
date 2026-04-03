# OfficeQA Retrieval Baselines

Retrieval-only OfficeQA project focused on page retrieval with:

- BM25 page retrieval
- BM25 + CLIP reranking
- BM25 + SigLIP reranking
- full-page vs fixed-crop ablations

The implementation is Python-module first and Colab-friendly. Heavy runs are meant for Google Colab Pro, while local development and debugging can happen on a laptop.

## Repository layout

- `src/officeqa_retrieval/`
  - core loaders, indexing, rendering, reranking, metrics, and pipeline code
- `scripts/`
  - thin wrappers for the main experiment commands
- `tests/`
  - unit and integration tests on synthetic data
- `notebooks/`
  - Colab runner notebook and lightweight analysis workflow

## Expected data layout

The code assumes an OfficeQA-style directory with:

- a questions CSV, such as `officeqa.csv` or `officeqa_pro.csv`
- a directory containing the source Treasury Bulletin PDFs

Example:

```text
data/
  officeqa.csv
  pdfs/
    treasury_bulletin_1941_01.pdf
    treasury_bulletin_1944_01.pdf
    ...
```

The public OfficeQA CSV uses these columns:

- `uid`
- `question`
- `answer`
- `source_docs`
- `source_files`
- `difficulty`

This project uses `source_docs` and `source_files` to derive gold page labels for retrieval evaluation.

## Installation

Base install:

```bash
python3 -m pip install -e .
```

Install with reranker dependencies:

```bash
python3 -m pip install -e ".[ml]"
```

Install development dependencies:

```bash
python3 -m pip install -e ".[ml,dev]"
```

## Quick start

1. Prepare page text and a sanity subset:

```bash
python3 scripts/download_officeqa_pdfs.py \
  --questions-csv data/officeqa_pro.csv \
  --pdf-dir data/pdfs \
  --skip-existing

python3 scripts/prepare_data.py \
  --questions-csv data/officeqa_pro.csv \
  --pdf-dir data/pdfs \
  --manifest-out artifacts/page_manifest.jsonl \
  --sanity-out artifacts/sanity_questions.json
```

2. Build a BM25 page index:

```bash
python3 scripts/build_page_index.py \
  --manifest artifacts/page_manifest.jsonl \
  --index-out artifacts/page_bm25.pkl
```

3. Run BM25 only:

```bash
python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa.csv \
  --manifest artifacts/page_manifest.jsonl \
  --index artifacts/page_bm25.pkl \
  --out-dir results/bm25 \
  --model bm25 \
  --shortlist-k 50
```

4. Run CLIP full-page reranking:

```bash
python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa.csv \
  --manifest artifacts/page_manifest.jsonl \
  --index artifacts/page_bm25.pkl \
  --out-dir results/clip_full \
  --model clip \
  --crop-mode full \
  --render-cache artifacts/render_cache \
  --shortlist-k 50
```

5. Run SigLIP fixed-crop reranking:

```bash
python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa.csv \
  --manifest artifacts/page_manifest.jsonl \
  --index artifacts/page_bm25.pkl \
  --out-dir results/siglip_crops \
  --model siglip \
  --crop-mode fixed_2x2 \
  --render-cache artifacts/render_cache \
  --shortlist-k 50
```

6. Build a summary table:

```bash
python3 scripts/make_report_tables.py \
  --results-dir results \
  --summary-out results/summary.csv
```

## Core experiment matrix

- `BM25`
- `BM25 + CLIP full-page`
- `BM25 + CLIP fixed_2x2`
- `BM25 + SigLIP full-page`
- `BM25 + SigLIP fixed_2x2`

Suggested shortlist depths:

- `10`
- `20`
- `50`

Primary metrics:

- `Page Recall@1`
- `Page Recall@5`
- `Page Recall@10`
- `Page MRR`
- `nDCG@10`

Optional secondary metrics:

- document-level recall after collapsing page rankings by document

## Notes on implementation

- No vector database is required in this version.
- `scripts/download_officeqa_pdfs.py` downloads the Treasury Bulletin PDFs referenced by an OfficeQA CSV directly from FRASER.
- BM25 is built over extracted page text from PDFs.
- Rerankers only see the BM25 candidate pages, which keeps the experiment fair and Colab-friendly.
- Fixed crops use a deterministic 2x2 grid and score a page by the best crop score.

## Colab workflow

Use `notebooks/officeqa_retrieval_colab.ipynb` as the main runner notebook. It mounts Google Drive, installs the package, and runs the exact CLI commands above.

## Course deliverable readiness

This repository is organized to support the final submission requirements:

- code in a clean project directory
- a README explaining structure and how to run
- reproducible experiment commands
- saved metrics and ranked outputs for paper tables
