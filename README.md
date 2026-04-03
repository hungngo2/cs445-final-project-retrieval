# OfficeQA Retrieval Baselines

Retrieval-only OfficeQA project focused on page retrieval with:

- BM25 page retrieval
- CLIP multimodal FAISS retrieval
- SigLIP multimodal FAISS retrieval
- full-page vs fixed-crop ablations

The implementation is Python-module first and Colab-friendly. Heavy runs are meant for Google Colab Pro, while local development and debugging can happen on a laptop.

The recommended execution path is the single Colab notebook at `notebooks/officeqa_retrieval_colab.ipynb`. It mounts Drive, downloads missing PDFs, builds the retrieval indexes, runs the experiment suite, and exports analysis artifacts for the report.

## Repository layout

- `src/officeqa_retrieval/`
  - core loaders, indexing, rendering, multimodal retrieval, metrics, and pipeline code
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

Install with model dependencies:

```bash
python3 -m pip install -e ".[ml]"
```

Install with FAISS and development dependencies:

```bash
python3 -m pip install -e ".[ml,faiss,dev]"
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

3. Build multimodal FAISS indexes:

```bash
python3 scripts/build_multimodal_index.py \
  --manifest artifacts/page_manifest.jsonl \
  --index-dir artifacts/clip_faiss_full \
  --model clip \
  --crop-mode full \
  --render-cache artifacts/render_cache

python3 scripts/build_multimodal_index.py \
  --manifest artifacts/page_manifest.jsonl \
  --index-dir artifacts/clip_faiss_fixed \
  --model clip \
  --crop-mode fixed_2x2 \
  --render-cache artifacts/render_cache

python3 scripts/build_multimodal_index.py \
  --manifest artifacts/page_manifest.jsonl \
  --index-dir artifacts/siglip_faiss_full \
  --model siglip \
  --crop-mode full \
  --render-cache artifacts/render_cache

python3 scripts/build_multimodal_index.py \
  --manifest artifacts/page_manifest.jsonl \
  --index-dir artifacts/siglip_faiss_fixed \
  --model siglip \
  --crop-mode fixed_2x2 \
  --render-cache artifacts/render_cache
```

4. Run retrieval experiments:

```bash
python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa_pro.csv \
  --index artifacts/page_bm25.pkl \
  --out-dir results/bm25 \
  --model bm25 \
  --top-k 50

python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa_pro.csv \
  --index artifacts/clip_faiss_full \
  --out-dir results/clip_faiss_full \
  --model clip_faiss \
  --top-k 50

python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa_pro.csv \
  --index artifacts/clip_faiss_fixed \
  --out-dir results/clip_faiss_fixed \
  --model clip_faiss \
  --top-k 50

python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa_pro.csv \
  --index artifacts/siglip_faiss_full \
  --out-dir results/siglip_faiss_full \
  --model siglip_faiss \
  --top-k 50

python3 scripts/run_retrieval_eval.py \
  --questions-csv data/officeqa_pro.csv \
  --index artifacts/siglip_faiss_fixed \
  --out-dir results/siglip_faiss_fixed \
  --model siglip_faiss \
  --top-k 50
```

5. Build a summary table:

```bash
python3 scripts/make_report_tables.py \
  --results-dir results \
  --summary-out results/summary.csv
```

Each run persists:

- `results/<run_name>/metrics.json`
- `results/<run_name>/ranked_pages.jsonl`
- `results/summary.csv`

The Colab notebook also exports analysis-friendly files under `results/analysis/`, including summary tables, per-query metrics, difficulty summaries, and qualitative slices for selected question IDs.

## Core experiment matrix

- `BM25`
- `CLIP-FAISS full-page`
- `CLIP-FAISS fixed_2x2`
- `SigLIP-FAISS full-page`
- `SigLIP-FAISS fixed_2x2`

Suggested retrieval depths:

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
- Multimodal retrieval uses CLIP or SigLIP text-to-image embeddings indexed in FAISS with inner-product similarity.
- Fixed crops use a deterministic 2x2 grid, index each crop separately, and aggregate retrieval scores back to the page level using the best crop score.

## Colab workflow

Use `notebooks/officeqa_retrieval_colab.ipynb` as the main runner notebook. It supports:

- `RUN_MODE = "smoke"` using the bundled `data/officeqa_pro_smoke.csv`
- `RUN_MODE = "full"` using `MyDrive/officeqa/officeqa_pro.csv`
- automatic PDF downloading into `MyDrive/officeqa/pdfs/`
- artifact reuse so reruns can skip expensive index builds
- analysis exports under `MyDrive/officeqa/results/<run_tag>/analysis/`

## Course deliverable readiness

This repository is organized to support the final submission requirements:

- code in a clean project directory
- a README explaining structure and how to run
- reproducible experiment commands
- saved metrics and ranked outputs for paper tables
