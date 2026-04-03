from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .dataset import collect_doc_ids, load_questions
from .schemas import PageRecord
from .utils import dump_jsonl, load_jsonl, normalize_doc_id


def _import_fitz():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required to build the page manifest.") from exc
    return fitz


def discover_pdfs(pdf_dir: str | Path) -> dict[str, Path]:
    root = Path(pdf_dir)
    pdf_paths = sorted(root.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {root}")
    return {normalize_doc_id(path.name): path for path in pdf_paths}


def build_page_manifest(
    pdf_dir: str | Path,
    questions_csv: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[PageRecord]:
    fitz = _import_fitz()
    pdf_map = discover_pdfs(pdf_dir)

    target_doc_ids: set[str] | None = None
    if questions_csv is not None:
        target_doc_ids = collect_doc_ids(load_questions(questions_csv))
        missing = sorted(doc_id for doc_id in target_doc_ids if doc_id not in pdf_map)
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"Could not find PDFs for {len(missing)} source files. Missing examples: {preview}"
            )

    page_records: list[PageRecord] = []
    iterable = sorted(pdf_map.items())
    for doc_id, pdf_path in tqdm(iterable, desc="Extracting PDF pages"):
        if target_doc_ids is not None and doc_id not in target_doc_ids:
            continue
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                page_records.append(
                    PageRecord(
                        doc_id=doc_id,
                        pdf_path=str(pdf_path.resolve()),
                        page_num=page_index,
                        page_text=page.get_text("text").strip(),
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                    )
                )

    if output_path is not None:
        dump_jsonl([record.to_dict() for record in page_records], output_path)
    return page_records


def load_page_manifest(path: str | Path) -> list[PageRecord]:
    rows = load_jsonl(path)
    return [PageRecord(**row) for row in rows]


def build_page_lookup(records: list[PageRecord]) -> dict[tuple[str, int], PageRecord]:
    return {record.key(): record for record in records}
