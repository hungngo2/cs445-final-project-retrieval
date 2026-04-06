from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .manifest import load_page_manifest
from .render import PageRenderer
from .schemas import PageRecord
from .utils import dump_json_atomic, dump_jsonl, ensure_dir, write_text_atomic


def _import_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError("pytesseract is required for OCR manifests.") from exc
    return pytesseract


def _ocr_page_image(image_path: Path, *, tesseract_cmd: str | None = None, lang: str = "eng") -> str:
    pytesseract = _import_pytesseract()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return pytesseract.image_to_string(image, lang=lang).strip()


def _ocr_cache_path(cache_dir: str | Path, page_record: PageRecord) -> Path:
    root = ensure_dir(cache_dir)
    return root / page_record.doc_id / f"page_{page_record.page_num:04d}.txt"


def _merge_native_and_ocr_text(native_text: str, ocr_text: str) -> str:
    native_text = native_text.strip()
    ocr_text = ocr_text.strip()
    if not native_text:
        return ocr_text
    if not ocr_text:
        return native_text
    if native_text == ocr_text:
        return native_text
    return f"{native_text}\n{ocr_text}"


def _write_ocr_progress(
    progress_path: str | Path,
    *,
    status: str,
    text_source: str,
    processed_pages: int,
    total_pages: int,
    current_page: str | None = None,
) -> None:
    percent_complete = 0.0 if total_pages == 0 else round((processed_pages / total_pages) * 100.0, 2)
    dump_json_atomic(
        {
            "status": status,
            "text_source": text_source,
            "processed_pages": processed_pages,
            "total_pages": total_pages,
            "percent_complete": percent_complete,
            "current_page": current_page,
        },
        progress_path,
    )


def build_ocr_page_manifest(
    input_manifest_path: str | Path,
    output_path: str | Path,
    render_cache: str | Path,
    ocr_cache_dir: str | Path,
    *,
    text_source: str = "ocr",
    dpi: int = 150,
    tesseract_cmd: str | None = None,
    lang: str = "eng",
    progress_path: str | Path | None = None,
) -> list[PageRecord]:
    if text_source not in {"ocr", "hybrid"}:
        raise ValueError(f"Unsupported text_source: {text_source}")

    page_records = load_page_manifest(input_manifest_path)
    renderer = PageRenderer(render_cache, dpi=dpi)
    output_records: list[PageRecord] = []
    total_pages = len(page_records)

    if progress_path is None:
        progress_path = Path(output_path).with_suffix(".progress.json")

    for index, page_record in enumerate(tqdm(page_records, desc=f"Building {text_source} OCR manifest"), start=1):
        cache_path = _ocr_cache_path(ocr_cache_dir, page_record)
        if cache_path.exists():
            ocr_text = cache_path.read_text(encoding="utf-8").strip()
        else:
            image_path = renderer.render_page(page_record)
            ocr_text = _ocr_page_image(image_path, tesseract_cmd=tesseract_cmd, lang=lang)
            write_text_atomic(ocr_text, cache_path)

        if text_source == "ocr":
            page_text = ocr_text
        else:
            page_text = _merge_native_and_ocr_text(page_record.page_text, ocr_text)

        output_records.append(
            PageRecord(
                doc_id=page_record.doc_id,
                pdf_path=page_record.pdf_path,
                page_num=page_record.page_num,
                page_text=page_text,
                page_width=page_record.page_width,
                page_height=page_record.page_height,
            )
        )

        if index == 1 or index == total_pages or index % 25 == 0:
            _write_ocr_progress(
                progress_path,
                status="running",
                text_source=text_source,
                processed_pages=index,
                total_pages=total_pages,
                current_page=f"{page_record.doc_id}:{page_record.page_num}",
            )

    dump_jsonl([record.to_dict() for record in output_records], output_path)
    _write_ocr_progress(
        progress_path,
        status="completed",
        text_source=text_source,
        processed_pages=total_pages,
        total_pages=total_pages,
        current_page=None,
    )
    return output_records
