from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .manifest import load_page_manifest
from .render import PageRenderer
from .schemas import PageRecord
from .utils import dump_jsonl, ensure_dir, write_text_atomic


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
) -> list[PageRecord]:
    if text_source not in {"ocr", "hybrid"}:
        raise ValueError(f"Unsupported text_source: {text_source}")

    page_records = load_page_manifest(input_manifest_path)
    renderer = PageRenderer(render_cache, dpi=dpi)
    output_records: list[PageRecord] = []

    for page_record in tqdm(page_records, desc=f"Building {text_source} OCR manifest"):
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

    dump_jsonl([record.to_dict() for record in output_records], output_path)
    return output_records
