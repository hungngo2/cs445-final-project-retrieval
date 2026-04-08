from __future__ import annotations

import inspect
from pathlib import Path

from tqdm import tqdm

from .manifest import load_page_manifest
from .render import PageRenderer
from .schemas import PageRecord
from .utils import dump_json_atomic, dump_jsonl, ensure_dir, write_text_atomic


_OCR_ENGINE_CACHE: dict[tuple[str, str | None], object] = {}


def _import_paddleocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError("paddleocr is required for OCR manifests.") from exc
    return PaddleOCR


def _build_paddleocr_engine(*, lang: str = "en", device: str | None = None):
    cache_key = (lang, device)
    if cache_key in _OCR_ENGINE_CACHE:
        return _OCR_ENGINE_CACHE[cache_key]

    paddleocr_cls = _import_paddleocr()
    signature = inspect.signature(paddleocr_cls)
    kwargs: dict[str, object] = {}
    if "lang" in signature.parameters:
        kwargs["lang"] = lang
    if "use_doc_orientation_classify" in signature.parameters:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in signature.parameters:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in signature.parameters:
        kwargs["use_textline_orientation"] = False
    if "device" in signature.parameters and device:
        kwargs["device"] = device
    elif "use_gpu" in signature.parameters:
        kwargs["use_gpu"] = bool(device and device.startswith("gpu"))
    if "show_log" in signature.parameters:
        kwargs["show_log"] = False

    engine = paddleocr_cls(**kwargs)
    _OCR_ENGINE_CACHE[cache_key] = engine
    return engine


def _extract_paddle_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        if "rec_texts" in result:
            return " ".join(str(text).strip() for text in result["rec_texts"] if str(text).strip()).strip()
        if "res" in result:
            return _extract_paddle_text(result["res"])
    if hasattr(result, "res"):
        return _extract_paddle_text(result.res)
    if isinstance(result, (list, tuple)):
        text_parts: list[str] = []
        for item in result:
            text = _extract_paddle_text(item)
            if text:
                text_parts.append(text)
        return " ".join(text_parts).strip()
    return ""


def _ocr_page_image(image_path: Path, *, lang: str = "en", device: str | None = None) -> str:
    engine = _build_paddleocr_engine(lang=lang, device=device)
    if hasattr(engine, "predict"):
        result = engine.predict(str(image_path))
        return _extract_paddle_text(result)
    if hasattr(engine, "ocr"):
        result = engine.ocr(str(image_path), cls=False)
        return _extract_paddle_text(result)
    raise AttributeError("Unsupported PaddleOCR engine interface: expected predict() or ocr().")


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
    device: str | None = None,
    lang: str = "en",
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
            ocr_text = _ocr_page_image(image_path, lang=lang, device=device)
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
