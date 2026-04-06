from __future__ import annotations

from pathlib import Path

from PIL import Image

from officeqa_retrieval.ocr_manifest import build_ocr_page_manifest
from officeqa_retrieval.schemas import PageRecord
from officeqa_retrieval.utils import dump_jsonl


def _write_input_manifest(path: Path, *, page_text: str = "native text") -> None:
    records = [
        PageRecord(
            doc_id="doc_a",
            pdf_path="/tmp/doc_a.pdf",
            page_num=1,
            page_text=page_text,
            page_width=100.0,
            page_height=200.0,
        )
    ]
    dump_jsonl([record.to_dict() for record in records], path)


def test_build_ocr_page_manifest_uses_cached_text(tmp_path, monkeypatch) -> None:
    input_manifest = tmp_path / "page_manifest.jsonl"
    output_manifest = tmp_path / "page_manifest_ocr.jsonl"
    render_cache = tmp_path / "render_cache"
    ocr_cache_dir = tmp_path / "ocr_cache"
    _write_input_manifest(input_manifest)

    cache_path = ocr_cache_dir / "doc_a" / "page_0001.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached ocr text", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OCR should not run when cache is present")

    monkeypatch.setattr("officeqa_retrieval.ocr_manifest._ocr_page_image", fail_if_called)

    records = build_ocr_page_manifest(
        input_manifest_path=input_manifest,
        output_path=output_manifest,
        render_cache=render_cache,
        ocr_cache_dir=ocr_cache_dir,
        text_source="ocr",
    )

    assert records[0].page_text == "cached ocr text"
    assert output_manifest.exists()


def test_build_ocr_page_manifest_hybrid_merges_native_and_ocr(tmp_path, monkeypatch) -> None:
    input_manifest = tmp_path / "page_manifest.jsonl"
    output_manifest = tmp_path / "page_manifest_hybrid.jsonl"
    render_cache = tmp_path / "render_cache"
    ocr_cache_dir = tmp_path / "ocr_cache"
    _write_input_manifest(input_manifest, page_text="native text")

    image_path = tmp_path / "page.png"
    Image.new("RGB", (32, 32), color="white").save(image_path)

    monkeypatch.setattr(
        "officeqa_retrieval.render.PageRenderer.render_page",
        lambda self, page_record: image_path,
    )
    monkeypatch.setattr(
        "officeqa_retrieval.ocr_manifest._ocr_page_image",
        lambda image_path, **kwargs: "ocr text",
    )

    records = build_ocr_page_manifest(
        input_manifest_path=input_manifest,
        output_path=output_manifest,
        render_cache=render_cache,
        ocr_cache_dir=ocr_cache_dir,
        text_source="hybrid",
    )

    assert records[0].page_text == "native text\nocr text"
    assert (ocr_cache_dir / "doc_a" / "page_0001.txt").exists()
