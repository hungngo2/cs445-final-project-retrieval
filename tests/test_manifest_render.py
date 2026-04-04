from __future__ import annotations

import pytest

pytest.importorskip("fitz")
pytest.importorskip("PIL")

from officeqa_retrieval.manifest import build_page_manifest
from officeqa_retrieval.render import PageRenderer


def _make_test_pdf(path) -> None:
    import fitz

    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Federal receipts and defense spending")
    first_page.insert_text((72, 220), "Quarterly summary table values for receipts and expenditures")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Table of monthly values")
    document.save(path)
    document.close()


def test_manifest_extracts_page_text_and_renderer_makes_crops(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "treasury_bulletin_1941_01.pdf"
    _make_test_pdf(pdf_path)

    manifest = build_page_manifest(pdf_dir)
    assert len(manifest) == 2
    assert manifest[0].doc_id == "treasury_bulletin_1941_01"
    assert "Federal receipts" in manifest[0].page_text

    renderer = PageRenderer(tmp_path / "render_cache", dpi=72)
    full_image = renderer.render_page(manifest[0])
    crops = renderer.fixed_2x2_crops(manifest[0])
    layout_crops = renderer.layout_aware_crops(manifest[0])

    assert full_image.exists()
    assert len(crops) == 4
    assert all(crop.exists() for crop in crops)
    assert layout_crops
    assert all(crop.exists() for crop in layout_crops)
    assert any(path.name.startswith("layout_") for path in layout_crops)
