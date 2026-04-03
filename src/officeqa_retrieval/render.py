from __future__ import annotations

from pathlib import Path

from .schemas import PageRecord
from .utils import ensure_dir


def _import_fitz():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for page rendering.") from exc
    return fitz


def _import_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for crop generation.") from exc
    return Image


class PageRenderer:
    def __init__(self, cache_dir: str | Path, dpi: int = 150) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.dpi = dpi

    def render_page(self, page_record: PageRecord) -> Path:
        output_path = self.cache_dir / page_record.doc_id / f"page_{page_record.page_num:04d}.png"
        if output_path.exists():
            return output_path

        fitz = _import_fitz()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(page_record.pdf_path) as document:
            page = document.load_page(page_record.page_num - 1)
            scale = self.dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output_path)
        return output_path

    def fixed_2x2_crops(self, page_record: PageRecord) -> list[Path]:
        image_path = self.render_page(page_record)
        crop_dir = self.cache_dir / page_record.doc_id / f"page_{page_record.page_num:04d}_crops"
        expected_paths = [
            crop_dir / "crop_r0_c0.png",
            crop_dir / "crop_r0_c1.png",
            crop_dir / "crop_r1_c0.png",
            crop_dir / "crop_r1_c1.png",
        ]
        if all(path.exists() for path in expected_paths):
            return expected_paths

        Image = _import_image()
        crop_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            x_mid = width // 2
            y_mid = height // 2
            boxes = [
                (0, 0, x_mid, y_mid),
                (x_mid, 0, width, y_mid),
                (0, y_mid, x_mid, height),
                (x_mid, y_mid, width, height),
            ]
            for output_path, box in zip(expected_paths, boxes, strict=True):
                if not output_path.exists():
                    image.crop(box).save(output_path)
        return expected_paths

    def get_image_paths(self, page_record: PageRecord, crop_mode: str) -> list[Path]:
        if crop_mode == "full":
            return [self.render_page(page_record)]
        if crop_mode == "fixed_2x2":
            return self.fixed_2x2_crops(page_record)
        raise ValueError(f"Unsupported crop mode: {crop_mode}")
