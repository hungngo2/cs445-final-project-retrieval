from __future__ import annotations

from pathlib import Path
from typing import Iterable

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

    @staticmethod
    def _box_area(box: tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @staticmethod
    def _intersection_area(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        x0 = max(first[0], second[0])
        y0 = max(first[1], second[1])
        x1 = min(first[2], second[2])
        y1 = min(first[3], second[3])
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @classmethod
    def _iou(cls, first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        intersection = cls._intersection_area(first, second)
        if intersection <= 0:
            return 0.0
        union = cls._box_area(first) + cls._box_area(second) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @classmethod
    def _coverage(cls, first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        area = cls._box_area(first)
        if area <= 0:
            return 0.0
        return cls._intersection_area(first, second) / area

    @staticmethod
    def _collect_block_text(block: dict) -> str:
        text_parts: list[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if text:
                    text_parts.append(text)
        return " ".join(text_parts).strip()

    @classmethod
    def _merge_text_boxes(
        cls,
        boxes: Iterable[tuple[float, float, float, float]],
        horizontal_gap: float = 24.0,
        vertical_gap: float = 20.0,
    ) -> list[tuple[float, float, float, float]]:
        merged: list[list[float]] = []
        for box in sorted(boxes, key=lambda item: (item[1], item[0], item[3], item[2])):
            x0, y0, x1, y1 = box
            box_height = max(1.0, y1 - y0)
            merged_into_existing = False
            for existing in merged:
                ex0, ey0, ex1, ey1 = existing
                vertical_overlap = min(y1, ey1) - max(y0, ey0)
                horizontal_overlap = min(x1, ex1) - max(x0, ex0)
                close_vertically = abs(y0 - ey1) <= max(vertical_gap, 0.25 * min(box_height, ey1 - ey0))
                close_horizontally = abs(x0 - ex1) <= horizontal_gap or abs(ex0 - x1) <= horizontal_gap
                overlaps = vertical_overlap > 0 or horizontal_overlap > 0
                if overlaps or (close_vertically and horizontal_overlap > -horizontal_gap) or (
                    close_horizontally and vertical_overlap > -vertical_gap
                ):
                    existing[0] = min(ex0, x0)
                    existing[1] = min(ey0, y0)
                    existing[2] = max(ex1, x1)
                    existing[3] = max(ey1, y1)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged.append([x0, y0, x1, y1])
        return [tuple(box) for box in merged]

    def _layout_crop_specs(self, page_record: PageRecord) -> list[tuple[str, tuple[int, int, int, int]]]:
        fitz = _import_fitz()
        scale = self.dpi / 72.0

        with fitz.open(page_record.pdf_path) as document:
            page = document.load_page(page_record.page_num - 1)
            page_rect = page.rect
            page_width = float(page_rect.width)
            page_height = float(page_rect.height)
            page_area = max(1.0, page_width * page_height)

            candidate_rows: list[dict] = []

            if hasattr(page, "find_tables"):
                try:
                    table_finder = page.find_tables()
                    for index, table in enumerate(getattr(table_finder, "tables", []) or []):
                        bbox = tuple(float(value) for value in table.bbox)
                        candidate_rows.append(
                            {
                                "kind": "table",
                                "index": index,
                                "bbox": bbox,
                                "priority": 3,
                                "area": self._box_area(bbox),
                            }
                        )
                except Exception:
                    pass

            page_dict = page.get_text("dict")
            text_boxes: list[tuple[float, float, float, float]] = []

            for block in page_dict.get("blocks", []):
                bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
                area = self._box_area(bbox)
                if area <= 0:
                    continue

                area_ratio = area / page_area
                block_type = block.get("type")
                if block_type == 1:
                    if area_ratio >= 0.01:
                        candidate_rows.append(
                            {
                                "kind": "figure",
                                "index": len([row for row in candidate_rows if row["kind"] == "figure"]),
                                "bbox": bbox,
                                "priority": 2,
                                "area": area,
                            }
                        )
                    continue

                if block_type != 0:
                    continue

                text = self._collect_block_text(block)
                if not text:
                    continue
                if area_ratio < 0.004 and len(text) < 40:
                    continue
                text_boxes.append(bbox)

            merged_text_boxes = self._merge_text_boxes(text_boxes)
            merged_text_boxes.sort(key=self._box_area, reverse=True)
            for index, bbox in enumerate(merged_text_boxes[:4]):
                candidate_rows.append(
                    {
                        "kind": "text",
                        "index": index,
                        "bbox": bbox,
                        "priority": 1,
                        "area": self._box_area(bbox),
                    }
                )

        candidate_rows.sort(key=lambda row: (-row["priority"], -row["area"], row["bbox"][1], row["bbox"][0]))
        selected_rows: list[dict] = []
        for candidate in candidate_rows:
            bbox = candidate["bbox"]
            area_ratio = candidate["area"] / page_area
            if area_ratio < 0.01:
                continue
            if area_ratio > 0.98:
                continue
            if any(
                self._iou(bbox, existing["bbox"]) >= 0.8
                or self._coverage(bbox, existing["bbox"]) >= 0.9
                or self._coverage(existing["bbox"], bbox) >= 0.95
                for existing in selected_rows
            ):
                continue
            selected_rows.append(candidate)
            if len(selected_rows) >= 8:
                break

        crop_specs: list[tuple[str, tuple[int, int, int, int]]] = []
        x_margin = max(12.0, page_width * 0.02)
        y_margin = max(12.0, page_height * 0.02)
        max_pixel_width = max(1, int(round(page_width * scale)))
        max_pixel_height = max(1, int(round(page_height * scale)))

        for candidate in selected_rows:
            x0, y0, x1, y1 = candidate["bbox"]
            pixel_box = (
                max(0, int(round((x0 - x_margin) * scale))),
                max(0, int(round((y0 - y_margin) * scale))),
                min(max_pixel_width, int(round((x1 + x_margin) * scale))),
                min(max_pixel_height, int(round((y1 + y_margin) * scale))),
            )
            if pixel_box[2] - pixel_box[0] < 32 or pixel_box[3] - pixel_box[1] < 32:
                continue
            crop_specs.append((f"layout_{candidate['kind']}_{candidate['index']:02d}.png", pixel_box))

        if crop_specs:
            return crop_specs
        return [("layout_full_page.png", (0, 0, max_pixel_width, max_pixel_height))]

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

    def layout_aware_crops(self, page_record: PageRecord) -> list[Path]:
        image_path = self.render_page(page_record)
        crop_dir = self.cache_dir / page_record.doc_id / f"page_{page_record.page_num:04d}_crops"
        crop_specs = self._layout_crop_specs(page_record)
        expected_paths = [crop_dir / file_name for file_name, _ in crop_specs]
        if all(path.exists() for path in expected_paths):
            return expected_paths

        Image = _import_image()
        crop_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for output_path, (_, box) in zip(expected_paths, crop_specs, strict=True):
                if not output_path.exists():
                    image.crop(box).save(output_path)
        return expected_paths

    def get_image_paths(self, page_record: PageRecord, crop_mode: str) -> list[Path]:
        if crop_mode == "full":
            return [self.render_page(page_record)]
        if crop_mode == "fixed_2x2":
            return self.fixed_2x2_crops(page_record)
        if crop_mode == "layout_aware":
            return self.layout_aware_crops(page_record)
        raise ValueError(f"Unsupported crop mode: {crop_mode}")
