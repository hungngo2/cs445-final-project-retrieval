from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoldReference:
    source_file: str
    doc_id: str
    page_num: int | None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionRecord:
    uid: str
    question: str
    answer: str | None
    difficulty: str | None
    source_docs: tuple[str, ...]
    source_files: tuple[str, ...]
    gold_references: tuple[GoldReference, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gold_references"] = [reference.to_dict() for reference in self.gold_references]
        return data


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    pdf_path: str
    page_num: int
    page_text: str
    page_width: float | None = None
    page_height: float | None = None

    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.page_num)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankedPage:
    uid: str
    doc_id: str
    page_num: int
    score: float
    rank: int
    method: str
    bm25_score: float | None = None
    crop_mode: str = "full"
    matched_crop: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)

    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.page_num)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "score": self.score,
            "rank": self.rank,
            "method": self.method,
            "bm25_score": self.bm25_score,
            "crop_mode": self.crop_mode,
            "matched_crop": self.matched_crop,
            "component_scores": self.component_scores,
        }
