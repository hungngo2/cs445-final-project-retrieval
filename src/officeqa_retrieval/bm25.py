from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from .schemas import PageRecord, QuestionRecord, RankedPage
from .utils import tokenize


def _import_bm25():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("rank-bm25 is required to build the retrieval index.") from exc
    return BM25Okapi


@dataclass
class PageBm25Index:
    page_records: list[PageRecord]
    tokenized_corpus: list[list[str]]
    bm25: object

    @classmethod
    def build(cls, page_records: list[PageRecord]) -> "PageBm25Index":
        if not page_records:
            raise ValueError("page_records cannot be empty")
        tokenized_corpus = [tokenize(record.page_text) for record in page_records]
        bm25_cls = _import_bm25()
        bm25 = bm25_cls(tokenized_corpus)
        return cls(page_records=page_records, tokenized_corpus=tokenized_corpus, bm25=bm25)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(
                {
                    "page_records": [record.to_dict() for record in self.page_records],
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "PageBm25Index":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        page_records = [PageRecord(**row) for row in payload["page_records"]]
        return cls(
            page_records=page_records,
            tokenized_corpus=payload["tokenized_corpus"],
            bm25=payload["bm25"],
        )

    def search(self, question: str, uid: str, top_k: int = 50) -> list[RankedPage]:
        tokens = tokenize(question)
        raw_scores = list(self.bm25.get_scores(tokens))
        query_token_set = set(tokens)
        scored_rows = []
        for index, score in enumerate(raw_scores):
            overlap = len(query_token_set.intersection(self.tokenized_corpus[index]))
            scored_rows.append((index, float(score), overlap))
        # Small corpora can yield tied BM25 scores of 0.0, so use lexical overlap
        # as a deterministic secondary signal before falling back to corpus order.
        scored_rows.sort(key=lambda item: (-item[1], -item[2], item[0]))
        results: list[RankedPage] = []
        for rank, (index, score, overlap) in enumerate(scored_rows[:top_k], start=1):
            record = self.page_records[index]
            results.append(
                RankedPage(
                    uid=uid,
                    doc_id=record.doc_id,
                    page_num=record.page_num,
                    score=score,
                    rank=rank,
                    method="bm25",
                    bm25_score=score,
                    component_scores={"token_overlap": float(overlap)},
                )
            )
        return results

    def batch_search(self, questions: list[QuestionRecord], top_k: int = 50) -> dict[str, list[RankedPage]]:
        return {question.uid: self.search(question.question, uid=question.uid, top_k=top_k) for question in questions}
