from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .schemas import GoldReference, QuestionRecord
from .utils import dump_json, normalize_doc_id


def split_multi_value_field(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    stripped = value.strip()
    if not stripped:
        return ()
    return tuple(part for part in stripped.split() if part)


def extract_page_number(source_url: str | None) -> int | None:
    if not source_url:
        return None

    parsed = urlparse(source_url)
    query = parse_qs(parsed.query)
    if "page" in query and query["page"]:
        try:
            return int(query["page"][0])
        except ValueError:
            return None

    if parsed.fragment.startswith("page="):
        try:
            return int(parsed.fragment.split("=", maxsplit=1)[1])
        except ValueError:
            return None

    return None


def _align_sources(source_files: tuple[str, ...], source_urls: tuple[str, ...]) -> list[tuple[str | None, str | None]]:
    if not source_files and not source_urls:
        return []
    if len(source_files) == len(source_urls):
        return list(zip(source_files, source_urls, strict=True))
    if len(source_files) == 1 and source_urls:
        return [(source_files[0], source_url) for source_url in source_urls]
    if len(source_urls) == 1 and source_files:
        return [(source_file, source_urls[0]) for source_file in source_files]

    count = max(len(source_files), len(source_urls))
    aligned: list[tuple[str | None, str | None]] = []
    for index in range(count):
        source_file = source_files[index] if index < len(source_files) else None
        source_url = source_urls[index] if index < len(source_urls) else None
        aligned.append((source_file, source_url))
    return aligned


def build_gold_references(source_files: tuple[str, ...], source_urls: tuple[str, ...]) -> tuple[GoldReference, ...]:
    references: list[GoldReference] = []
    for source_file, source_url in _align_sources(source_files, source_urls):
        if source_file is None and source_url is None:
            continue
        resolved_file = source_file or ""
        doc_id = normalize_doc_id(resolved_file) if resolved_file else ""
        references.append(
            GoldReference(
                source_file=resolved_file,
                doc_id=doc_id,
                page_num=extract_page_number(source_url),
                source_url=source_url,
            )
        )
    return tuple(references)


def load_questions(path: str | Path) -> list[QuestionRecord]:
    question_path = Path(path)
    questions: list[QuestionRecord] = []
    with question_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_urls = split_multi_value_field(row.get("source_docs"))
            source_files = split_multi_value_field(row.get("source_files"))
            questions.append(
                QuestionRecord(
                    uid=row["uid"].strip(),
                    question=row["question"].strip(),
                    answer=(row.get("answer") or "").strip() or None,
                    difficulty=(row.get("difficulty") or "").strip() or None,
                    source_docs=source_urls,
                    source_files=source_files,
                    gold_references=build_gold_references(source_files, source_urls),
                )
            )
    return questions


def collect_doc_ids(questions: list[QuestionRecord]) -> set[str]:
    doc_ids: set[str] = set()
    for question in questions:
        for reference in question.gold_references:
            if reference.doc_id:
                doc_ids.add(reference.doc_id)
    return doc_ids


def build_sanity_subset(questions: list[QuestionRecord], size: int = 10) -> list[dict]:
    if size <= 0:
        raise ValueError("size must be positive")

    easy_questions = [question for question in questions if question.difficulty == "easy"]
    hard_questions = [question for question in questions if question.difficulty == "hard"]
    selected: list[QuestionRecord] = []

    easy_target = min(len(easy_questions), size // 2)
    hard_target = min(len(hard_questions), size - easy_target)

    selected.extend(easy_questions[:easy_target])
    selected.extend(hard_questions[:hard_target])

    if len(selected) < size:
        remaining = [question for question in questions if question not in selected]
        selected.extend(remaining[: size - len(selected)])

    return [question.to_dict() for question in selected[:size]]


def save_sanity_subset(questions: list[QuestionRecord], path: str | Path, size: int = 10) -> None:
    dump_json(build_sanity_subset(questions, size=size), path)
