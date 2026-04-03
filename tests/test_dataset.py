from __future__ import annotations

import csv

from officeqa_retrieval.dataset import build_gold_references, extract_page_number, load_questions


def test_extract_page_number_from_query_parameter() -> None:
    source_url = "https://example.com/document.pdf?page=15&deep=true"
    assert extract_page_number(source_url) == 15


def test_build_gold_references_repeats_single_file_for_multiple_urls() -> None:
    references = build_gold_references(
        source_files=("treasury_bulletin_1941_01.txt",),
        source_urls=(
            "https://example.com/a?page=5",
            "https://example.com/a?page=7",
        ),
    )
    assert [reference.page_num for reference in references] == [5, 7]
    assert all(reference.doc_id == "treasury_bulletin_1941_01" for reference in references)


def test_load_questions_reads_officeqa_schema(tmp_path) -> None:
    csv_path = tmp_path / "officeqa.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uid", "question", "answer", "source_docs", "source_files", "difficulty"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uid": "UID0001",
                "question": "What is the value?",
                "answer": "42",
                "source_docs": "https://example.com/a?page=3",
                "source_files": "treasury_bulletin_1941_01.txt",
                "difficulty": "easy",
            }
        )

    questions = load_questions(csv_path)
    assert len(questions) == 1
    assert questions[0].uid == "UID0001"
    assert questions[0].gold_references[0].page_num == 3
