from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)?")


def normalize_doc_id(value: str) -> str:
    return Path(value).stem.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_jsonl(records: Iterable[dict], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: str | Path) -> dict | list:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: dict | list, path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def dump_json_atomic(data: dict | list, path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def batched(values: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def stable_topk_indices(scores: Sequence[float], top_k: int) -> list[int]:
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (-item[1], item[0]))
    return [index for index, _ in indexed[:top_k]]
