from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.request
from pathlib import Path


def source_file_to_stem(source_file: str) -> str:
    source_file = source_file.strip()
    if not source_file:
        raise ValueError("source_file cannot be empty")
    return Path(source_file).stem


def stem_to_fraser_url(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) != 4 or parts[:2] != ["treasury", "bulletin"]:
        raise ValueError(f"Unsupported source file stem: {stem}")
    year = parts[2]
    month = parts[3]
    return f"https://fraser.stlouisfed.org/files/docs/publications/tbulletin/{year}_{month}_treasurybulletin.pdf"


def collect_required_stems(questions_csv: Path) -> list[str]:
    stems: set[str] = set()
    with questions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for source_file in (row.get("source_files") or "").split():
                if source_file.strip():
                    stems.add(source_file_to_stem(source_file))
    return sorted(stems)


def download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        data = response.read()
    destination.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Treasury Bulletin PDFs required by an OfficeQA CSV.")
    parser.add_argument("--questions-csv", required=True, help="Path to officeqa_pro.csv or a similar OfficeQA CSV.")
    parser.add_argument("--pdf-dir", required=True, help="Directory where PDFs should be stored.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not re-download files that already exist.")
    parser.add_argument("--max-files", type=int, help="Download at most this many PDFs, for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between downloads.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per-download timeout.")
    args = parser.parse_args()

    questions_csv = Path(args.questions_csv)
    pdf_dir = Path(args.pdf_dir)
    stems = collect_required_stems(questions_csv)
    if args.max_files is not None:
        stems = stems[: args.max_files]

    successes = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for index, stem in enumerate(stems, start=1):
        pdf_path = pdf_dir / f"{stem}.pdf"
        if args.skip_existing and pdf_path.exists():
            skipped += 1
            print(f"[{index}/{len(stems)}] skip {pdf_path.name}")
            continue

        url = stem_to_fraser_url(stem)
        try:
            print(f"[{index}/{len(stems)}] download {pdf_path.name}")
            download_file(url, pdf_path, timeout_seconds=args.timeout_seconds)
            successes += 1
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            failures.append((stem, str(exc)))
            print(f"[{index}/{len(stems)}] failed {pdf_path.name}: {exc}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(
        f"done: requested={len(stems)} downloaded={successes} skipped={skipped} failed={len(failures)}"
    )
    if failures:
        print("failed stems:")
        for stem, error in failures[:25]:
            print(f"- {stem}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
