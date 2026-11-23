from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path


CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")


def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARACTERS.sub("", text)
    return WHITESPACE.sub(" ", text).strip()


def discover_text_files(inputs: list[str], output_path: Path) -> list[Path]:
    discovered: list[Path] = []
    output_resolved = output_path.resolve()
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(path.rglob("*.txt"))
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"Input not found: {path}")
        discovered.extend(
            candidate for candidate in candidates if candidate.resolve() != output_resolved
        )
    unique = {path.resolve(): path for path in discovered}
    return [unique[key] for key in sorted(unique, key=str)]


def prepare_corpus(
    inputs: list[str],
    output: str,
    min_chars: int = 20,
    max_chars: int = 4_000,
    deduplicate: bool = True,
) -> tuple[int, int]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = discover_text_files(inputs, destination)
    if not files:
        raise ValueError("No .txt corpus files were found")

    fingerprints: set[bytes] = set()
    kept = 0
    rejected = 0
    with destination.open("w", encoding="utf-8", newline="\n") as writer:
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as reader:
                for raw_line in reader:
                    line = clean_line(raw_line)
                    if len(line) < min_chars or len(line) > max_chars:
                        rejected += 1
                        continue
                    if deduplicate:
                        digest = hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()
                        if digest in fingerprints:
                            rejected += 1
                            continue
                        fingerprints.add(digest)
                    writer.write(line + "\n")
                    kept += 1
    return kept, rejected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and combine English/Hindi corpora")
    parser.add_argument("--input", nargs="+", required=True, help="Files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=4_000)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kept, rejected = prepare_corpus(
        args.input,
        args.output,
        args.min_chars,
        args.max_chars,
        not args.no_deduplicate,
    )
    print(f"Corpus written to {args.output}")
    print(f"Kept lines: {kept:,} | rejected/duplicate lines: {rejected:,}")


if __name__ == "__main__":
    main()

