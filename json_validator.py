#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import List, Tuple


def detect_encoding(path: str) -> str:
    """Very simple BOM-based encoding detection."""
    with open(path, "rb") as f:
        start = f.read(4)

    # UTF-8 BOM
    if start.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    # UTF-16 (LE/BE) BOM
    if start.startswith(b"\xff\xfe") or start.startswith(b"\xfe\xff"):
        return "utf-16"

    # Default assumption
    return "utf-8"


def read_lines_any_encoding(path: str) -> List[str]:
    encoding = detect_encoding(path)
    try:
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        print(
            f"Failed to decode '{path}' with detected encoding '{encoding}': {e}",
            file=sys.stderr,
        )
        raise
    return lines


# Pattern to match delimiter/header lines in delimited JSONL files
# Matches:
#   - Lines consisting only of dashes (e.g., "-----")
#   - TMUA section headers (e.g., "TMUA P1 2023", "TMUA P2 2022", "TMUA OTHER")
DELIMITER_PATTERN = re.compile(
    r"^-+$"  # Line of dashes only
    r"|^TMUA\s+(?:P[12]\s+\d{4}|OTHER)$",  # TMUA headers
    re.IGNORECASE,
)


def is_delimiter_line(line: str) -> bool:
    """Check if a line is a delimiter or section header (not JSON data)."""
    stripped = line.strip()
    return bool(DELIMITER_PATTERN.match(stripped))


def validate_lines(lines: List[str]) -> List[Tuple[int, str]]:
    invalid = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue  # ignore completely blank lines
        if is_delimiter_line(stripped):
            continue  # ignore delimiter/header lines (e.g., "-----", "TMUA P1 2023")
        try:
            json.loads(line)
        except Exception as e:
            invalid.append((i, str(e)))
    return invalid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a JSONL file and report invalid lines."
    )
    parser.add_argument(
        "file",
        help="Path to JSONL file, or '-' to read from stdin"
    )
    args = parser.parse_args()

    if args.file == "-":
        lines = sys.stdin.readlines()
    else:
        try:
            lines = read_lines_any_encoding(args.file)
        except OSError as e:
            print(f"Error opening file '{args.file}': {e}", file=sys.stderr)
            return 2

    invalid = validate_lines(lines)

    if not invalid:
        print("All data lines are valid JSON (blank lines and delimiters skipped).")
        return 0

    print(f"Found {len(invalid)} invalid line(s):")
    for line_no, msg in invalid:
        print(f"  Line {line_no} is invalid JSON: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
