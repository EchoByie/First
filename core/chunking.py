"""Split big inputs into parts ("chunks") the worker model can read.

Rules:
  - Cut only between lines, so a log line or CSV row is never split in two.
    (A single line longer than a whole chunk is the one exception: it is
    cut into pieces, and each piece is marked as a continuation.)
  - Every chunk remembers which lines of the original it covers, so notes
    can say "lines 120-180" and you can find the spot yourself.
  - There is a maximum number of chunks (max_chunks in the manifest).
    Better to refuse clearly than to run a small model for hours.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    index: int          # 1-based
    total: int
    first_line: int     # 1-based line numbers in the original text
    last_line: int
    text: str

    @property
    def label(self) -> str:
        lines = (f"line {self.first_line}" if self.first_line == self.last_line
                 else f"lines {self.first_line}-{self.last_line}")
        return f"PART {self.index} of {self.total} ({lines} of the original content)"


class TooManyChunks(Exception):
    pass


def split_text(text: str, chunk_size: int, max_chunks: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pieces: list[tuple[int, int, str]] = []    # (first_line, last_line, text)
    current: list[str] = []
    current_len = 0
    first = 1

    def flush(last_line: int) -> None:
        nonlocal current, current_len
        if current:
            pieces.append((first, last_line, "\n".join(current)))
        current, current_len = [], 0

    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > chunk_size:
            # An enormous single line: close the current chunk, then slice it.
            flush(number - 1)
            for start in range(0, len(line), chunk_size):
                part = line[start:start + chunk_size]
                marker = "" if start == 0 else "[continued] "
                pieces.append((number, number, marker + part))
            first = number + 1
            continue
        added = len(line) + (1 if current else 0)   # +1 for the newline
        if current and current_len + added > chunk_size:
            flush(number - 1)
            first = number
            added = len(line)
        current.append(line)
        current_len += added
    flush(len(text.splitlines()))

    if len(pieces) > max_chunks:
        raise TooManyChunks(
            f"input would need {len(pieces)} parts of {chunk_size:,} characters; "
            f"this protocol allows at most {max_chunks}")
    return [Chunk(i, len(pieces), a, b, t) for i, (a, b, t) in enumerate(pieces, start=1)]
