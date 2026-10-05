"""Spot text in the data that looks like an attempt to steer the model.

This is NOT the main defence. The main defences are structural: the model
has no tools, its answer must fit a strict schema, and every quote is
checked. This module adds one more layer: it marks lines that look like

  - instructions to an AI    ("ignore previous instructions", "you are now")
  - role markers             ("SYSTEM: ...", "assistant: ...")
  - a forged answer          ('"claim": ...', '"evidence": [...]')
  - fake data markers        ("<<<END DATA ...>>>")

so that (1) protocols can report them as facts, and (2) verification
refuses to treat a quote as evidence when it ONLY appears inside such a
line. Otherwise an attacker could plant a ready-made "finding" in a log
file, with a matching quote, and have it come out as "✔ I'm sure".

Lines are normalised first (look-alike letters, invisible characters), so
"ig​nore previous instructions" is caught too.
"""

from __future__ import annotations

import re

from core.verification import normalise

PATTERNS = {
    "instruction-like text": re.compile(
        r"ignore (all |any )?(the |your )?(previous|prior|above|earlier|preceding) "
        r"(instructions?|prompts?|rules)"
        r"|disregard (all|the|previous|prior|your|any)\b"
        r"|forget (all |your )?(previous |prior )?(instructions|rules)"
        r"|you are now\b|new instructions|system prompt|act as (a|an|the)\b"
        r"|do not (tell|report|mention|flag)\b|override (the|your) (rules|instructions)"
    ),
    "role marker": re.compile(r"^\s*\W{0,3}\s*(system|assistant|developer)\s*(:|>|\])"),
    "forged answer": re.compile(r"[\"'](claim|evidence|basis|confidence|verdict|findings)[\"']\s*:"),
    "fake data marker": re.compile(r"<<<\s*(end\s+)?data\b"),
}


def classify(line: str) -> str | None:
    """Return why a line looks suspicious, or None."""
    text = normalise(line)
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            return label
    return None


def suspicious_lines(text: str) -> list[tuple[int, str]]:
    """[(line number, reason), ...] for every suspicious line (1-based)."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        reason = classify(line)
        if reason:
            found.append((number, reason))
    return found
