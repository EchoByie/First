"""Evidence verification: code, not the model, decides how sure we are.

The model must back every finding with exact quotes from the data it was
shown. This module looks for each quote in that same data:

  ✔ SURE   "I'm sure that X"
           the model says it OBSERVED it, every quote was found in the data,
           and its confidence is not low.

  ? THINK  "I think X, but I'm not sure"
           anything else: an inference, low confidence, or evidence we
           couldn't find.

If ANY quote can't be found, the model may have made it up (hallucinated),
so the finding is downgraded to basis=inferred, confidence=low and tagged
"unverified evidence". What the model originally said is kept, so the audit
log shows what was changed and why.

A limit to be honest about: finding a quote proves the quote exists in the
data, not that the claim drawn from it is correct. That is why only
*observed* findings can be SURE. An inference stays THINK however good its
evidence is.
"""

from __future__ import annotations

import copy
import re
import unicodedata

SURE = "SURE"
THINK = "THINK"

# Quotes shorter than this (after normalising) are too easy to match by
# accident: "1" or "ok" appear almost everywhere. They don't count as proof.
MIN_QUOTE_CHARS = 4

# Characters that render as nothing. They are a known trick for hiding text
# from people, so we strip them before comparing.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

# Curly quotes and dashes that models like to "tidy up" into plain ones.
_LOOKALIKES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
})

# A model may shorten a long quote: "first part ... last part".
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")

# Status of one quote
FOUND = "found"
MISSING = "missing"
TOO_SHORT = "too_short"


def normalise(text: str) -> str:
    """Make two texts comparable while ignoring harmless differences.

    - NFKC turns look-alike Unicode (e.g. full-width letters) into plain forms
    - curly quotes and long dashes become plain ones
    - invisible characters are removed
    - upper/lower case is ignored (casefold)
    - any run of spaces, tabs or newlines counts as one space
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE).translate(_LOOKALIKES).casefold()
    return " ".join(text.split())


def _strip_wrapping(quote: str) -> str:
    """Remove quote marks a model may put around a quote: "abc" -> abc."""
    return quote.strip().strip("\"'`").strip()


def check_quote(quote: str, corpus_normalised: str) -> str:
    """Look for one quote in the (already normalised) data.

    With an ellipsis, every piece must be found, in order.
    """
    pieces = [normalise(p) for p in _ELLIPSIS.split(_strip_wrapping(quote))]
    pieces = [p for p in pieces if p]
    if not pieces or sum(len(p) for p in pieces) < MIN_QUOTE_CHARS:
        return TOO_SHORT
    position = 0
    for piece in pieces:
        if len(pieces) > 1 and len(piece) < MIN_QUOTE_CHARS:
            return TOO_SHORT  # "a ... b" proves nothing
        index = _find_whole(piece, corpus_normalised, position)
        if index == -1:
            return MISSING
        position = index + len(piece)
    return FOUND


def _find_whole(piece: str, corpus: str, start: int) -> int:
    """Like str.find, but the match may not cut a word or number in half.

    Without this, the quote "192.168.1.2" would be "found" inside
    "192.168.1.20", which is a different device.
    """
    index = corpus.find(piece, start)
    while index != -1:
        end = index + len(piece)
        cuts_start = piece[0].isalnum() and _joined(corpus, index - 1, -1)
        cuts_end = piece[-1].isalnum() and _joined(corpus, end, +1)
        if not cuts_start and not cuts_end:
            return index
        index = corpus.find(piece, index + 1)
    return -1


def _joined(corpus: str, i: int, step: int) -> bool:
    """Is the character at corpus[i] still part of the same word/number?

    Letters and digits are. So are . : - when a letter or digit follows
    them, because those join the parts of IP addresses (192.168.1.20),
    MAC addresses (a4:2b:b0) and dates (2024-05-01). Thanks to this, the
    fragment "168.1.20" is not accepted as a quote from "192.168.1.20".
    """
    if not 0 <= i < len(corpus):
        return False
    if corpus[i].isalnum():
        return True
    nxt = i + step
    return corpus[i] in ".:-" and 0 <= nxt < len(corpus) and corpus[nxt].isalnum()


def statement(claim: str, verdict: str) -> str:
    """Turn a claim into the sentence shown to you."""
    claim = claim.strip().rstrip(".")
    # Lower-case the first letter so it reads naturally after "that",
    # but leave acronyms alone ("IP address ..." stays "IP address ...").
    if len(claim) > 1 and claim[0].isupper() and not claim[1].isupper():
        claim = claim[0].lower() + claim[1:]
    if verdict == SURE:
        return f"I'm sure that {claim}."
    return f"I think {claim}, but I'm not sure."


def verify_finding(finding: dict, corpus_normalised: str) -> dict:
    """Return a copy of one finding with a 'verification' section added."""
    result = copy.deepcopy(finding)
    checks = [
        {"quote": quote, "status": check_quote(quote, corpus_normalised)}
        for quote in finding["evidence"]
    ]
    all_found = all(c["status"] == FOUND for c in checks)
    notes = []

    if not all_found:
        bad = sum(c["status"] != FOUND for c in checks)
        notes.append(f"unverified evidence: {bad} of {len(checks)} quote(s) not found in the data")
        # Downgrade. Keep the model's own words so nothing is hidden.
        if finding["basis"] != "inferred" or finding["confidence"] != "low":
            result["basis"] = "inferred"
            result["confidence"] = "low"
            notes.append(
                f"downgraded from basis={finding['basis']}, confidence={finding['confidence']}"
            )

    sure = all_found and result["basis"] == "observed" and result["confidence"] != "low"
    if all_found and not sure:
        if result["basis"] == "inferred":
            notes.append("evidence found, but the claim is an inference")
        else:
            notes.append("evidence found, but the model's confidence is low")

    verdict = SURE if sure else THINK
    result["verification"] = {
        "verdict": verdict,
        "statement": statement(finding["claim"], verdict),
        "evidence_checks": checks,
        "evidence_verified": all_found,
        "model_said": {"basis": finding["basis"], "confidence": finding["confidence"]},
        "notes": notes,
    }
    return result


def verify_output(output: dict, corpus: str) -> dict:
    """Verify every finding in a validated model answer.

    `corpus` must be exactly the data text that was shown to the model,
    because those are the only quotes the model could have honestly copied.
    Returns a new dict; the input is not changed.
    """
    corpus_normalised = normalise(corpus)
    result = copy.deepcopy(output)
    result["findings"] = [verify_finding(f, corpus_normalised) for f in output["findings"]]

    verdicts = [f["verification"] for f in result["findings"]]
    result["verification_summary"] = {
        "total": len(verdicts),
        "sure": sum(v["verdict"] == SURE for v in verdicts),
        "think": sum(v["verdict"] == THINK for v in verdicts),
        "unverified_evidence": sum(not v["evidence_verified"] for v in verdicts),
    }
    return result
