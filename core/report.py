"""Save a run's result as JSON (for programs) and Markdown (for people).

Files go to reports/<protocol>/<run_id>.json and .md.

The Markdown puts the code's verdict first: what the console is SURE of,
then what it only THINKS, then anything it couldn't verify. Model text is
untrusted, so it is flattened onto one line and evidence goes inside code
spans. A model can't inject headings or links into your report that way.
"""

from __future__ import annotations

import json
from pathlib import Path


def _flat(text) -> str:
    """One line, no Markdown tricks: collapse whitespace, neutralise |, <, >."""
    text = " ".join(str(text).split())
    return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _code(text) -> str:
    """Show text inside a Markdown code span, safely."""
    return "`" + " ".join(str(text).split()).replace("`", "'") + "`"


def to_markdown(report: dict) -> str:
    lines = [f"# {_flat(report['title'])}", ""]
    lines += [
        f"- **Run:** {_code(report['run_id'])}  ",
        f"- **Status:** {report['status']}  ",
        f"- **When (UTC):** {report['created_at']}  ",
        f"- **Input:** {_flat(report['input'].get('description', 'none'))}  ",
    ]
    for role, model in report.get("models", {}).items():
        lines.append(f"- **{role}:** {_code(model)}  ")
    lines.append("")

    if report["status"] != "ok":
        lines += ["## ⚠ Run did not complete", ""]
        for error in report.get("errors", []):
            lines.append(f"- {_flat(error)}")
        lines.append("")

    answer = report.get("answer")
    if answer:
        lines += ["## Summary", "", _flat(answer["summary"]), ""]
        vs = answer["verification_summary"]
        lines += [
            f"**Findings:** {vs['total']} — ✔ sure: {vs['sure']}, "
            f"? not sure: {vs['think']}, unverified evidence: {vs['unverified_evidence']}",
            "",
        ]
        sure = [f for f in answer["findings"] if f["verification"]["verdict"] == "SURE"]
        think = [f for f in answer["findings"] if f["verification"]["verdict"] != "SURE"]
        for heading, group in (("✔ Sure", sure), ("? Not sure", think)):
            if not group:
                continue
            lines += [f"## {heading}", ""]
            for f in group:
                v = f["verification"]
                label = f" _{_flat(f['kind'])}_" if f.get("kind") else ""
                subject = f" ({_flat(f['subject'])})" if f.get("subject") else ""
                lines.append(f"- **{_flat(v['statement'])}**{subject}{label}")
                lines.append(f"  - basis: {f['basis']}, confidence: {f['confidence']}")
                for check in v["evidence_checks"]:
                    mark = "✔" if check["status"] == "found" else "✗"
                    lines.append(f"  - {mark} evidence {_code(check['quote'])} ({check['status']})")
                for note in v["notes"]:
                    lines.append(f"  - note: {_flat(note)}")
            lines.append("")

    caveats = report.get("caveats", [])
    if caveats:
        lines += ["## Caveats", ""] + [f"- {_flat(c)}" for c in caveats] + [""]
    return "\n".join(lines)


def save_report(report: dict, reports_dir: Path) -> tuple[Path, Path]:
    folder = Path(reports_dir) / report["protocol"]
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{report['run_id']}.json"
    md_path = folder / f"{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
