#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = ROOT / "paper" / "main.tex"
DEFAULT_VENUE_AUDIT = ROOT / "runs" / "venue_submission_compliance_audit.json"
DEFAULT_SHAPE_AUDIT = ROOT / "runs" / "paper_submission_shape_audit.json"
DEFAULT_PDF_AUDIT = ROOT / "runs" / "submission_pdf_shape_audit.json"
DEFAULT_ARCHIVE_AUDIT = ROOT / "runs" / "public_release_archive_audit.json"
DEFAULT_SNAPSHOT_AUDIT = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_RUNBOOK = ROOT / "runs" / "external_validation_runbook.json"
DEFAULT_RECEIPT_TEMPLATE = ROOT / "runs" / "external_validation_receipt_template.json"
DEFAULT_OUT_JSON = ROOT / "runs" / "submission_handoff.json"
DEFAULT_OUT_MD = ROOT / "docs" / "submission_handoff.md"

KEYWORDS = [
    "lottery ticket hypothesis",
    "Bayesian deep learning",
    "posterior modes",
    "network pruning",
    "SGLD",
    "Laplace approximation",
    "model sparsity",
    "reproducibility",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--venue-audit", type=Path, default=DEFAULT_VENUE_AUDIT)
    parser.add_argument("--shape-audit", type=Path, default=DEFAULT_SHAPE_AUDIT)
    parser.add_argument("--pdf-audit", type=Path, default=DEFAULT_PDF_AUDIT)
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_ARCHIVE_AUDIT)
    parser.add_argument("--snapshot-audit", type=Path, default=DEFAULT_SNAPSHOT_AUDIT)
    parser.add_argument("--runbook", type=Path, default=DEFAULT_RUNBOOK)
    parser.add_argument("--receipt-template", type=Path, default=DEFAULT_RECEIPT_TEMPLATE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def tex_block(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\s*\{{", text)
    if not match:
        return ""
    start = match.end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        previous = text[index - 1] if index > 0 else ""
        if char == "{" and previous != "\\":
            depth += 1
        elif char == "}" and previous != "\\":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    return ""


def normalize_tex_text(text: str) -> str:
    replacements = {
        r"\\": " ",
        r"\%": "%",
        r"\_": "_",
        r"\&": "&",
        "~": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def read_paper_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    title = normalize_tex_text(tex_block(text, "title"))
    abstract_match = re.search(
        r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}",
        text,
        flags=re.DOTALL,
    )
    abstract = normalize_tex_text(abstract_match.group("body") if abstract_match else "")
    return {
        "title": title,
        "abstract": abstract,
        "abstract_words": word_count(abstract),
        "author_field": "Anonymous",
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    paper = read_paper_metadata(args.paper)
    venue = load_json(args.venue_audit)
    shape = load_json(args.shape_audit)
    pdf = load_json(args.pdf_audit)
    archive = load_json(args.archive_audit)
    snapshot = load_json(args.snapshot_audit)
    runbook = load_json(args.runbook)
    receipt_template = load_json(args.receipt_template)

    risk_flags: list[str] = []
    if not paper["title"]:
        risk_flags.append("paper_title_missing")
    if not paper["abstract"]:
        risk_flags.append("paper_abstract_missing")
    if int(paper["abstract_words"]) > 250:
        risk_flags.append("abstract_over_250_words")
    if venue.get("content_packet_ready") is not True:
        risk_flags.append("venue_content_packet_not_ready")
    if venue.get("venue_binding_ready") is not True:
        risk_flags.append("venue_binding_not_ready")
    if shape.get("submission_shape_ready") is not True:
        risk_flags.append("paper_submission_shape_not_ready")
    if pdf.get("submission_pdf_ready") is not True:
        risk_flags.append("submission_pdf_not_ready")
    if archive.get("archive_ready") is not True:
        risk_flags.append("public_release_archive_not_ready")
    if snapshot.get("public_repository_snapshot_ready") is not True:
        risk_flags.append("public_repository_snapshot_not_ready")
    if runbook.get("external_validation_runbook_ready") is not True:
        risk_flags.append("external_validation_runbook_not_ready")
    if receipt_template.get("external_validation_receipt_template_ready") is not True:
        risk_flags.append("external_validation_receipt_template_not_ready")

    paper_files = {
        "primary_submission_pdf": "paper/neurips_submission.pdf",
        "main_only_pdf": "paper/main_submission.pdf",
        "full_reproducibility_pdf": "paper/main.pdf",
        "source_tex": "paper/main.tex",
        "checklist_tex": "paper/neurips_checklist.tex",
    }
    supplement_files = {
        "artifact_archive": str(archive.get("archive", "")),
        "artifact_archive_sha256": str(archive.get("archive_sha256", "")),
        "source_repository_snapshot": str(snapshot.get("stage_dir", "")),
        "source_repository_snapshot_commit": str(snapshot.get("git", {}).get("commit", "")),
        "external_validation_runbook": "docs/external_validation_runbook.md",
        "external_receipt_template": "docs/external_validation_receipt_template.md",
        "external_receipt_registry": "docs/external_validation_receipts.json",
    }
    check_commands = [
        "make check",
        "make paper-neurips-check",
        "make container-check",
        "make external-validation-readiness",
    ]
    final_external_commands = [
        ".venv/bin/python scripts/audit_external_validation_readiness.py --strict",
        ".venv/bin/python scripts/verify_research_artifacts.py",
    ]
    return {
        "submission_handoff_ready": not risk_flags,
        "risk_flags": risk_flags,
        "metadata": {
            **paper,
            "keywords": KEYWORDS,
            "venue": "NeurIPS 2026 Main Track",
            "paper_type": "empirical negative result with reproducibility artifact",
        },
        "venue_audit": {
            "content_packet_ready": venue.get("content_packet_ready"),
            "venue_binding_ready": venue.get("venue_binding_ready"),
            "venue_submission_ready": venue.get("venue_submission_ready"),
            "abstract_words": venue.get("abstract_word_count"),
            "main_content_pages_before_references": venue.get(
                "content_pages_before_references"
            ),
            "main_content_page_budget": venue.get("max_pages_before_references"),
            "total_submission_pdf_pages": venue.get("submission_pdf_pages"),
            "open_release_blockers": {
                "checklist_release_risk_flags": venue.get(
                    "checklist_release_risk_flags", []
                ),
                "release_packaging_risk_flags": venue.get(
                    "release_packaging_risk_flags", []
                ),
            },
        },
        "paper_files": paper_files,
        "supplement_files": supplement_files,
        "check_commands": check_commands,
        "final_external_commands": final_external_commands,
        "submission_notes": [
            "Use the NeurIPS-style PDF as the primary submission file.",
            "Keep the code/data availability answer conservative until the public archive, public source repository, external CI, and external GPU-container receipts are filled.",
            "Use scripts/update_external_validation_receipts.py to validate and write externally observed receipt values after the public URLs and GPU image digest exist.",
            "The artifact archive is the supplementary payload; the source-only repository snapshot is the public code seed without large nested run artifacts.",
            "Do not claim top-conference release readiness until the strict external gate passes.",
        ],
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    metadata = payload["metadata"]
    venue = payload["venue_audit"]
    supplement = payload["supplement_files"]
    status = "ready" if payload["submission_handoff_ready"] else "not ready"
    lines = [
        "# Submission Handoff",
        "",
        "This generated handoff collects the fields and files needed for the",
        "submission UI and supplementary-material upload.",
        "",
        f"Handoff status: {status}.",
        f"Venue submission status: {venue['venue_submission_ready']}.",
        "",
        "## Submission Metadata",
        "",
        f"Title: {metadata['title']}",
        "",
        f"Authors: {metadata['author_field']}",
        "",
        f"Venue: {metadata['venue']}",
        "",
        f"Paper type: {metadata['paper_type']}",
        "",
        f"Keywords: {', '.join(metadata['keywords'])}",
        "",
        f"Abstract words: {metadata['abstract_words']}",
        "",
        "Abstract:",
        "",
        metadata["abstract"],
        "",
        "## Venue Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Content packet ready | {venue['content_packet_ready']} |",
        f"| Venue binding ready | {venue['venue_binding_ready']} |",
        f"| Venue submission ready | {venue['venue_submission_ready']} |",
        f"| Main-content pages before references | {venue['main_content_pages_before_references']} |",
        f"| Main-content page budget | {venue['main_content_page_budget']} |",
        f"| Total submission PDF pages | {venue['total_submission_pdf_pages']} |",
        "",
        "## Files To Upload Or Reference",
        "",
        "| Role | Path or value |",
        "| --- | --- |",
    ]
    for key, value in payload["paper_files"].items():
        lines.append(f"| {key} | `{value}` |")
    for key, value in supplement.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Local Checks",
            "",
            "```bash",
            *payload["check_commands"],
            "```",
            "",
            "## Final External Gate",
            "",
            "```bash",
            *payload["final_external_commands"],
            "```",
            "",
            "## Release Blockers",
            "",
        ]
    )
    blockers = venue["open_release_blockers"]
    for key, values in blockers.items():
        if values:
            lines.extend(f"- {key}: {value}" for value in values)
        else:
            lines.append(f"- {key}: none")
    lines.extend(["", "## Submission Notes", ""])
    lines.extend(f"- {note}" for note in payload["submission_notes"])
    lines.extend(
        [
            "",
            "This file is generated by `scripts/build_submission_handoff.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "submission_handoff_ready": payload["submission_handoff_ready"],
                "risk_flags": payload["risk_flags"],
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["submission_handoff_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
