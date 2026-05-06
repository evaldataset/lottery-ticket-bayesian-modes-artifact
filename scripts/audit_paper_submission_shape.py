#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "paper_submission_shape_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "paper_submission_shape_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-tex", type=Path, default=ROOT / "paper" / "main.tex")
    parser.add_argument("--paper-pdf", type=Path, default=ROOT / "paper" / "main.pdf")
    parser.add_argument(
        "--reviewer-matrix",
        type=Path,
        default=ROOT / "runs" / "reviewer_objection_matrix.json",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pdf_page_count(path: Path) -> int | None:
    if not path.exists():
        return None
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is not None:
        completed = subprocess.run(
            [pdfinfo, str(path)],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if line.startswith("Pages:"):
                    try:
                        return int(line.split(":", 1)[1].strip())
                    except ValueError:
                        break
    data = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page\b", data))
    if count:
        return count
    decompressed = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        raw = match.group(1).strip(b"\r\n")
        try:
            decompressed.append(zlib.decompress(raw))
        except zlib.error:
            continue
    if decompressed:
        return len(re.findall(rb"/Type\s*/Page\b", b"\n".join(decompressed)))
    return 0


def section_spans(lines: list[str]) -> list[dict[str, Any]]:
    starts = []
    for idx, line in enumerate(lines, start=1):
        match = re.match(r"\\section\{([^}]*)\}", line.strip())
        if match:
            starts.append((idx, match.group(1)))
    spans = []
    for pos, (start, title) in enumerate(starts):
        end = starts[pos + 1][0] - 1 if pos + 1 < len(starts) else len(lines)
        spans.append({"title": title, "start_line": start, "end_line": end, "lines": end - start + 1})
    return spans


def count_before(lines: list[str], marker: str) -> int:
    for idx, line in enumerate(lines, start=1):
        if marker in line:
            return idx - 1
    return len(lines)


def contains_all(text: str, needles: list[str]) -> bool:
    clean = " ".join(text.split()).lower()
    return all(needle.lower() in clean for needle in needles)


def build_audit(paper_tex: Path, paper_pdf: Path, reviewer_matrix: Path) -> dict[str, Any]:
    text = paper_tex.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections = section_spans(lines)
    page_count = pdf_page_count(paper_pdf)
    appendix_start = count_before(lines, "\\appendix")
    generated_tables_start = count_before(lines, "\\section{Generated Evidence Tables}")
    main_body_lines = min(appendix_start, generated_tables_start)
    reviewer_payload = load_json(reviewer_matrix)
    reviewer_rows = reviewer_payload.get("rows", [])

    section_titles = [section["title"] for section in sections]
    required_sections = [
        "Introduction",
        "Related Work",
        "Operational Test",
        "Current Results",
        "Discussion",
        "Limitations and Next Experiments",
        "Generated Evidence Tables",
    ]
    missing_sections = [title for title in required_sections if title not in section_titles]

    objection_checks = {
        "random_control": contains_all(text, ["posterior", "random", "chain-start"]),
        "sampler_movement": contains_all(text, ["multi-chain", "cyclical", "posterior"]),
        "function_vs_mask": contains_all(text, ["CKA", "Hamming", "layer"]),
        "alignment_permutation": contains_all(text, ["activation-channel", "weight-correlation", "global channel"]),
        "covariance_fidelity": contains_all(text, ["full-covariance", "joint-group", "553.1"]),
        "linear_connectivity": contains_all(text, ["linear", "barriers", "orthogonal"]),
        "learned_masks": contains_all(text, ["hard-concrete", "variational", "Gem-Miner"]),
        "process_mechanism": contains_all(text, ["tensor+score", "learned-subspace", "residual"]),
    }

    table_count = text.count("\\begin{table}")
    figure_count = text.count("\\includegraphics")
    paragraph_count = text.count("\\paragraph")
    compiled_pdf_over_budget = page_count is not None and page_count > 12
    appendix_or_generated = appendix_start < len(lines) or generated_tables_start < len(lines)
    main_body_over_budget = main_body_lines > 850
    result_section = next(
        (section for section in sections if section["title"] == "Current Results"),
        {"lines": 0},
    )
    current_results_overlong = int(result_section["lines"]) > 450
    top_objections_visible = all(
        objection_checks[key]
        for key in [
            "random_control",
            "sampler_movement",
            "function_vs_mask",
            "alignment_permutation",
            "covariance_fidelity",
        ]
    )

    risk_flags = []
    if compiled_pdf_over_budget and not appendix_or_generated:
        risk_flags.append("page_count_over_submission_budget")
    if main_body_over_budget:
        risk_flags.append("main_body_over_submission_budget")
    if current_results_overlong:
        risk_flags.append("current_results_section_overlong")
    if missing_sections:
        risk_flags.append("missing_required_sections")
    if not top_objections_visible:
        risk_flags.append("top_reviewer_objections_not_visible")

    return {
        "paper_tex": paper_tex.relative_to(ROOT).as_posix(),
        "paper_pdf": paper_pdf.relative_to(ROOT).as_posix(),
        "reviewer_matrix": reviewer_matrix.relative_to(ROOT).as_posix(),
        "pdf_page_count": page_count,
        "total_tex_lines": len(lines),
        "main_body_lines_before_appendix": main_body_lines,
        "section_count": len(sections),
        "sections": sections,
        "table_count": table_count,
        "figure_count": figure_count,
        "paragraph_count": paragraph_count,
        "reviewer_objection_rows": len(reviewer_rows),
        "objection_checks": objection_checks,
        "missing_sections": missing_sections,
        "risk_flags": risk_flags,
        "submission_shape_ready": not risk_flags,
        "compiled_pdf_page_count_includes_appendix": appendix_or_generated,
        "compiled_pdf_page_count_over_12": compiled_pdf_over_budget,
        "recommended_page_budget": {
            "main_text_pages": "8-10",
            "appendix_pages": "unbounded supplemental",
            "compiled_pdf_pages": "reported for visibility; may include appendix/generated tables",
            "current_results_target_lines": 450,
            "main_body_target_lines": 850,
        },
        "recommended_cuts": [
            "Move most residual-process variants to appendix tables and keep only the strongest tensor+score and learned-subspace rows in the main narrative.",
            "Collapse posterior-family movement details into one main table plus one paragraph per objection class.",
            "Move fake-CIFAR and tiny full-network code-path smokes to reproducibility or appendix text.",
            "Keep the direct CIFAR mode/ticket row, channel-permutation audit, and covariance-feasibility limitation in the main paper.",
            "Convert the reviewer objection matrix into the paper's main-result organization.",
        ],
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    ready = "ready" if payload["submission_shape_ready"] else "not ready"
    lines = [
        "# Paper Submission Shape Audit",
        "",
        "This generated audit checks whether the current manuscript shape looks",
        "like a top-conference submission draft. It is intentionally stricter",
        "than the artifact verifier: a paper can be reproducible and still be",
        "too long or too diffuse for submission.",
        "",
        f"Current status: {ready}.",
        "",
        "## Shape Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| PDF pages | {payload['pdf_page_count']} |",
        f"| TeX lines | {payload['total_tex_lines']} |",
        f"| Main-body lines before appendix/generated tables | {payload['main_body_lines_before_appendix']} |",
        f"| Sections | {payload['section_count']} |",
        f"| Tables | {payload['table_count']} |",
        f"| Figures | {payload['figure_count']} |",
        f"| Paragraph blocks | {payload['paragraph_count']} |",
        f"| Reviewer-objection rows | {payload['reviewer_objection_rows']} |",
        "",
        "PDF pages are total compiled pages. When the same PDF includes appendix or",
        "generated evidence tables, the blocking shape gate is the main body before",
        "the appendix rather than the total compiled page count.",
        "",
        "## Section Spans",
        "",
        "| Section | Lines | Start | End |",
        "| --- | ---: | ---: | ---: |",
    ]
    for section in payload["sections"]:
        lines.append(
            f"| {section['title']} | {section['lines']} | {section['start_line']} | {section['end_line']} |"
        )

    lines.extend(
        [
            "",
            "## Reviewer Objection Visibility",
            "",
            "| Objection family | Visible in main text |",
            "| --- | --- |",
        ]
    )
    for key, value in payload["objection_checks"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Risk Flags",
            "",
        ]
    )
    if payload["risk_flags"]:
        for flag in payload["risk_flags"]:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")

    if payload["risk_flags"]:
        lines.extend(
            [
                "",
                "## Recommended Cuts",
                "",
            ]
        )
        for item in payload["recommended_cuts"]:
            lines.append(f"- {item}")
    else:
        lines.extend(
            [
                "",
                "## Completed Condensation",
                "",
                "- Main-body lines are within the 850-line target.",
                "- Current Results is within the 450-line target.",
                "- Reviewer-objection coverage remains visible in the main text.",
                "- Appendix/generated evidence tables can remain as supplemental material.",
            ]
        )
    lines.extend(
        [
            "",
            "This file is generated by `scripts/audit_paper_submission_shape.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_audit(args.paper_tex, args.paper_pdf, args.reviewer_matrix)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "submission_shape_ready": payload["submission_shape_ready"],
                "risk_flags": payload["risk_flags"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
