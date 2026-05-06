#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from audit_paper_submission_shape import ROOT, pdf_page_count


DEFAULT_OUT_JSON = ROOT / "runs" / "venue_submission_compliance_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "venue_submission_compliance_audit.md"
DEFAULT_TARGET_VENUE = "NeurIPS 2026 Main Track"
DEFAULT_VENUE_PROFILE = "neurips_2026_main"
DEFAULT_EXTERNAL_VALIDATION_AUDIT = ROOT / "runs" / "external_validation_readiness_audit.json"
DEFAULT_LOCAL_GPU_VALIDATION = ROOT / "runs" / "local_gpu_container_validation.json"

RELEASE_METADATA_DOCS = {
    "compute_resources_not_reported": (
        "compute_resource_accounting",
        ROOT / "docs" / "compute_resource_accounting.md",
        [
            "Compute Resource Accounting",
            "NVIDIA GeForce RTX 5090",
            "Torch CUDA 13.0",
            "553.1 GiB",
            "wall-clock",
        ],
    ),
    "existing_asset_licenses_not_reported": (
        "asset_license_inventory",
        ROOT / "docs" / "asset_license_inventory.md",
        [
            "Asset License Inventory",
            "MIT License",
            "LICENSE",
            "MNIST",
            "Fashion-MNIST",
            "CIFAR-10",
            "CIFAR-100",
            "Raw benchmark datasets",
        ],
    ),
    "new_asset_metadata_not_reported": (
        "new_asset_inventory",
        ROOT / "docs" / "new_asset_inventory.md",
        [
            "New Asset Inventory",
            "public_release_manifest",
            "mask_artifacts.npz",
            "paper/neurips_submission.pdf",
            "data/",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-tex", type=Path, default=ROOT / "paper" / "main.tex")
    parser.add_argument(
        "--submission-pdf",
        type=Path,
        default=ROOT / "paper" / "neurips_submission.pdf",
    )
    parser.add_argument("--official-style", type=Path, default=ROOT / "paper" / "neurips_2026.sty")
    parser.add_argument(
        "--checklist-tex",
        type=Path,
        default=ROOT / "paper" / "neurips_checklist.tex",
    )
    parser.add_argument(
        "--compute-resource-doc",
        type=Path,
        default=ROOT / "docs" / "compute_resource_accounting.md",
    )
    parser.add_argument(
        "--asset-license-doc",
        type=Path,
        default=ROOT / "docs" / "asset_license_inventory.md",
    )
    parser.add_argument(
        "--new-asset-doc",
        type=Path,
        default=ROOT / "docs" / "new_asset_inventory.md",
    )
    parser.add_argument(
        "--external-validation-audit",
        type=Path,
        default=DEFAULT_EXTERNAL_VALIDATION_AUDIT,
    )
    parser.add_argument(
        "--local-gpu-validation",
        type=Path,
        default=DEFAULT_LOCAL_GPU_VALIDATION,
    )
    parser.add_argument("--target-venue", default=DEFAULT_TARGET_VENUE)
    parser.add_argument("--venue-profile", default=DEFAULT_VENUE_PROFILE)
    parser.add_argument("--max-pages", type=int, default=9)
    parser.add_argument("--max-abstract-words", type=int, default=250)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def latex_group(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\{{(.*?)\}}", text, flags=re.S)
    return " ".join(match.group(1).split()) if match else ""


def abstract_text(text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S)
    return " ".join(match.group(1).split()) if match else ""


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\\-]*", text))


def document_class(text: str) -> str:
    match = re.search(r"\\documentclass(?:\[[^\]]*\])?\{([^}]*)\}", text)
    return match.group(1) if match else ""


def command_order(text: str, commands: list[str]) -> bool:
    positions = []
    for command in commands:
        position = text.find(command)
        if position < 0:
            return False
        positions.append(position)
    return positions == sorted(positions)


def page_texts(path: Path, total_pages: int | None) -> list[str] | None:
    if total_pages is None or total_pages <= 0 or not path.exists():
        return None
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return None
    pages = []
    for page in range(1, total_pages + 1):
        completed = subprocess.run(
            [pdftotext, "-layout", "-f", str(page), "-l", str(page), str(path), "-"],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            return None
        pages.append(completed.stdout)
    return pages


def find_reference_start_page(pages: list[str] | None) -> int | None:
    if pages is None:
        return None
    reference_heading = re.compile(r"(?im)^\s*(?:\d+\s+)?references\s*$")
    for index, text in enumerate(pages, start=1):
        if reference_heading.search(text):
            return index
    return None


def checklist_answer_is_no(checklist: str, item_phrase: str) -> bool:
    start = checklist.lower().find(item_phrase.lower())
    if start < 0:
        return False
    match = re.search(r"\n\\item\s+\{\\bf", checklist[start + 10 :])
    end = start + 10 + match.start() if match else start + 1000
    block = checklist[start:end]
    return r"\answerNo" in block


def release_metadata_doc_status(path: Path, required_terms: list[str]) -> dict[str, Any]:
    exists = path.exists()
    text = path.read_text(encoding="utf-8") if exists else ""
    missing_terms = [term for term in required_terms if term not in text]
    return {
        "path": relpath(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "required_terms_present": not missing_terms,
        "missing_terms": missing_terms,
    }


def local_gpu_validation_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": relpath(path),
            "exists": False,
            "local_gpu_container_ready": False,
            "image_id": "",
            "risk_flags": ["local_gpu_container_validation_missing"],
        }
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    image = payload.get("image", {}) if isinstance(payload, dict) else {}
    return {
        "path": relpath(path),
        "exists": True,
        "local_gpu_container_ready": payload.get("local_gpu_container_ready") is True,
        "image_id": str(image.get("id", "")),
        "risk_flags": list(payload.get("risk_flags", [])),
    }


def external_release_packaging_flags(
    path: Path, *, local_gpu_container_ready: bool
) -> tuple[list[str], list[str]]:
    fallback = [
        "public_release_upload_not_verified",
        "external_ci_run_not_observed",
        "gpu_container_run_not_observed",
        "public_repository_upload_not_verified",
    ]
    if not path.exists():
        return fallback, []
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    external_to_venue = {
        "public_release_upload_not_verified": "public_release_upload_not_verified",
        "public_repository_state_not_verified": "public_repository_upload_not_verified",
        "external_ci_run_not_observed": "external_ci_run_not_observed",
        "external_gpu_container_run_not_observed": "gpu_container_run_not_observed",
    }
    flags: list[str] = []
    warnings: list[str] = []
    for flag in payload.get("risk_flags", []):
        mapped = external_to_venue.get(str(flag))
        if mapped == "gpu_container_run_not_observed" and local_gpu_container_ready:
            warnings.append("external_gpu_container_run_not_observed")
            continue
        if mapped and mapped not in flags:
            flags.append(mapped)
    return flags, warnings


def build_audit(
    paper_tex: Path,
    submission_pdf: Path,
    official_style: Path,
    checklist_tex: Path,
    compute_resource_doc: Path,
    asset_license_doc: Path,
    new_asset_doc: Path,
    external_validation_audit: Path,
    local_gpu_validation: Path,
    target_venue: str,
    venue_profile: str,
    max_pages: int,
    max_abstract_words: int,
) -> dict[str, Any]:
    text = paper_tex.read_text(encoding="utf-8")
    checklist = checklist_tex.read_text(encoding="utf-8") if checklist_tex.exists() else ""
    title = latex_group(text, "title")
    author = latex_group(text, "author")
    abstract = abstract_text(text)
    doc_class = document_class(text)
    total_pdf_pages = pdf_page_count(submission_pdf)
    pages = page_texts(submission_pdf, total_pdf_pages)
    reference_start_page = find_reference_start_page(pages)
    content_pages_before_references = (
        reference_start_page - 1 if reference_start_page is not None else None
    )
    abstract_words = word_count(abstract)
    size_bytes = submission_pdf.stat().st_size if submission_pdf.exists() else 0

    official_style_file_present = official_style.exists() and official_style.stat().st_size > 1000
    neurips_style_source_wired = r"\usepackage[main]{neurips_2026}" in text
    neurips_natbib_options_wired = (
        r"\PassOptionsToPackage{numbers,compress}{natbib}" in text
    )
    neurips_checklist_input_wired = r"\input{neurips_checklist.tex}" in text
    appendix_after_references_wired = command_order(
        text,
        [r"\bibliography{refs}", r"\appendix", r"\input{neurips_checklist.tex}"],
    )

    checklist_present = checklist_tex.exists() and "NeurIPS Paper Checklist" in checklist
    checklist_no_todos = not any(
        marker in checklist for marker in [r"\answerTODO", r"\justificationTODO", "TODO", "TBD", "FIXME"]
    )

    release_metadata_docs = {
        "compute_resource_accounting": release_metadata_doc_status(
            compute_resource_doc,
            RELEASE_METADATA_DOCS["compute_resources_not_reported"][2],
        ),
        "asset_license_inventory": release_metadata_doc_status(
            asset_license_doc,
            RELEASE_METADATA_DOCS["existing_asset_licenses_not_reported"][2],
        ),
        "new_asset_inventory": release_metadata_doc_status(
            new_asset_doc,
            RELEASE_METADATA_DOCS["new_asset_metadata_not_reported"][2],
        ),
    }
    local_gpu = local_gpu_validation_status(local_gpu_validation)

    checklist_release_risk_flags: list[str] = []
    for phrase, flag in [
        ("Open access to data and code", "public_code_or_data_not_yet_open"),
        ("Compute resources", "compute_resources_not_reported"),
        ("Licenses for existing assets", "existing_asset_licenses_not_reported"),
        ("New assets", "new_asset_metadata_not_reported"),
    ]:
        if checklist_answer_is_no(checklist, phrase):
            checklist_release_risk_flags.append(flag)
    for flag in [
        "compute_resources_not_reported",
        "existing_asset_licenses_not_reported",
        "new_asset_metadata_not_reported",
    ]:
        doc_key = RELEASE_METADATA_DOCS[flag][0]
        status = release_metadata_docs[doc_key]
        if (
            (not status["exists"] or not status["required_terms_present"])
            and flag not in checklist_release_risk_flags
        ):
            checklist_release_risk_flags.append(flag)

    content_risk_flags: list[str] = []
    if not title:
        content_risk_flags.append("missing_title")
    if author.lower() != "anonymous":
        content_risk_flags.append("non_anonymous_author")
    if not abstract:
        content_risk_flags.append("missing_abstract")
    if abstract_words > max_abstract_words:
        content_risk_flags.append("abstract_over_word_budget")
    if total_pdf_pages is None:
        content_risk_flags.append("submission_pdf_page_count_unavailable")
    if reference_start_page is None:
        content_risk_flags.append("reference_start_page_unavailable")
    elif content_pages_before_references is not None and content_pages_before_references > max_pages:
        content_risk_flags.append("main_content_over_neurips_page_budget")
    if size_bytes < 100_000:
        content_risk_flags.append("submission_pdf_too_small")
    if r"\bibliography{" not in text:
        content_risk_flags.append("missing_bibliography")
    if text.count(r"\includegraphics") < 2:
        content_risk_flags.append("too_few_main_figures")
    if any(marker in text for marker in ["TODO", "TBD", "FIXME"]):
        content_risk_flags.append("draft_marker_in_paper")
    if not appendix_after_references_wired:
        content_risk_flags.append("appendix_or_checklist_not_after_references")

    venue_binding_risk_flags: list[str] = []
    if not target_venue:
        venue_binding_risk_flags.append("target_venue_unspecified")
    if venue_profile != DEFAULT_VENUE_PROFILE:
        venue_binding_risk_flags.append("unexpected_venue_profile")
    if not official_style_file_present:
        venue_binding_risk_flags.append("official_neurips_style_file_missing")
    if not neurips_style_source_wired:
        venue_binding_risk_flags.append("neurips_style_flag_not_wired")
    if not neurips_natbib_options_wired:
        venue_binding_risk_flags.append("neurips_natbib_options_not_wired")
    if not checklist_present:
        venue_binding_risk_flags.append("neurips_checklist_missing")
    if not neurips_checklist_input_wired:
        venue_binding_risk_flags.append("neurips_checklist_not_included")
    if not checklist_no_todos:
        venue_binding_risk_flags.append("neurips_checklist_has_todo")

    release_packaging_risk_flags, release_packaging_warning_flags = external_release_packaging_flags(
        external_validation_audit,
        local_gpu_container_ready=local_gpu["local_gpu_container_ready"],
    )
    if not local_gpu["local_gpu_container_ready"]:
        for flag in local_gpu["risk_flags"]:
            if flag not in release_packaging_risk_flags:
                release_packaging_risk_flags.append(flag)

    content_packet_ready = not content_risk_flags
    venue_binding_ready = not venue_binding_risk_flags
    checklist_release_ready = not checklist_release_risk_flags
    release_packaging_ready = not release_packaging_risk_flags
    venue_submission_ready = (
        content_packet_ready
        and venue_binding_ready
        and checklist_release_ready
        and release_packaging_ready
    )
    blocking_next_steps = []
    if content_risk_flags:
        blocking_next_steps.append("Repair the NeurIPS PDF content/page-budget risk flags.")
    if venue_binding_risk_flags:
        blocking_next_steps.append("Repair the official NeurIPS style/checklist binding flags.")
    if checklist_release_risk_flags:
        if checklist_release_risk_flags == ["public_code_or_data_not_yet_open"]:
            blocking_next_steps.append("Publish the anonymized public code/data release.")
        else:
            blocking_next_steps.append(
                "Publish public code/data and repair any missing checklist metadata documentation."
            )
    if release_packaging_risk_flags:
        if release_packaging_risk_flags == ["local_gpu_container_validation_missing"]:
            blocking_next_steps.append("Run make local-gpu-container-validation on a CUDA Docker host.")
        elif release_packaging_risk_flags == ["gpu_container_run_not_observed"]:
            blocking_next_steps.append(
                "Run the GPU container on an external CUDA host and record the passing log URL."
            )
        else:
            blocking_next_steps.append(
                "Verify the public archive upload, external CI/GPU-container runs, and clean public repository upload."
            )

    return {
        "paper_tex": relpath(paper_tex),
        "submission_pdf": relpath(submission_pdf),
        "official_style": relpath(official_style),
        "checklist_tex": relpath(checklist_tex),
        "external_validation_audit": relpath(external_validation_audit),
        "local_gpu_validation": local_gpu,
        "release_metadata_docs": release_metadata_docs,
        "venue_profile": venue_profile,
        "target_venue": target_venue,
        "document_class": doc_class,
        "known_venue_style_detected": official_style_file_present and neurips_style_source_wired,
        "official_style_file_present": official_style_file_present,
        "neurips_style_source_wired": neurips_style_source_wired,
        "neurips_natbib_options_wired": neurips_natbib_options_wired,
        "neurips_checklist_input_wired": neurips_checklist_input_wired,
        "neurips_checklist_present": checklist_present,
        "neurips_checklist_no_todos": checklist_no_todos,
        "appendix_after_references_wired": appendix_after_references_wired,
        "title_present": bool(title),
        "anonymous_author": author.lower() == "anonymous",
        "abstract_word_count": abstract_words,
        "max_abstract_words": max_abstract_words,
        "submission_pdf_pages": total_pdf_pages,
        "reference_start_page": reference_start_page,
        "content_pages_before_references": content_pages_before_references,
        "max_pages_before_references": max_pages,
        "size_bytes": size_bytes,
        "figure_count": text.count(r"\includegraphics"),
        "table_count": text.count(r"\begin{table}"),
        "bibliography_present": r"\bibliography{" in text,
        "content_risk_flags": content_risk_flags,
        "venue_binding_risk_flags": venue_binding_risk_flags,
        "checklist_release_risk_flags": checklist_release_risk_flags,
        "release_packaging_risk_flags": release_packaging_risk_flags,
        "release_packaging_warning_flags": release_packaging_warning_flags,
        "content_packet_ready": content_packet_ready,
        "venue_binding_ready": venue_binding_ready,
        "checklist_release_ready": checklist_release_ready,
        "release_packaging_ready": release_packaging_ready,
        "venue_submission_ready": venue_submission_ready,
        "blocking_next_steps": blocking_next_steps,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    content_status = "ready" if payload["content_packet_ready"] else "not ready"
    venue_status = "ready" if payload["venue_binding_ready"] else "not ready"
    checklist_status = "ready" if payload["checklist_release_ready"] else "not ready"
    release_status = "ready" if payload["release_packaging_ready"] else "not ready"
    submission_status = "ready" if payload["venue_submission_ready"] else "not ready"
    lines = [
        "# Venue Submission Compliance Audit",
        "",
        "This generated audit checks the NeurIPS 2026 submission binding separately",
        "from public release packaging. NeurIPS page budget is measured as PDF",
        "pages before the References heading; references, appendix material, and",
        "the checklist are tracked but not counted against that main-content limit.",
        "",
        f"Current content-packet status: {content_status}.",
        f"Current venue-binding status: {venue_status}.",
        f"Current checklist-release status: {checklist_status}.",
        f"Current release-packaging status: {release_status}.",
        f"Current venue-submission status: {submission_status}.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Paper source | `{payload['paper_tex']}` |",
        f"| Submission PDF | `{payload['submission_pdf']}` |",
        f"| Official style | `{payload['official_style']}` |",
        f"| Checklist source | `{payload['checklist_tex']}` |",
        f"| External validation audit | `{payload['external_validation_audit']}` |",
        f"| Local GPU validation | `{payload['local_gpu_validation']['path']}` |",
        f"| Venue profile | `{payload['venue_profile']}` |",
        f"| Target venue | `{payload['target_venue']}` |",
        f"| Document class | `{payload['document_class']}` |",
        f"| Known venue style detected | {payload['known_venue_style_detected']} |",
        f"| Official style file present | {payload['official_style_file_present']} |",
        f"| NeurIPS source flag wired | {payload['neurips_style_source_wired']} |",
        f"| NeurIPS checklist present | {payload['neurips_checklist_present']} |",
        f"| NeurIPS checklist has no TODOs | {payload['neurips_checklist_no_todos']} |",
        f"| Anonymous author | {payload['anonymous_author']} |",
        f"| Abstract words | {payload['abstract_word_count']} |",
        f"| Abstract word budget | {payload['max_abstract_words']} |",
        f"| Total submission PDF pages | {payload['submission_pdf_pages']} |",
        f"| References start page | {payload['reference_start_page']} |",
        f"| Main-content pages before references | {payload['content_pages_before_references']} |",
        f"| Main-content page budget | {payload['max_pages_before_references']} |",
        f"| Size bytes | {payload['size_bytes']} |",
        f"| Figures | {payload['figure_count']} |",
        f"| Tables | {payload['table_count']} |",
        f"| Bibliography present | {payload['bibliography_present']} |",
        f"| Appendix/checklist after references | {payload['appendix_after_references_wired']} |",
        f"| Local GPU container ready | {payload['local_gpu_validation']['local_gpu_container_ready']} |",
        f"| Local GPU image ID | `{payload['local_gpu_validation']['image_id']}` |",
        "",
        "## Release Metadata Docs",
        "",
        "| Document | Path | Present | Required terms present | Missing terms |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, status in payload["release_metadata_docs"].items():
        missing_terms = ", ".join(status["missing_terms"]) if status["missing_terms"] else "none"
        lines.append(
            f"| {name} | `{status['path']}` | {status['exists']} | "
            f"{status['required_terms_present']} | {missing_terms} |"
        )
    lines.extend(
        [
            "",
        "## Content Risk Flags",
        "",
        ]
    )
    if payload["content_risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["content_risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Venue Binding Risk Flags", ""])
    if payload["venue_binding_risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["venue_binding_risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Checklist Release Risk Flags", ""])
    if payload["checklist_release_risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["checklist_release_risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Release Packaging Risk Flags", ""])
    if payload["release_packaging_risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["release_packaging_risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Release Packaging Warning Flags", ""])
    if payload["release_packaging_warning_flags"]:
        lines.extend(f"- {flag}" for flag in payload["release_packaging_warning_flags"])
    else:
        lines.append("- none")
    if payload["blocking_next_steps"]:
        lines.extend(["", "## Blocking Next Steps", ""])
        lines.extend(f"- {step}" for step in payload["blocking_next_steps"])
    lines.extend(
        [
            "",
            "This file is generated by `scripts/audit_venue_submission_compliance.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_audit(
        paper_tex=args.paper_tex,
        submission_pdf=args.submission_pdf,
        official_style=args.official_style,
        checklist_tex=args.checklist_tex,
        compute_resource_doc=args.compute_resource_doc,
        asset_license_doc=args.asset_license_doc,
        new_asset_doc=args.new_asset_doc,
        external_validation_audit=args.external_validation_audit,
        local_gpu_validation=args.local_gpu_validation,
        target_venue=args.target_venue,
        venue_profile=args.venue_profile,
        max_pages=args.max_pages,
        max_abstract_words=args.max_abstract_words,
    )
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "content_packet_ready": payload["content_packet_ready"],
                "venue_binding_ready": payload["venue_binding_ready"],
                "checklist_release_ready": payload["checklist_release_ready"],
                "release_packaging_ready": payload["release_packaging_ready"],
                "venue_submission_ready": payload["venue_submission_ready"],
                "content_risk_flags": payload["content_risk_flags"],
                "venue_binding_risk_flags": payload["venue_binding_risk_flags"],
                "checklist_release_risk_flags": payload["checklist_release_risk_flags"],
                "release_packaging_risk_flags": payload["release_packaging_risk_flags"],
                "release_packaging_warning_flags": payload["release_packaging_warning_flags"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
