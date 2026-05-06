#!/usr/bin/env python
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = ROOT / "docs" / "external_validation_receipts.json"
DEFAULT_TEMPLATE = ROOT / "runs" / "external_validation_receipt_template.json"

URL_PATTERN = re.compile(r"^(https?://|doi:|10\.)", re.IGNORECASE)
HEX40_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
IMAGE_DIGEST_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
PLACEHOLDER_URL_TOKENS = {
    "<",
    ">",
    "example.com",
    "example.org",
    "example.net",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "your-",
    "placeholder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely update docs/external_validation_receipts.json from the "
            "current receipt template and externally observed URLs/evidence."
        )
    )
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write docs/external_validation_receipts.json. Without this flag, only validate and print the candidate JSON.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail unless all four external receipt groups are supplied.",
    )
    parser.add_argument("--public-release-url", default="")
    parser.add_argument("--public-repository-url", default="")
    parser.add_argument("--public-repository-clean-tree-evidence", default="")
    parser.add_argument("--external-ci-url", default="")
    parser.add_argument("--external-ci-passed", action="store_true")
    parser.add_argument("--external-gpu-url", default="")
    parser.add_argument("--external-gpu-image-digest", default="")
    parser.add_argument("--external-gpu-passed", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def meaningful(value: Any) -> bool:
    return bool(str(value).strip())


def require_url(value: str, field: str, errors: list[str]) -> None:
    if not meaningful(value):
        errors.append(f"{field}_missing")
    elif not URL_PATTERN.search(value.strip()):
        errors.append(f"{field}_not_url_like")
    elif url_has_placeholder(value):
        errors.append(f"{field}_placeholder")


def normalized_url(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("doi:"):
        return f"https://doi.org/{text.split(':', 1)[1].strip()}"
    if text.startswith("10."):
        return f"https://doi.org/{text}"
    return text


def url_has_placeholder(value: Any) -> bool:
    text = str(value).strip().lower()
    if any(token in text for token in PLACEHOLDER_URL_TOKENS):
        return True
    parsed = urllib.parse.urlparse(normalized_url(value))
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def template_receipts(template: dict[str, Any]) -> dict[str, Any]:
    receipts = template.get("receipt_template", {}).get("receipts", {})
    return receipts if isinstance(receipts, dict) else {}


def validate_template(template: dict[str, Any], errors: list[str]) -> tuple[str, str]:
    receipts = template_receipts(template)
    archive_sha = str(
        receipts.get("public_release_upload", {}).get("artifact_sha256", "")
    ).strip()
    commit = str(receipts.get("public_repository", {}).get("commit", "")).strip()
    if template.get("external_validation_receipt_template_ready") is not True:
        errors.append("receipt_template_not_ready")
    if set(receipts) != {
        "public_release_upload",
        "public_repository",
        "external_ci",
        "external_gpu_container",
    }:
        errors.append("receipt_template_keys_changed")
    if not HEX64_PATTERN.match(archive_sha):
        errors.append("receipt_template_archive_sha256_invalid")
    if not HEX40_PATTERN.match(commit):
        errors.append("receipt_template_source_commit_invalid")
    for key in ["external_ci", "external_gpu_container"]:
        key_commit = str(receipts.get(key, {}).get("commit", "")).strip()
        if key_commit != commit:
            errors.append(f"receipt_template_{key}_commit_mismatch")
    return archive_sha, commit


def group_supplied(*values: Any) -> bool:
    return any(bool(value) for value in values)


def build_candidate(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    updated: list[str] = []
    receipts_payload = load_json(args.receipts)
    template_payload = load_json(args.template)
    archive_sha, commit = validate_template(template_payload, errors)

    candidate = json.loads(json.dumps(receipts_payload))
    receipts = candidate.setdefault("receipts", {})
    if not isinstance(receipts, dict):
        errors.append("receipt_registry_receipts_not_object")
        receipts = {}
        candidate["receipts"] = receipts

    release_supplied = group_supplied(args.public_release_url)
    repository_supplied = group_supplied(
        args.public_repository_url,
        args.public_repository_clean_tree_evidence,
    )
    ci_supplied = group_supplied(args.external_ci_url, args.external_ci_passed)
    gpu_supplied = group_supplied(
        args.external_gpu_url,
        args.external_gpu_image_digest,
        args.external_gpu_passed,
    )

    if release_supplied:
        require_url(args.public_release_url, "public_release_url", errors)
        receipts["public_release_upload"] = {
            "status": "observed",
            "url": args.public_release_url.strip(),
            "artifact_sha256": archive_sha,
            "notes": (
                "Updated by scripts/update_external_validation_receipts.py from "
                "the current external validation receipt template."
            ),
        }
        updated.append("public_release_upload")

    if repository_supplied:
        require_url(args.public_repository_url, "public_repository_url", errors)
        if not meaningful(args.public_repository_clean_tree_evidence):
            errors.append("public_repository_clean_tree_evidence_missing")
        receipts["public_repository"] = {
            "status": "observed",
            "url": args.public_repository_url.strip(),
            "commit": commit,
            "clean_tree_evidence": args.public_repository_clean_tree_evidence.strip(),
            "notes": (
                "Updated by scripts/update_external_validation_receipts.py from "
                "the current external validation receipt template."
            ),
        }
        updated.append("public_repository")

    if ci_supplied:
        require_url(args.external_ci_url, "external_ci_url", errors)
        if args.external_ci_passed is not True:
            errors.append("external_ci_passed_missing")
        receipts["external_ci"] = {
            "status": "observed",
            "url": args.external_ci_url.strip(),
            "commit": commit,
            "passed": True,
            "notes": (
                "Updated by scripts/update_external_validation_receipts.py from "
                "the current external validation receipt template."
            ),
        }
        updated.append("external_ci")

    if gpu_supplied:
        require_url(args.external_gpu_url, "external_gpu_url", errors)
        digest = args.external_gpu_image_digest.strip()
        if not IMAGE_DIGEST_PATTERN.match(digest):
            errors.append("external_gpu_image_digest_invalid")
        if args.external_gpu_passed is not True:
            errors.append("external_gpu_passed_missing")
        receipts["external_gpu_container"] = {
            "status": "observed",
            "url": args.external_gpu_url.strip(),
            "commit": commit,
            "image_digest": digest,
            "passed": True,
            "notes": (
                "Updated by scripts/update_external_validation_receipts.py from "
                "the current external validation receipt template."
            ),
        }
        updated.append("external_gpu_container")

    expected = {
        "public_release_upload",
        "public_repository",
        "external_ci",
        "external_gpu_container",
    }
    if args.require_all and set(updated) != expected:
        errors.append(
            "require_all_missing:"
            + ",".join(sorted(expected.difference(updated)))
        )
    if not updated:
        errors.append("no_receipt_groups_supplied")
    return candidate, updated, errors


def main() -> None:
    args = parse_args()
    candidate, updated, errors = build_candidate(args)
    summary = {
        "receipt_update_ready": not errors,
        "updated_receipts": updated,
        "risk_flags": errors,
        "receipts": relpath(args.receipts),
        "template": relpath(args.template),
        "write": args.write,
    }
    if errors:
        print(json.dumps(summary))
        raise SystemExit(1)
    if args.write:
        write_json(candidate, args.receipts)
        print(json.dumps(summary))
    else:
        print(json.dumps(summary))
        print(json.dumps(candidate, indent=2))


if __name__ == "__main__":
    main()
