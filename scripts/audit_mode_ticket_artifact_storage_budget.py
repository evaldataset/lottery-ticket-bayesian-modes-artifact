#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "mode_ticket_artifact_storage_budget.json"
DEFAULT_OUT_MD = ROOT / "docs" / "mode_ticket_artifact_storage_budget.md"
FAKE_ARTIFACT_ROOT = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_smoke"


SCENARIOS = [
    {
        "name": "sgld_activation_aligned_save_states",
        "description": "Five-seed SGLD direct rerun with activation alignment and saved states",
        "seeds": 5,
        "samples_per_chain": 10,
        "posterior_chains": 1,
        "alignment_method": "activation",
        "save_states": True,
        "posterior_sampler": "sgld",
        "recommended": True,
    },
    {
        "name": "sgld_weight_aligned_save_states",
        "description": "Five-seed SGLD direct rerun with weight alignment and saved states",
        "seeds": 5,
        "samples_per_chain": 10,
        "posterior_chains": 1,
        "alignment_method": "weight",
        "save_states": True,
        "posterior_sampler": "sgld",
        "recommended": False,
    },
    {
        "name": "csgld_independent_multichain_save_states",
        "description": "Independent-start cyclical-SGLD rerun with saved states",
        "seeds": 5,
        "samples_per_chain": 5,
        "posterior_chains": 3,
        "alignment_method": "none",
        "save_states": True,
        "posterior_sampler": "cyclical-sgld",
        "recommended": False,
    },
    {
        "name": "lowrank128_laplace_save_states",
        "description": "Rank-128 low-rank Laplace direct rerun with saved states",
        "seeds": 5,
        "samples_per_chain": 10,
        "posterior_chains": 1,
        "alignment_method": "none",
        "save_states": True,
        "posterior_sampler": "lowrank-laplace",
        "recommended": False,
    },
    {
        "name": "jointdiag_laplace_save_states",
        "description": "270k streamed joint-group Laplace direct rerun with saved states",
        "seeds": 5,
        "samples_per_chain": 5,
        "posterior_chains": 1,
        "alignment_method": "none",
        "save_states": True,
        "posterior_sampler": "jointdiag-laplace",
        "recommended": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-06")
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--blocks-per-stage", type=int, default=3)
    return parser.parse_args()


def resnet20_weight_parameter_count(
    *,
    width: int,
    num_classes: int,
    blocks_per_stage: int,
    input_channels: int = 3,
) -> int:
    total = width * input_channels * 3 * 3
    stage_in = width
    for stage_idx, stage_width in enumerate([width, width * 2, width * 4], start=1):
        for block_idx in range(blocks_per_stage):
            in_channels = stage_in if block_idx == 0 else stage_width
            total += stage_width * in_channels * 3 * 3
            total += stage_width * stage_width * 3 * 3
            if block_idx == 0 and stage_idx > 1:
                total += stage_width * in_channels
        stage_in = stage_width
    total += num_classes * width * 4
    return int(total)


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def latest_fake_artifact_size() -> dict[str, Any]:
    candidates = sorted(FAKE_ARTIFACT_ROOT.glob("*/mask_artifacts.npz"))
    if not candidates:
        return {"path": None, "bytes": 0}
    path = candidates[-1]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
    }


def scenario_budget(
    scenario: dict[str, Any],
    *,
    parameter_count: int,
) -> dict[str, Any]:
    seeds = int(scenario["seeds"])
    samples = int(scenario["samples_per_chain"])
    chains = int(scenario["posterior_chains"])
    chain_start = seeds * chains
    posterior_sample = seeds * chains * samples
    posterior_mode_upper = posterior_sample
    ticket = seeds
    base_records = chain_start + posterior_sample + posterior_mode_upper + ticket
    alignment_multiplier = 2 if scenario["alignment_method"] != "none" else 1
    mask_records = base_records * alignment_multiplier
    state_records = mask_records if scenario["save_states"] else 0
    mask_bytes = mask_records * parameter_count
    state_bytes = state_records * parameter_count * 4
    total_bytes = mask_bytes + state_bytes
    return {
        **scenario,
        "parameter_count": parameter_count,
        "chain_start_records": chain_start,
        "posterior_sample_records": posterior_sample,
        "posterior_mode_records_upper_bound": posterior_mode_upper,
        "ticket_records": ticket,
        "base_record_count_upper_bound": base_records,
        "alignment_multiplier": alignment_multiplier,
        "mask_record_count_upper_bound": mask_records,
        "state_record_count_upper_bound": state_records,
        "mask_bytes_uncompressed": mask_bytes,
        "state_bytes_uncompressed": state_bytes,
        "total_bytes_uncompressed": total_bytes,
        "total_mib_uncompressed": total_bytes / (1024 * 1024),
    }


def recommended_command() -> str:
    return (
        ".venv/bin/python scripts/run_mode_ticket_distribution_probe.py "
        "--dataset cifar10 --model resnet20 --resnet-width 16 "
        "--seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 "
        "--imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 "
        "--prune-fraction 0.30 --samples 10 --sgld-steps 200 "
        "--sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 "
        "--batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 "
        "--augment --alignment-method activation --alignment-batches 10 "
        "--save-mask-artifacts --save-state-artifacts "
        "--out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3"
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    parameter_count = resnet20_weight_parameter_count(
        width=args.resnet_width,
        num_classes=args.num_classes,
        blocks_per_stage=args.blocks_per_stage,
    )
    scenario_rows = [
        scenario_budget(scenario, parameter_count=parameter_count)
        for scenario in SCENARIOS
    ]
    recommended = [row for row in scenario_rows if row["recommended"]][0]
    fake_artifact = latest_fake_artifact_size()
    return {
        "date": args.date,
        "model": "resnet20",
        "dataset": "cifar10",
        "resnet_width": args.resnet_width,
        "num_classes": args.num_classes,
        "blocks_per_stage": args.blocks_per_stage,
        "parameter_count": parameter_count,
        "fake_cifar_fixture": fake_artifact,
        "scenarios": scenario_rows,
        "recommended_next_rerun": {
            "scenario": recommended["name"],
            "command": recommended_command(),
            "estimated_total_bytes_uncompressed": recommended[
                "total_bytes_uncompressed"
            ],
            "estimated_total_mib_uncompressed": recommended[
                "total_mib_uncompressed"
            ],
            "posthoc_command": (
                ".venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py "
                "--artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/<RUN_ID>/mask_artifacts.npz "
                "--out-json runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json "
                "--out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md "
                "--max-channel-pair-count 1"
            ),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    fake = payload["fake_cifar_fixture"]
    recommended = payload["recommended_next_rerun"]
    lines = [
        "# Mode/Ticket Artifact Storage Budget",
        "",
        f"Date: {payload['date']}",
        "",
        "This audit estimates the raw storage footprint of rerunning full-data",
        "CIFAR-10 ResNet-20 direct mode/ticket probes with saved mask and state",
        "artifacts. It is intentionally dependency-light so the estimate runs in",
        "the artifact-verification container.",
        "",
        f"- Model: `{payload['model']}` width `{payload['resnet_width']}`",
        f"- Weight parameter count: `{payload['parameter_count']}`",
        f"- Latest fake-CIFAR fixture: `{fake['path']}`",
        f"- Latest fake-CIFAR fixture bytes: `{fake['bytes']}` ({fmt_bytes(int(fake['bytes']))})",
        "",
        "## Scenario Estimates",
        "",
        "| Scenario | Sampler | Align | Samples | Chains | Mask records | State records | Uncompressed total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["scenarios"]:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['posterior_sampler']} | "
            f"{row['alignment_method']} | "
            f"{row['samples_per_chain']} | "
            f"{row['posterior_chains']} | "
            f"{row['mask_record_count_upper_bound']} | "
            f"{row['state_record_count_upper_bound']} | "
            f"{fmt_bytes(int(row['total_bytes_uncompressed']))} |"
        )
    lines.extend(
        [
            "",
            "## Reference Full-Data Rerun",
            "",
            "The claim-level saved-artifact rerun uses the activation-aligned",
            "SGLD configuration, because it directly closes the artifact gap for",
            "the strongest existing first-order channel-alignment row.",
            "",
            "```bash",
            recommended["command"],
            "```",
            "",
            "Then run the post-hoc audit on the produced `mask_artifacts.npz`:",
            "",
            "```bash",
            recommended["posthoc_command"],
            "```",
            "",
            "This budget estimates an upper bound using one posterior-mode record",
            "per posterior sample. Real compressed `.npz` size may be lower, but",
            "the uncompressed state matrix dominates when `--save-state-artifacts`",
            "is enabled.",
            "",
            "This file is generated by",
            "`scripts/audit_mode_ticket_artifact_storage_budget.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, payload)
    recommended = payload["recommended_next_rerun"]
    print(
        json.dumps(
            {
                "out_json": args.out_json.relative_to(ROOT).as_posix(),
                "out_md": args.out_md.relative_to(ROOT).as_posix(),
                "parameter_count": payload["parameter_count"],
                "recommended_total_mib_uncompressed": recommended[
                    "estimated_total_mib_uncompressed"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
