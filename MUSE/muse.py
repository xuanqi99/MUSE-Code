#!/usr/bin/env python
"""Dataset-aware launcher for MUSE fine-tuning and generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
STRENGTHS = (0.6, 0.7, 0.8, 0.9, 1.0)

DOMAIN_PRESETS = {
    "office31": {
        "A": "amazon",
        "D": "dslr",
        "W": "webcam",
    },
    "officehome": {
        "A": "Art",
        "C": "Clipart",
        "P": "Product",
        "R": "Real World",
    },
    "minidomainnet": {
        "C": "clipart",
        "P": "painting",
        "R": "real",
        "S": "sketch",
    },
}

TASK_PRESETS = {
    "office31": {
        "A2DW": ("A", ["D", "W"]),
        "D2AW": ("D", ["A", "W"]),
        "W2AD": ("W", ["A", "D"]),
    },
    "officehome": {
        "A2CPR": ("A", ["C", "P", "R"]),
        "C2APR": ("C", ["A", "P", "R"]),
        "P2ACR": ("P", ["A", "C", "R"]),
        "R2ACP": ("R", ["A", "C", "P"]),
    },
    "minidomainnet": {
        "C2PRS": ("C", ["P", "R", "S"]),
        "P2CRS": ("P", ["C", "R", "S"]),
        "R2CPS": ("R", ["C", "P", "S"]),
        "S2CPR": ("S", ["C", "P", "R"]),
    },
}


def _split_domains(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _canonical_domain(dataset: str, value: str) -> str:
    aliases = DOMAIN_PRESETS[dataset]
    key = value.strip()
    if key.upper() in aliases:
        return aliases[key.upper()]

    compact = key.replace("_", " ").replace("-", " ").lower()
    for domain in aliases.values():
        if compact == domain.replace("_", " ").replace("-", " ").lower():
            return domain
    return key


def _resolve_domains(args: argparse.Namespace) -> tuple[str, list[str], str]:
    dataset = args.dataset
    task_name = None
    source = args.source_domain
    targets = _split_domains(args.target_domains)

    if args.task:
        normalized_task = args.task.upper().replace("->", "2").replace("-", "")
        if normalized_task not in TASK_PRESETS[dataset]:
            known = ", ".join(sorted(TASK_PRESETS[dataset]))
            raise SystemExit(f"Unknown task {args.task!r} for {dataset}. Available: {known}")
        source_key, target_keys = TASK_PRESETS[dataset][normalized_task]
        source = source or source_key
        targets = targets or target_keys
        task_name = normalized_task

    if not source:
        raise SystemExit("--source_domain is required when --task is not provided.")

    source = _canonical_domain(dataset, source)
    if source not in DOMAIN_PRESETS[dataset].values():
        raise SystemExit(f"Unknown source domain: {source}")
    if not targets:
        targets = [
            domain for domain in DOMAIN_PRESETS[dataset].values() if domain != source
        ]
    targets = [_canonical_domain(dataset, item) for item in targets]
    if len(set(targets)) != len(targets) or source in targets:
        raise SystemExit("Target domains must be distinct and exclude the source domain.")
    if any(target not in DOMAIN_PRESETS[dataset].values() for target in targets):
        raise SystemExit(f"Unknown target domain in {targets}")
    if not targets:
        raise SystemExit("At least one target domain is required.")

    if task_name is None:
        compact_targets = "".join(_short_name(dataset, target) for target in targets)
        task_name = f"{_short_name(dataset, source)}2{compact_targets}"

    return source, targets, task_name


def _short_name(dataset: str, domain: str) -> str:
    canonical = _canonical_domain(dataset, domain)
    for key, value in DOMAIN_PRESETS[dataset].items():
        if value == canonical:
            return key
    return canonical.replace(" ", "_")


def _safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _output_dir(root: str, prefix: str, dataset: str, task: str) -> str:
    return str(Path(root) / f"{prefix}_{dataset}_{_safe_name(task)}")


def _append_common_dataset_args(
    cmd: list[str],
    args: argparse.Namespace,
    source: str,
    targets: Sequence[str],
) -> None:
    cmd.extend(["--dataset", args.dataset])
    if args.data_root:
        cmd.extend(["--data_root", args.data_root])
    if args.dataset == "minidomainnet" and args.minidomainnet_split_dir:
        cmd.extend(["--minidomainnet_split_dir", args.minidomainnet_split_dir])
    cmd.extend(["--source_domain", source, "--target_domains", ",".join(targets)])


def _run(cmd: Sequence[str], dry_run: bool) -> int:
    printable = " ".join(f'"{item}"' if " " in item else item for item in cmd)
    print(printable)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=True).returncode


def _train(args: argparse.Namespace, extra: Sequence[str]) -> int:
    source, targets, task = _resolve_domains(args)
    output_dir = args.output_dir or _output_dir(args.output_root, "muse", args.dataset, task)
    cmd = [
        sys.executable,
        str(HERE / "train_MUSE.py"),
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--instance_data_dir",
        args.data_root,
        "--source_domain",
        source,
        "--target_domain",
        targets[0],
        "--target_domains",
        ",".join(targets),
        "--output_dir",
        output_dir,
        "--instance_prompt",
        args.instance_prompt,
    ]
    cmd.extend(["--dataset", args.dataset])
    if args.validation_prompt:
        cmd.extend(["--validation_prompt", args.validation_prompt])
    if args.pretrained_vae_model_name_or_path:
        cmd.extend(["--pretrained_vae_model_name_or_path", args.pretrained_vae_model_name_or_path])
    if args.dataset == "minidomainnet" and args.minidomainnet_split_dir:
        cmd.extend(["--minidomainnet_split_dir", args.minidomainnet_split_dir])
    cmd.extend(extra)
    return _run(cmd, args.dry_run)


def _generate(args: argparse.Namespace, extra: Sequence[str]) -> int:
    source, targets, task = _resolve_domains(args)
    prefix = "pure_noise" if args.generation == "pure_noise" else "ddim_inversion"
    output_dir = args.output_dir or _output_dir(args.output_root, prefix, args.dataset, task)
    script = "generate_pure_noise.py" if args.generation == "pure_noise" else "generate_ddim_inversion.py"

    if not args.checkpoint_dir:
        raise SystemExit("--checkpoint_dir is required for generation.")

    cmd = [
        sys.executable,
        str(HERE / script),
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--output_dir",
        output_dir,
    ]
    if args.pretrained_vae_model_name_or_path:
        cmd.extend(["--pretrained_vae_model_name_or_path", args.pretrained_vae_model_name_or_path])

    if args.generation == "pure_noise":
        cmd.extend(["--dataset", args.dataset, "--source_domain", source, "--target_domains", ",".join(targets)])
        if args.data_root:
            cmd.extend(["--data_root", args.data_root])
        if args.dataset == "minidomainnet" and args.minidomainnet_split_dir:
            cmd.extend(["--minidomainnet_split_dir", args.minidomainnet_split_dir])
    else:
        _append_common_dataset_args(cmd, args, source, targets)

    cmd.extend(extra)
    return _run(cmd, args.dry_run)


def bridge_groups() -> list[dict]:
    groups = []
    for inversion in STRENGTHS:
        for adapter in STRENGTHS:
            groups.append({
                "name": f"ddim_inv_{inversion:.1f}_adapter_{adapter:.1f}",
                "generation": "ddim_inversion",
                "inversion_strength": inversion,
                "adapter_strength": adapter,
            })
    for adapter in STRENGTHS:
        groups.append({
            "name": f"class_condition_adapter_{adapter:.1f}",
            "generation": "pure_noise",
            "inversion_strength": None,
            "adapter_strength": adapter,
        })
    return groups


def _bridge(args: argparse.Namespace, extra: Sequence[str]) -> int:
    if extra:
        raise SystemExit(f"Unknown bridge arguments: {' '.join(extra)}")
    if args.num_inversion_steps != args.num_inference_steps:
        raise SystemExit("DDIM inversion and generation must use the same step count.")
    if args.num_inference_steps <= 0 or args.num_images_per_class <= 0 or args.max_images_per_class < 0:
        raise SystemExit("Step and class-conditioned image counts must be positive; DDIM cap must be nonnegative.")
    source, targets, task = _resolve_domains(args)
    root = Path(args.output_dir or _output_dir(args.output_root, "bridge", args.dataset, task))
    plan = {
        "dataset": args.dataset, "task": task, "source": source,
        "targets": targets, "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint_dir).resolve()),
        "base_model": args.pretrained_model_name_or_path,
        "vae": args.pretrained_vae_model_name_or_path,
        "data_root": str(Path(args.data_root).resolve()),
        "split_dir": str(Path(args.minidomainnet_split_dir).resolve()) if args.minidomainnet_split_dir else None,
        "num_inference_steps": args.num_inference_steps,
        "num_inversion_steps": args.num_inversion_steps,
        "num_images_per_class": args.num_images_per_class,
        "max_images_per_class": args.max_images_per_class,
        "height": args.height, "width": args.width, "resolution": args.resolution, "dtype": args.dtype,
        "guidance_scale": args.guidance_scale,
        "generation_guidance_scale": args.generation_guidance_scale,
        "inversion_guidance_scale": args.inversion_guidance_scale,
        "semantic_scale": args.semantic_scale, "style_scale": args.style_scale,
        "groups": bridge_groups(),
    }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise SystemExit("This output directory contains a different generation plan. Choose a new --output_dir.")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Bridge: {len(plan['groups'])} groups for each target: {', '.join(targets)}", flush=True)
    for group in plan["groups"]:
        generation_args = argparse.Namespace(**vars(args))
        generation_args.generation = group["generation"]
        generation_args.output_dir = str(root / "groups" / group["name"])
        options = [
            "--seed", str(args.seed), "--lora_scale", str(group["adapter_strength"]),
            "--semantic_scale", str(args.semantic_scale), "--style_scale", str(args.style_scale),
            "--num_inference_steps", str(args.num_inference_steps),
            "--dtype", args.dtype,
        ]
        if args.device:
            options.extend(["--device", args.device])
        if group["generation"] == "ddim_inversion":
            options.extend([
                "--inversion_end_ratio", str(group["inversion_strength"]),
                "--generation_guidance_scale", str(args.generation_guidance_scale),
                "--inversion_guidance_scale", str(args.inversion_guidance_scale),
                "--resolution", str(args.resolution),
                "--inversion_lora_scale", "1.0",
                "--inversion_num_inference_steps", str(args.num_inversion_steps),
                "--max_images_per_class", str(args.max_images_per_class),
                "--prior_cache_dir", str(root / "inversion_cache" / f"inv_{group['inversion_strength']:.1f}"),
            ])
        else:
            options.extend(["--num_images_per_class", str(args.num_images_per_class),
                            "--guidance_scale", str(args.guidance_scale),
                            "--height", str(args.height), "--width", str(args.width)])
        _generate(generation_args, options)

    cmd = [sys.executable, str(HERE / "prepare_bridge.py"), "--root", str(root)]
    return _run(cmd, args.dry_run)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(DOMAIN_PRESETS), required=True)
    parser.add_argument("--task", default=None, help="Preset task, e.g. A2DW, P2ACR, S2CPR.")
    parser.add_argument("--data_root", required=True, help="Dataset root directory.")
    parser.add_argument("--source_domain", default=None, help="Domain name or short key.")
    parser.add_argument("--target_domains", default=None, help="Comma-separated domain names or short keys.")
    parser.add_argument("--minidomainnet_split_dir", default=None)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved command without executing it.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MUSE launcher for domain-specific fine-tuning and generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Fine-tune MUSE for a dataset/task.")
    _add_common(train)
    train.add_argument("--instance_prompt", default="A [CLASS]")
    train.add_argument("--validation_prompt", default=None)
    train.set_defaults(func=_train)

    generate = subparsers.add_parser("generate", help="Generate adapted target-domain images.")
    _add_common(generate)
    generate.add_argument("--generation", choices=["pure_noise", "ddim_inversion"], required=True)
    generate.add_argument("--checkpoint_dir", default=None)
    generate.set_defaults(func=_generate)

    bridge = subparsers.add_parser("bridge", help="Generate the 25 DDIM + 5 class-conditioned groups and pool them for UDA.")
    _add_common(bridge)
    bridge.add_argument("--checkpoint_dir", required=True)
    bridge.add_argument("--seed", type=int, default=0)
    bridge.add_argument("--num_images_per_class", type=int, default=50, help="Class-conditioned images per class in each adapter group.")
    bridge.add_argument("--max_images_per_class", type=int, default=0, help="Source images per class for each DDIM group; 0 uses all images.")
    bridge.add_argument("--num_inference_steps", type=int, default=50)
    bridge.add_argument("--num_inversion_steps", type=int, default=50)
    bridge.add_argument("--guidance_scale", type=float, default=7.5)
    bridge.add_argument("--generation_guidance_scale", type=float, default=1.0, help="DDIM translation guidance.")
    bridge.add_argument("--inversion_guidance_scale", type=float, default=1.0)
    bridge.add_argument("--semantic_scale", type=float, default=0.8)
    bridge.add_argument("--style_scale", type=float, default=2.0)
    bridge.add_argument("--resolution", type=int, default=1024, help="DDIM source long-side resolution.")
    bridge.add_argument("--height", type=int, default=1024)
    bridge.add_argument("--width", type=int, default=1024)
    bridge.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    bridge.add_argument("--device", default=None)
    bridge.set_defaults(func=_bridge)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    return args.func(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
