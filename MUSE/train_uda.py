"""Train ELS or SSRT separately on each inversion/semantic bridge pair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:
    from .muse import DOMAIN_PRESETS, TASK_PRESETS, bridge_pairs
    from .prepare_bridge import IMAGE_EXTS, safe_domain
except ImportError:
    from muse import DOMAIN_PRESETS, TASK_PRESETS, bridge_pairs
    from prepare_bridge import IMAGE_EXTS, safe_domain


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "--source-train-dir", "--source-train-dirs", "--target-train-dir", "--target-test-dir",
    "--target-train-list", "--target-test-list", "--source", "--target", "--root",
    "--log", "--phase", "--checkpoint", "--weight_path",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bridge_root", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--method", choices=["els", "ssrt"], required=True)
    parser.add_argument("--log_root", type=Path, default=Path("logs"))
    parser.add_argument("--minidomainnet_split_dir", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def commands(args: argparse.Namespace, extra: Sequence[str]) -> list[list[str]]:
    for option in extra:
        name = option.split("=", 1)[0]
        if name in {"-s", "-t"} or (name.startswith("--") and any(name == protected or protected.startswith(name) for protected in PROTECTED)):
            raise ValueError(f"{name} is set separately for each bridge and cannot be overridden.")
    root = args.bridge_root.resolve()
    report = json.loads((root / "bridge.json").read_text(encoding="utf-8"))
    if report.get("schema") != 2:
        raise ValueError("Expected 25 independent bridge pairs; regenerate with the current bridge command.")
    dataset, task = report["dataset"], report["task"]
    if dataset not in TASK_PRESETS or task not in TASK_PRESETS[dataset]:
        raise ValueError("Unknown dataset/task in bridge.json.")
    source_key, _ = TASK_PRESETS[dataset][task]
    if report["source"] != DOMAIN_PRESETS[dataset][source_key]:
        raise ValueError("Bridge source does not match its task.")
    domains = set(DOMAIN_PRESETS[dataset].values()) - {report["source"]}
    if not report["targets"] or not set(report["targets"]).issubset(domains):
        raise ValueError("Invalid target domains in bridge.json.")
    pairs = bridge_pairs()
    expected_names = {pair["name"] for pair in pairs}
    jobs = []
    reference_classes = None
    for target, target_info in report["targets"].items():
        bridges = target_info["bridges"]
        if set(bridges) != expected_names:
            raise ValueError(f"Expected all 25 inversion/semantic bridges for {target}.")
        real = args.data_root.resolve() / target
        if dataset == "office31" and (real / "images").is_dir():
            real /= "images"
        splits = []
        if dataset == "minidomainnet":
            split_root = args.minidomainnet_split_dir or args.data_root / "splits_mini"
            splits = ["--target-train-list", str((split_root / f"{target}_train.txt").resolve()),
                      "--target-test-list", str((split_root / f"{target}_test.txt").resolve())]
            if not args.dry_run:
                for path in splits[1::2]:
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
        if not args.dry_run and not real.is_dir():
            raise FileNotFoundError(real)
        for pair in pairs:
            entry = bridges[pair["name"]]
            relative = Path("uda") / safe_domain(target) / pair["name"] / "images"
            if (entry["path"] != relative.as_posix()
                    or entry["inversion_strength"] != pair["inversion_strength"]
                    or entry["semantic_scale"] != pair["semantic_scale"]):
                raise ValueError(f"Bridge path or strengths do not match: {target}/{pair['name']}")
            images = root / relative
            if not args.dry_run:
                classes = json.loads((images.parent / "classes.json").read_text(encoding="utf-8"))
                if reference_classes is None:
                    reference_classes = classes
                if (not images.is_dir() or classes != reference_classes or len(classes) != entry["classes"]
                        or sorted(classes.values()) != list(range(len(classes)))):
                    raise ValueError(f"Bridge class mapping differs: {images}")
                count = sum(1 for p in images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
                if count <= 0 or count != entry["images"]:
                    raise ValueError(f"Bridge image count differs: {images}")
            script = ROOT / "UDA" / ("ELS/cdan_mcc_sdat.py" if args.method == "els" else "SSRT/ssrt_sdat.py")
            command = [sys.executable, str(script)]
            if args.method == "els":
                command.append(str(args.data_root.resolve()))
            log = args.log_root.resolve() / args.method / f"{dataset}_{task}" / safe_domain(target) / pair["name"]
            command.extend(["--source-train-dir", str(images), "--target-train-dir", str(real),
                            "--target-test-dir", str(real), "--log", str(log), "--phase", "train"])
            command.extend(splits)
            command.extend(extra)
            jobs.append(command)
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    jobs = commands(args, extra)
    print(f"{len(jobs)} UDA jobs ({args.method}), one model per bridge and target.", flush=True)
    for command in jobs:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
