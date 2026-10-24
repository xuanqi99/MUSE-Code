"""Pair DDIM and class-conditioned images into 25 bridge datasets per target."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path

try:
    from .muse import bridge_groups, bridge_pairs
except ImportError:
    from muse import bridge_groups, bridge_pairs


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_domain(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def validated_group(root: Path, image_root: Path, plan: dict, group: dict, target: str) -> list[Path]:
    directory = image_root.parent
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    metadata_bytes = (directory / "metadata.jsonl").read_bytes()
    config_bytes = (directory.parent / "generation_config.json").read_bytes()
    if (hashlib.sha256(metadata_bytes).hexdigest() != completion["metadata_sha256"]
            or hashlib.sha256(config_bytes).hexdigest() != completion["generation_config_sha256"]):
        raise ValueError(f"Generation metadata/configuration changed after completion: {directory}")
    config = json.loads(config_bytes)
    mode = "class_conditioned" if group["generation"] == "pure_noise" else "ddim_inversion"
    required = {"mode": mode, "lora_scale": 1.0,
                "semantic_scale": group["semantic_scale"], "style_scale": 1.0}
    for key in ["dataset", "seed", "source_domain", "num_inference_steps"]:
        plan_key = "source" if key == "source_domain" else key
        if plan_key in plan:
            required[key] = plan[plan_key]
    required["target_domains"] = ",".join(plan["targets"])
    if mode == "ddim_inversion":
        required["inversion_end_ratio"] = group["inversion_strength"]
        required["inversion_lora_scale"] = 1.0
    else:
        required["num_images_per_class"] = plan["num_images_per_class"]
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f"Group settings do not match the bridge plan: {directory}")
    if "checkpoint" in plan:
        checkpoint = Path(plan["checkpoint"]).resolve()
        recorded = Path(config["checkpoint"]["path"]).resolve()
        if recorded != checkpoint and recorded.parent != checkpoint:
            raise ValueError(f"Group checkpoint does not match the bridge plan: {directory}")

    rows = [json.loads(line) for line in metadata_bytes.splitlines() if line.strip()]
    expected = completion["expected_images"]
    if not rows or len(rows) != expected or completion["image_count"] != expected or completion["metadata_records"] != expected:
        raise ValueError(f"Incomplete generation metadata: {directory}")
    images = []
    classes = sorted(completion["classes"])
    for row in rows:
        sample = directory / row["image"]
        try:
            relative = sample.resolve().relative_to(image_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Image is outside the group ImageFolder: {sample}") from exc
        if (not sample.is_file() or len(relative.parts) < 2 or relative.parts[0] != row["class_name"]
                or row["class_name"] not in classes or row["label"] != classes.index(row["class_name"])
                or row["target_domain"] != target or row["mode"] != mode
                or row["adapter_strength"] != 1.0 or row["style_scale"] != 1.0
                or row["semantic_scale"] != group["semantic_scale"]):
            raise ValueError(f"Invalid generated image record: {sample}")
        if mode == "ddim_inversion" and row["inversion_strength"] != group["inversion_strength"]:
            raise ValueError(f"Inversion strength differs from the group plan: {sample}")
        images.append(sample)
    actual = {p.resolve() for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    if len(set(images)) != len(images) or actual != {p.resolve() for p in images}:
        raise ValueError(f"Images differ from the completed generation manifest: {directory}")
    if sorted({p.relative_to(image_root).parts[0] for p in images}) != classes:
        raise ValueError(f"Completion class vocabulary differs from the images: {directory}")
    return images


def pooled_row(root, output, image_root, sample, group, target, class_to_idx, bridge):
    relative = sample.relative_to(image_root)
    destination = output / "images" / relative.parts[0] / group["name"] / Path(*relative.parts[1:])
    stat = sample.stat()
    return destination, {
        "file_name": destination.relative_to(output).as_posix(),
        "label": class_to_idx[relative.parts[0]], "class_name": relative.parts[0],
        "target_domain": target, "generation": group["generation"],
        "inversion_strength": group["inversion_strength"], "adapter_strength": group["adapter_strength"],
        "semantic_scale": group["semantic_scale"], "style_scale": 1.0,
        "bridge": bridge["name"], "bridge_inversion_strength": bridge["inversion_strength"],
        "group_image": sample.relative_to(root).as_posix(),
        "group_metadata": (image_root.parent / "metadata.jsonl").relative_to(root).as_posix(),
        "source_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
    }


def identical_copy(source, destination):
    if source.stat().st_size != destination.stat().st_size:
        return False
    with source.open("rb") as left, destination.open("rb") as right:
        while True:
            chunk = left.read(1024 * 1024)
            if chunk != right.read(1024 * 1024):
                return False
            if not chunk:
                return True


def prepare(root: Path, link_method: str = "hardlink") -> dict:
    root = root.resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if (plan.get("schema") != 2 or plan.get("groups") != bridge_groups()
            or plan.get("bridges") != bridge_pairs()
            or plan.get("lora_scale") != 1.0 or plan.get("style_scale") != 1.0):
        raise ValueError("Expected the 5 x 5 inversion/semantic plan with lora_scale=style_scale=1.0. Use a new output directory.")

    # Validate every group before creating the pooled dataset.
    inputs = {}
    expected_classes = None
    for target in plan["targets"]:
        domain = safe_domain(target)
        group_inputs = {}
        counts_by_method = {}
        for group in plan["groups"]:
            image_root = root / "groups" / group["name"] / domain / "images"
            if not image_root.is_dir():
                raise FileNotFoundError(f"Missing generation group: {image_root}")
            class_dirs = sorted((p for p in image_root.iterdir() if p.is_dir()), key=lambda p: p.name)
            classes = [p.name for p in class_dirs]
            if expected_classes is None:
                expected_classes = classes
            if not classes or classes != expected_classes:
                raise ValueError(f"Class vocabulary differs in {image_root}")
            images = validated_group(root, image_root, plan, group, target)
            counts = {}
            for class_dir in class_dirs:
                samples = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
                if not samples:
                    raise ValueError(f"Empty generated class: {class_dir}")
                counts[class_dir.name] = len(samples)
            method = group["generation"]
            if method in counts_by_method and counts != counts_by_method[method]:
                raise ValueError(f"Incomplete group or unequal counts: {image_root}")
            counts_by_method[method] = counts
            if method == "pure_noise" and any(n != plan["num_images_per_class"] for n in counts.values()):
                raise ValueError(f"Class-conditioned counts do not match plan: {image_root}")
            group_inputs[group["name"]] = (group, image_root, images)
        inputs[target] = group_inputs

    report = {"schema": 2, "dataset": plan["dataset"], "task": plan["task"],
              "source": plan["source"], "targets": {}}
    class_to_idx = {name: index for index, name in enumerate(expected_classes)}
    for target, available in inputs.items():
        report["targets"][target] = {"bridges": {}}
        for bridge in plan["bridges"]:
            group_inputs = [available[name] for name in bridge["groups"]]
            output = root / "uda" / safe_domain(target) / bridge["name"]
            count = assemble(root, output, group_inputs, target, class_to_idx, bridge, link_method)
            report["targets"][target]["bridges"][bridge["name"]] = {
                "images": count, "classes": len(expected_classes),
                "path": (output / "images").relative_to(root).as_posix(),
                "inversion_strength": bridge["inversion_strength"],
                "semantic_scale": bridge["semantic_scale"],
            }
    temporary = root / "bridge.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "bridge.json")
    return report


def assemble(root, output, group_inputs, target, class_to_idx, bridge, link_method):
    image_output = output / "images"
    manifest = output / "metadata.jsonl"
    rows = (pooled_row(root, output, image_root, sample, group, target, class_to_idx, bridge)
            for group, image_root, images in group_inputs for sample in images)
    count = 0
    if manifest.exists():
        with manifest.open(encoding="utf-8") as stream:
            stored = (json.loads(line) for line in stream if line.strip())
            for stored_row, expected in itertools.zip_longest(stored, rows):
                if expected is None or stored_row != expected[1]:
                    raise ValueError(f"Existing bridge does not match generation inputs: {output}")
                destination, row = expected
                source = root / row["group_image"]
                if not destination.is_file() or not (os.path.samefile(source, destination) or identical_copy(source, destination)):
                    raise ValueError(f"Bridge image differs from its generation input: {destination}")
                count += 1
        mapping = json.loads((output / "classes.json").read_text(encoding="utf-8"))
        if mapping != class_to_idx:
            raise ValueError(f"Bridge class indices differ: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "metadata.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for destination, row in rows:
                source = root / row["group_image"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if not os.path.samefile(source, destination) and not (link_method == "copy" and identical_copy(source, destination)):
                        raise FileExistsError(f"Refusing to replace an existing sample: {destination}")
                elif link_method == "copy":
                    shutil.copy2(source, destination)
                else:
                    os.link(source, destination)
                stream.write(json.dumps(row) + "\n")
                count += 1
        (output / "classes.json").write_text(json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8")
        actual_count = sum(1 for p in image_output.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if actual_count != count:
            raise ValueError(f"Unexpected images in bridge: {output}")
        temporary.replace(manifest)
    actual_count = sum(1 for p in image_output.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if actual_count != count:
        raise ValueError(f"Unexpected images in bridge: {output}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Directory containing plan.json and groups/.")
    parser.add_argument("--link_method", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.link_method), indent=2))


if __name__ == "__main__":
    main()
