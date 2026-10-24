"""Class names and manifests shared by the two synthesis entry points."""

import hashlib
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prompt_class(class_name):
    return class_name.replace("_", " ").replace("-", " ").lower()


def source_classes(data_root, source_domain, dataset, split_dir=None):
    """Keep folder names intact: ImageFolder sorts these names to assign labels."""
    root = Path(data_root)
    if dataset == "minidomainnet":
        split_root = Path(split_dir) if split_dir else root / "splits_mini"
        split_file = split_root / f"{source_domain}_train.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        classes = set()
        with split_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rel_path = line.rsplit(maxsplit=1)[0].strip().replace("\\", "/")
                    classes.add(Path(rel_path).parent.name)
        return sorted(classes)

    domain_root = root / source_domain
    if dataset == "office31" and (domain_root / "images").is_dir():
        domain_root /= "images"
    if not domain_root.is_dir():
        raise FileNotFoundError(domain_root)
    return sorted(
        folder.name for folder in domain_root.iterdir()
        if folder.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in folder.rglob("*"))
    )


def stable_image_id(relative_path):
    return hashlib.sha256(str(relative_path).replace("\\", "/").encode("utf-8")).hexdigest()[:16]


def checkpoint_identity(path):
    root = Path(path)
    candidates = [root] if root.is_file() else [
        root / "pytorch_lora_weights.safetensors", root / "pytorch_lora_weights.bin"
    ]
    for candidate in candidates:
        if candidate.is_file():
            stat = candidate.stat()
            return {"path": str(candidate.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    raise FileNotFoundError(f"No MUSE LoRA weights found at {path}")


def load_lora_state(path):
    identity = checkpoint_identity(path)
    checkpoint = Path(identity["path"])
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device="cpu")
    else:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    normalized = {}
    for key, value in state.items():
        while key.startswith("unet.unet."):
            key = key[len("unet."):]
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint key after prefix normalization: {key}")
        normalized[key] = value
    return normalized


def check_target_order(checkpoint, target_domains):
    path = Path(checkpoint)
    config_file = (path if path.is_dir() else path.parent) / "muse_config.json"
    if config_file.is_file():
        with config_file.open(encoding="utf-8") as handle:
            config = json.load(handle)
        trained_order = config.get("target_domains")
        if isinstance(trained_order, str):
            trained_order = [name.strip() for name in trained_order.split(",") if name.strip()]
        if trained_order and trained_order != target_domains:
            raise ValueError(f"Target order must match the checkpoint: {trained_order}; got {target_domains}.")


def write_config(directory, config, filename="generation_config.json", overwrite=False):
    """Reject reuse of a run/cache directory for a different synthesis setting."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if filename == "generation_config.json" and not path.exists():
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in directory.rglob("*")):
            raise ValueError(f"Existing images in {directory} have no generation settings; use a separate output directory.")
    if path.exists() and not overwrite:
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) != config:
                raise ValueError(f"Settings differ from {path}; use a separate output/cache directory.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def begin_target(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "completion.json").unlink(missing_ok=True)


def complete_target(directory, expected_images, classes):
    directory = Path(directory)
    metadata = (directory / "metadata.jsonl").read_bytes()
    record_count = len(metadata.splitlines())
    if record_count != expected_images:
        raise RuntimeError(f"Expected {expected_images} image records, found {record_count} in {directory}")
    config = (directory.parent / "generation_config.json").read_bytes()
    write_config(directory, {
        "schema": 1, "expected_images": expected_images, "image_count": record_count,
        "metadata_records": record_count, "classes": sorted(classes),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "generation_config_sha256": hashlib.sha256(config).hexdigest(),
    }, filename="completion.json", overwrite=True)
