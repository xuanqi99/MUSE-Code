"""Load an ImageFolder, optionally restricted to an official image-list split."""

from pathlib import Path
import re

from torchvision.datasets import ImageFolder


def remap_class_indices(class_to_idx, reference):
    def normalized(name):
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    reference_names = {}
    for name, index in reference.items():
        key = normalized(name)
        if key in reference_names:
            raise ValueError(f"Ambiguous source class names after normalization: {name}")
        reference_names[key] = index
    remap = {}
    seen = set()
    for name, index in class_to_idx.items():
        key = normalized(name)
        if key not in reference_names:
            raise ValueError(f"Class {name!r} is absent from the source class mapping")
        if key in seen:
            raise ValueError(f"Ambiguous class names after normalization: {name}")
        seen.add(key)
        remap[index] = reference_names[key]
    missing = set(reference_names) - seen
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} source classes: {sorted(missing)[:5]}")
    return remap


def load_imagefolder(root, transform=None, split_file=None):
    dataset = ImageFolder(root, transform=transform)
    if not split_file:
        return dataset

    domain_root = Path(root).resolve()
    available = {Path(path).resolve(): (path, label) for path, label in dataset.samples}
    samples = []
    seen = set()
    split_labels = {}
    split_classes = {}
    with open(split_file, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                relative_path, label_text = line.strip().rsplit(maxsplit=1)
                split_label = int(label_text)
            except ValueError as exc:
                raise ValueError(f"{split_file}:{line_number}: expected 'image_path integer_label'") from exc
            relative_path = Path(relative_path.replace("\\", "/"))
            candidates = [domain_root / relative_path, domain_root.parent / relative_path]
            matched = next((path.resolve() for path in candidates if path.resolve() in available), None)
            if matched is None:
                raise FileNotFoundError(
                    f"{split_file}:{line_number}: {relative_path} is not an image under {domain_root}"
                )
            if matched in seen:
                raise ValueError(f"{split_file}:{line_number}: repeated image {relative_path}")
            seen.add(matched)
            sample = available[matched]
            class_name = dataset.classes[sample[1]]
            if split_labels.setdefault(class_name, split_label) != split_label:
                raise ValueError(f"{split_file}:{line_number}: inconsistent label for {class_name}")
            if split_classes.setdefault(split_label, class_name) != class_name:
                raise ValueError(f"{split_file}:{line_number}: label {split_label} names multiple classes")
            samples.append(sample)
    if not samples:
        raise ValueError(f"No images listed in {split_file}")

    # Class names align generated and real domains; split IDs need not equal
    # ImageFolder's alphabetical IDs (e.g. miniDomainNet's 126-class subset).
    selected = sorted({dataset.classes[label] for _, label in samples})
    class_to_idx = {name: index for index, name in enumerate(selected)}
    dataset.samples = [(path, class_to_idx[dataset.classes[label]]) for path, label in samples]
    dataset.classes = selected
    dataset.class_to_idx = class_to_idx
    dataset.targets = [label for _, label in dataset.samples]
    dataset.imgs = dataset.samples
    return dataset
