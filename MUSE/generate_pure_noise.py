import argparse
import gc
import os
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

try:
    from .generation_io import begin_target, check_target_order, checkpoint_identity, complete_target, load_lora_state, prompt_class, source_classes, write_config, write_record
except ImportError:
    from generation_io import begin_target, check_target_order, checkpoint_identity, complete_target, load_lora_state, prompt_class, source_classes, write_config, write_record


OFFICEHOME_CLASSES = [
    "drill", "exit sign", "bottle", "glasses", "computer", "file cabinet", "shelf", "toys", "sink",
    "laptop", "kettle", "folder", "keyboard", "flipflops", "pencil", "bed", "hammer", "toothbrush",
    "couch", "bike", "postit notes", "mug", "webcam", "desk lamp", "telephone", "helmet", "mouse", "pen",
    "monitor", "mop", "sneakers", "notebook", "backpack", "alarm clock", "push pin", "paper clip",
    "batteries", "radio", "fan", "ruler", "pan", "screwdriver", "trash can", "printer", "speaker",
    "eraser", "bucket", "chair", "calendar", "calculator", "flowers", "lamp shade", "spoon", "candles",
    "clipboards", "scissors", "tv", "curtains", "fork", "soda", "table", "knives", "oven",
    "refrigerator", "marker",
]

OFFICE31_CLASSES = [
    "back pack", "bike", "bike helmet", "bookcase", "bottle", "calculator", "desk chair",
    "desk lamp", "desktop computer", "file cabinet", "headphones", "keyboard", "laptop computer",
    "letter tray", "mobile phone", "monitor", "mouse", "mug", "paper notebook", "pen", "phone",
    "printer", "projector", "punchers", "ring binder", "ruler", "scissors", "speaker", "stapler",
    "tape dispenser", "trash can",
]


def _normalize_lora_path(path: str) -> str:
    if os.path.isdir(path):
        for fname in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
            candidate = os.path.join(path, fname)
            if os.path.exists(candidate):
                return candidate
    return path


def target_layer_from_sd_name(k: str) -> str:
    if ".processor.to_" in k:
        target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
        target_layer = target_layer.replace("to_out", "to_out[0]")
    else:
        target_layer = k.split(".lora.")[0]

    for i in range(10):
        target_layer = target_layer.replace(f".{i}", f"[{i}]")

    if target_layer.startswith("unet."):
        target_layer = target_layer[len("unet.") :]

    return target_layer


def _resolve_module_path(root: object, path: str) -> object:
    cur = root
    for raw_part in path.split("."):
        part = raw_part
        bracket_index = None

        if "[" in part and part.endswith("]"):
            base, idx_str = part[:-1].split("[", 1)
            part = base
            bracket_index = int(idx_str)

        if part:
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)

        if bracket_index is not None:
            cur = cur[bracket_index]

    return cur


def _sanitize_name(s: str) -> str:
    return s.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_minidomainnet_split_dir(args) -> str:
    candidates = []
    if args.minidomainnet_split_dir:
        candidates.append(args.minidomainnet_split_dir)
    if args.data_root:
        candidates.append(os.path.join(args.data_root, "splits_mini"))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise ValueError(
        "miniDomainNet generation requires a valid split directory. "
        "Please pass --minidomainnet_split_dir or --data_root that contains splits_mini/."
    )


def _load_minidomainnet_classes(split_dir: str) -> List[str]:
    classes = set()
    for fname in os.listdir(split_dir):
        if not fname.endswith("_train.txt"):
            continue
        split_path = os.path.join(split_dir, fname)
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.split()[0]
                parts = rel_path.split("/")
                if len(parts) >= 3:
                    classes.add(parts[-2])
    classes = sorted(classes)
    if not classes:
        raise RuntimeError(f"No classes could be parsed from miniDomainNet split dir: {split_dir}")
    return classes


def inject_style_lora_into_unet(
    pipe: StableDiffusionXLPipeline,
    lora_path_or_dir: str,
    target_style_idx: int,
    lora_scale: float = 1.0,
    semantic_scale: float = 0.8,
    style_scale: float = 2.0,
    debug: bool = False,
) -> None:
    lora_path = _normalize_lora_path(lora_path_or_dir)
    lora_state_dict = load_lora_state(lora_path)

    device = next(pipe.unet.parameters()).device
    loaded = 0
    skipped = 0
    style_branch_layers = 0
    style_branch_missing = 0
    style_branch_delta_norm = 0.0

    for down_key, down_weight in lora_state_dict.items():
        if not down_key.endswith("down.weight"):
            continue
        if not down_key.startswith("unet."):
            continue
        # Skip style projection keys; they are consumed through style_cores path only.
        if "style_down.weight" in down_key:
            continue

        up_key = down_key.replace("down.weight", "up.weight")
        if up_key not in lora_state_dict:
            skipped += 1
            continue

        style_down_key = down_key.replace("down.weight", "style_down.weight")
        style_up_key = down_key.replace("down.weight", "style_up.weight")
        style_cores_key = down_key.replace("down.weight", "style_cores")

        sem_down = down_weight.to(device=device, dtype=torch.float32)
        sem_up = lora_state_dict[up_key].to(device=device, dtype=torch.float32)
        sem_delta = sem_up @ sem_down
        dW = sem_delta * float(semantic_scale)

        has_style = (
            style_down_key in lora_state_dict
            and style_up_key in lora_state_dict
            and style_cores_key in lora_state_dict
        )

        if has_style:
            style_down = lora_state_dict[style_down_key].to(device=device, dtype=torch.float32)
            style_up = lora_state_dict[style_up_key].to(device=device, dtype=torch.float32)
            style_cores = lora_state_dict[style_cores_key].to(device=device, dtype=torch.float32)

            if style_cores.ndim == 3 and 0 <= target_style_idx < style_cores.shape[0]:
                core = style_cores[target_style_idx]
                style_delta = style_up @ core @ style_down
                style_delta = style_delta * float(style_scale)
                dW = dW + style_delta
                style_branch_layers += 1
                style_branch_delta_norm += float(torch.linalg.vector_norm(style_delta).item())
            elif debug:
                print(f"[warn] style index {target_style_idx} invalid for {style_cores_key}, shape={tuple(style_cores.shape)}")
        else:
            if target_style_idx is not None:
                style_branch_missing += 1

        dW = dW * float(lora_scale)

        target_layer = target_layer_from_sd_name(down_key)
        try:
            mod = _resolve_module_path(pipe.unet, target_layer)
        except Exception as e:
            skipped += 1
            if debug:
                print(f"[skip] resolve failed: {down_key} -> {target_layer} ({e})")
            continue

        if not hasattr(mod, "weight"):
            skipped += 1
            if debug:
                print(f"[skip] no weight on module: {target_layer}")
            continue

        if tuple(mod.weight.shape) != tuple(dW.shape):
            skipped += 1
            if debug:
                print(f"[skip] shape mismatch: {target_layer}, dW={tuple(dW.shape)}, W={tuple(mod.weight.shape)}")
            continue

        mod.weight.data = mod.weight.data + dW.to(dtype=mod.weight.dtype)
        loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "No LoRA layers were injected. Please check checkpoint path and LoRA key format."
        )

    if target_style_idx is not None and style_branch_layers == 0:
        raise RuntimeError(
            "No style branch tensors were applied for this target style. "
            "This means the checkpoint likely does not contain `style_down/style_up/style_cores`, "
            "or key naming does not match. Generation would collapse to identical outputs across targets."
        )

    print(
        f"[inject] style_idx={target_style_idx}, loaded={loaded}, skipped={skipped}, "
        f"style_layers={style_branch_layers}, style_missing={style_branch_missing}, "
        f"style_delta_norm_sum={style_branch_delta_norm:.4f}"
    )


def _build_pipeline(args, torch_dtype: torch.dtype) -> StableDiffusionXLPipeline:
    if args.pretrained_vae_model_name_or_path:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=torch_dtype,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
        )

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def _parse_classes(args) -> List[str]:
    if args.classes:
        classes = [x.strip() for x in args.classes.split(",") if x.strip()]
        if not classes:
            raise ValueError("--classes is empty after parsing.")
        return classes

    if args.data_root and args.source_domain:
        classes = source_classes(args.data_root, args.source_domain, args.dataset, args.minidomainnet_split_dir)
        if not classes:
            raise ValueError("No source classes were found.")
        return classes

    dataset = args.dataset.lower()
    if dataset == "officehome":
        return OFFICEHOME_CLASSES
    if dataset == "office31":
        return OFFICE31_CLASSES
    if dataset == "minidomainnet":
        split_dir = _resolve_minidomainnet_split_dir(args)
        return _load_minidomainnet_classes(split_dir)

    raise ValueError("Custom generation requires explicit --classes.")


def _render_prompt(prompt_template: str, cls_name: str) -> str:
    # Keep compatibility with training-time validation prompt style: "A [CLASS]".
    prompt_tmpl = prompt_template.replace("[CLASS]", "{cls}")
    return prompt_tmpl.format(cls=cls_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate class images for each target style from a MUSE LoRA checkpoint.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory produced by MUSE fine-tuning.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default="officehome",
        choices=["officehome", "office31", "minidomainnet", "custom"],
        help="Dataset preset used to infer the class list when --classes is not provided.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional dataset root used for automatic class discovery (currently used by miniDomainNet).",
    )
    parser.add_argument(
        "--minidomainnet_split_dir",
        type=str,
        default=None,
        help="Path to miniDomainNet splits_mini directory. Used when --dataset=minidomainnet.",
    )
    parser.add_argument("--source_domain", default=None, help="Source domain used to preserve dataset class folder names.")
    parser.add_argument("--target_domains", type=str, default="Clipart,Product,Real World", help="Comma-separated target styles in training order.")
    parser.add_argument("--classes", type=str, default=None, help="Optional comma-separated classes. Overrides dataset-based class discovery.")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="A [CLASS]",
        help="Prompt template used in fine-tuning-compatible style. Supports [CLASS] or {cls}.",
    )

    parser.add_argument("--num_images_per_class", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--semantic_scale", type=float, default=0.8, help="Scale of semantic LoRA branch during generation.")
    parser.add_argument("--style_scale", type=float, default=2.0, help="Scale of target-style LoRA branch during generation.")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if ("{cls}" not in args.prompt_template) and ("[CLASS]" not in args.prompt_template):
        raise ValueError("--prompt_template must contain '{cls}' or '[CLASS]', e.g. 'A [CLASS]'.")

    classes = _parse_classes(args)
    target_domains = [x.strip() for x in args.target_domains.split(",") if x.strip()]
    if len(target_domains) == 0:
        raise ValueError("No target domains found in --target_domains.")
    check_target_order(args.checkpoint_dir, target_domains)
    if args.num_images_per_class <= 0 or args.num_inference_steps <= 0:
        raise ValueError("Image count and inference steps must be positive.")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique.")
    if any(Path(cls).name != cls or cls in {".", ".."} for cls in classes):
        raise ValueError("Class names must be individual directory names.")

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if args.device.startswith("cpu") and torch_dtype != torch.float32:
        print("[warn] CPU only supports fp32 reliably, overriding dtype to fp32.")
        torch_dtype = torch.float32

    config = dict(vars(args))
    config.update({"mode": "class_conditioned", "classes": sorted(classes), "checkpoint": checkpoint_identity(args.checkpoint_dir)})
    write_config(args.output_dir, config)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}

    print(f"[info] classes={len(classes)}, target_domains={target_domains}, images_per_class={args.num_images_per_class}")

    for style_idx, target_domain in enumerate(target_domains):
        print(f"\n[info] Generating style {style_idx}: {target_domain}")

        pipe = _build_pipeline(args, torch_dtype=torch_dtype)
        inject_style_lora_into_unet(
            pipe,
            lora_path_or_dir=args.checkpoint_dir,
            target_style_idx=style_idx,
            lora_scale=args.lora_scale,
            semantic_scale=args.semantic_scale,
            style_scale=args.style_scale,
            debug=args.debug,
        )

        style_dir = Path(args.output_dir) / _sanitize_name(target_domain)
        image_dir = style_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        begin_target(style_dir)
        metadata = (style_dir / "metadata.jsonl").open("w", encoding="utf-8")

        for class_idx, cls_name in enumerate(classes):
            class_dir = image_dir / cls_name
            os.makedirs(class_dir, exist_ok=True)

            prompt_cls = prompt_class(cls_name)
            prompt = _render_prompt(args.prompt_template, prompt_cls)

            for i in range(args.num_images_per_class):
                seed = args.seed + style_idx * 1_000_000 + class_idx * 1_000 + i
                generator_device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(seed)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                )
                image = result.images[0]

                out_name = f"seed{seed}.png"
                out_path = class_dir / out_name
                image.save(out_path)
                write_record(metadata, {
                    "image": out_path.relative_to(style_dir).as_posix(),
                    "class_name": cls_name, "label": class_to_idx[cls_name],
                    "target_domain": target_domain, "style_index": style_idx,
                    "mode": "class_conditioned", "adapter_strength": args.lora_scale,
                    "semantic_scale": args.semantic_scale, "style_scale": args.style_scale,
                    "seed": seed, "prompt": prompt,
                })
                print(out_path)

        metadata.close()
        complete_target(style_dir, len(classes) * args.num_images_per_class, classes)
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
