import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline

try:
    from .generation_io import begin_target, check_target_order, checkpoint_identity, complete_target, load_lora_state, prompt_class, stable_image_id, write_config, write_record
except ImportError:
    from generation_io import begin_target, check_target_order, checkpoint_identity, complete_target, load_lora_state, prompt_class, stable_image_id, write_config, write_record


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAE_SPATIAL_FACTOR = 8


class DDIMFromPriorScheduler(DDIMScheduler):
    """Denoise a cached latent from its inversion endpoint on the same DDIM grid."""

    def set_timesteps(self, num_inference_steps, device=None):
        super().set_timesteps(num_inference_steps, device=device)
        endpoint = getattr(self, "prior_timestep", None)
        if endpoint is not None:
            if endpoint not in self.timesteps.tolist():
                raise ValueError("The inversion endpoint is absent from the denoising grid; use equal inversion and generation step counts.")
            self.timesteps = self.timesteps[self.timesteps <= endpoint]


def inversion_endpoint(scheduler, num_inference_steps, ratio, device=None):
    scheduler.set_timesteps(num_inference_steps, device=device)
    ascending = scheduler.timesteps.flip(0)
    pairs = len(ascending) - 1
    used = 0 if ratio == 0 else max(1, int(round(pairs * ratio)))
    return used, int(ascending[used].item())


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


def _resolve_data_root(args) -> Path:
    if args.data_root:
        return Path(args.data_root)
    if args.dataset == "officehome":
        return Path(args.officehome_root)
    raise ValueError(
        f"--data_root is required when --dataset={args.dataset}. "
        "For OfficeHome, the legacy --officehome_root argument can still be used."
    )


def _resolve_minidomainnet_split_dir(args) -> Path:
    candidates = []
    if args.minidomainnet_split_dir:
        candidates.append(Path(args.minidomainnet_split_dir))
    if args.data_root:
        candidates.append(Path(args.data_root) / "splits_mini")
    if args.officehome_root and args.dataset == "minidomainnet":
        candidates.append(Path(args.officehome_root) / "splits_mini")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not resolve miniDomainNet split dir. "
        "Please provide --minidomainnet_split_dir or a --data_root that contains splits_mini/."
    )


def _domain_split_stems(domain_name: str) -> List[str]:
    stems: List[str] = []
    raw = domain_name.strip()
    variants = [
        raw,
        raw.lower(),
        raw.replace(" ", "_"),
        raw.lower().replace(" ", "_"),
        raw.replace(" ", ""),
        raw.lower().replace(" ", ""),
        raw.replace("-", "_"),
        raw.lower().replace("-", "_"),
        raw.replace("-", "").replace(" ", ""),
        raw.lower().replace("-", "").replace(" ", ""),
    ]
    for stem in variants:
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def _resolve_domain_train_split(split_dir: Path, domain_name: str) -> Path:
    for stem in _domain_split_stems(domain_name):
        candidate = split_dir / f"{stem}_train.txt"
        if candidate.exists():
            return candidate
    tried = [str(split_dir / f"{stem}_train.txt") for stem in _domain_split_stems(domain_name)]
    raise FileNotFoundError(
        f"Could not find train split for domain '{domain_name}'. Tried:\n"
        + "\n".join([f"  - {p}" for p in tried])
    )


def inject_style_lora_into_unet(
    pipe: StableDiffusionXLPipeline,
    lora_path_or_dir: str,
    target_style_idx: Optional[int],
    lora_scale: float = 1.0,
    semantic_scale: float = 1.0,
    style_scale: float = 1.0,
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
    sem_branch_delta_norm = 0.0

    for down_key, down_weight in lora_state_dict.items():
        if not down_key.endswith("down.weight"):
            continue
        if not down_key.startswith("unet."):
            continue
        # IMPORTANT: skip style branch projection keys themselves.
        # We only want base semantic down/up as backbone, then add target-specific style via style_cores.
        # If style_down/style_up are treated as generic down/up here, all targets get an identical extra delta.
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
        sem_branch_delta_norm += float(torch.linalg.vector_norm(sem_delta).item())

        if target_style_idx is not None:
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
                    print(
                        f"[warn] style index {target_style_idx} invalid for {style_cores_key}, shape={tuple(style_cores.shape)}"
                    )
            else:
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
                print(
                    f"[skip] shape mismatch: {target_layer}, dW={tuple(dW.shape)}, W={tuple(mod.weight.shape)}"
                )
            continue

        mod.weight.data = mod.weight.data + dW.to(dtype=mod.weight.dtype)
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No LoRA layers were injected. Please check checkpoint path and key format.")

    if target_style_idx is not None and style_branch_layers == 0:
        raise RuntimeError(
            "No style branch tensors were applied for this target style. "
            "Checkpoint may not include `style_down/style_up/style_cores` or key naming mismatches. "
            "This would produce identical outputs across target styles."
        )

    style_name = "source-semantic-only" if target_style_idx is None else f"style_idx={target_style_idx}"
    sem_effective_norm = sem_branch_delta_norm * abs(float(semantic_scale))
    style_to_sem_ratio = style_branch_delta_norm / max(sem_effective_norm, 1e-12)
    print(
        f"[inject] {style_name}, loaded={loaded}, skipped={skipped}, "
        f"style_layers={style_branch_layers}, style_missing={style_branch_missing}, "
        f"style_delta_norm_sum={style_branch_delta_norm:.4f}, "
        f"sem_delta_norm_sum={sem_branch_delta_norm:.4f}, "
        f"sem_effective_norm_sum={sem_effective_norm:.4f}, "
        f"style_sem_ratio={style_to_sem_ratio:.4f}"
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

    pipe.scheduler = DDIMFromPriorScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_tqdm)
    return pipe


def _build_image_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _compute_paper_resize_hw(width: int, height: int, resolution: int) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    if width >= height:
        new_width = resolution
        new_height = max(1, int(round(height * float(resolution) / float(width))))
    else:
        new_height = resolution
        new_width = max(1, int(round(width * float(resolution) / float(height))))

    return new_height, new_width


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resize_long_side_to_resolution(image: Image.Image, resolution: int) -> Tuple[Image.Image, int, int]:
    width, height = image.size
    new_height, new_width = _compute_paper_resize_hw(width, height, resolution)
    resized = image.resize((new_width, new_height), resample=Image.BILINEAR)
    return resized, new_height, new_width


def _pad_tensor_for_model(image_tensor: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    model_height = _round_up_to_multiple(height, multiple)
    model_width = _round_up_to_multiple(width, multiple)

    pad_bottom = model_height - height
    pad_right = model_width - width
    if pad_bottom == 0 and pad_right == 0:
        return image_tensor, model_height, model_width

    padded = F.pad(image_tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return padded, model_height, model_width


def _scan_domain_images(domain_root: Path, strip_images_prefix: bool = False) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for p in sorted(domain_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        rel_parts = list(p.relative_to(domain_root).parts)
        if len(rel_parts) < 2:
            continue
        if strip_images_prefix and rel_parts[0].lower() == "images":
            rel_parts = rel_parts[1:]
        if len(rel_parts) < 2:
            continue
        class_name = rel_parts[0]
        items.append((p, class_name))
    return items


def _load_minidomainnet_source_images(data_root: Path, split_dir: Path, source_domain: str) -> List[Tuple[Path, str]]:
    split_file = _resolve_domain_train_split(split_dir, source_domain)
    items: List[Tuple[Path, str]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path = line.split()[0]
            img_path = data_root / rel_path
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image listed in miniDomainNet split does not exist:\n"
                    f"  split_file = {split_file}\n"
                    f"  rel_path   = {rel_path}\n"
                    f"  resolved   = {img_path}"
                )
            parts = Path(rel_path).parts
            if len(parts) < 3:
                continue
            class_name = parts[-2]
            items.append((img_path, class_name))
    return items




def _limit_items_per_class(
    items: List[Tuple[Path, str]], max_images_per_class: Optional[int]
) -> List[Tuple[Path, str]]:
    if max_images_per_class is None or int(max_images_per_class) <= 0:
        return items

    kept: List[Tuple[Path, str]] = []
    class_counts = {}
    for item_path, cls_name in items:
        count = class_counts.get(cls_name, 0)
        if count >= int(max_images_per_class):
            continue
        kept.append((item_path, cls_name))
        class_counts[cls_name] = count + 1
    return kept




def _limit_items_to_first_n_classes(
    items: List[Tuple[Path, str]], max_classes: Optional[int]
) -> List[Tuple[Path, str]]:
    if max_classes is None or int(max_classes) <= 0:
        return items

    kept_classes = set(sorted({cls_name for _, cls_name in items})[: int(max_classes)])
    return [(item_path, cls_name) for item_path, cls_name in items if cls_name in kept_classes]






def _get_add_time_ids(
    pipe: StableDiffusionXLPipeline,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
):
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    proj_dim = None
    if getattr(pipe, "text_encoder_2", None) is not None and getattr(pipe.text_encoder_2, "config", None) is not None:
        proj_dim = getattr(pipe.text_encoder_2.config, "projection_dim", None)

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=dtype,
        text_encoder_projection_dim=proj_dim,
    )
    return add_time_ids.to(device)


def _encode_image_to_latents(pipe: StableDiffusionXLPipeline, image_tensor: torch.Tensor, generator=None) -> torch.Tensor:
    if next(pipe.vae.parameters()).dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False):
        pipe.upcast_vae()
    vae_dtype = next(pipe.vae.parameters()).dtype
    image_tensor = image_tensor.to(device=pipe.device, dtype=vae_dtype)
    latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def ddim_invert_sdxl(
    pipe: StableDiffusionXLPipeline,
    image_tensor: torch.Tensor,
    prompt: str,
    num_inference_steps: int,
    inversion_end_ratio: float,
    guidance_scale: float,
    negative_prompt: Optional[str],
    generator=None,
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=pipe.device)

    latents = _encode_image_to_latents(pipe, image_tensor, generator=generator)
    latents = latents.to(dtype=next(pipe.unet.parameters()).dtype)

    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=[prompt],
        negative_prompt=[negative_prompt] if negative_prompt else None,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
    )

    add_time_ids = _get_add_time_ids(
        pipe,
        image_tensor.shape[-2],
        image_tensor.shape[-1],
        dtype=prompt_embeds.dtype,
        device=pipe.device,
    )

    asc_timesteps = torch.flip(scheduler.timesteps, dims=[0])
    used_pairs, _ = inversion_endpoint(scheduler, num_inference_steps, inversion_end_ratio, device=pipe.device)

    for idx in range(used_pairs):
        t = asc_timesteps[idx]
        t_next = asc_timesteps[idx + 1]

        latent_model_input = latents
        if do_cfg:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids_in = torch.cat([add_time_ids, add_time_ids], dim=0)
        else:
            encoder_hidden_states = prompt_embeds
            add_text_embeds = pooled_prompt_embeds
            add_time_ids_in = add_time_ids

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids_in},
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        alpha_t = scheduler.alphas_cumprod[int(t.item())].to(device=latents.device, dtype=latents.dtype)
        alpha_next = scheduler.alphas_cumprod[int(t_next.item())].to(device=latents.device, dtype=latents.dtype)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        pred_x0 = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

        latents = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * noise_pred

    return latents.detach()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Source-image DDIM inversion priors followed by multi-target MUSE style denoising."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default="officehome",
        choices=["officehome", "office31", "minidomainnet"],
        help="Dataset layout used to enumerate source images.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root. Required for Office31 and miniDomainNet. Optional alias for OfficeHome.",
    )
    parser.add_argument(
        "--officehome_root",
        type=str,
        default=None,
        help="Legacy OfficeHome root containing Art/Clipart/Product/Real World.",
    )
    parser.add_argument(
        "--minidomainnet_split_dir",
        type=str,
        default=None,
        help="Path to miniDomainNet splits_mini directory. Required when --dataset=minidomainnet unless data_root contains it.",
    )
    parser.add_argument("--source_domain", type=str, default="Art")
    parser.add_argument("--target_domains", type=str, default="Clipart,Product,Real World")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prior_cache_dir", type=str, default=None)

    parser.add_argument("--prompt_template", type=str, default="A {cls}")
    parser.add_argument(
        "--generation_prompt_template",
        type=str,
        default="A {cls}",
        help="Prompt used during stage-2 generation. Literal (e.g. 'An image') or template with {cls}.",
    )
    parser.add_argument(
        "--generation_prompt_mode",
        type=str,
        choices=["prior", "template"],
        default="prior",
        help="Use prompt saved in prior file (`prior`) or use `--generation_prompt_template` (`template`).",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument(
        "--inversion_num_inference_steps",
        type=int,
        default=None,
        help="Optional inversion-only DDIM steps. If unset, uses --num_inference_steps.",
    )
    parser.add_argument(
        "--inversion_end_ratio",
        type=float,
        default=1.0,
        help="Fraction of inversion trajectory to run (0~1). 1.0 means invert to the last DDIM timestep.",
    )

    parser.add_argument("--inversion_guidance_scale", type=float, default=1.0)
    parser.add_argument("--generation_guidance_scale", type=float, default=1.0)
    parser.add_argument("--cross_attention_scale", type=float, default=0.9)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--inversion_lora_scale", type=float, default=1.0, help="Semantic adapter scale during source inversion; independent of generation --lora_scale.")
    parser.add_argument("--semantic_scale", type=float, default=0.8, help="Scale of semantic LoRA branch during generation.")
    parser.add_argument("--style_scale", type=float, default=2, help="Scale of target-style LoRA branch during generation.")

    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_classes",
        type=int,
        default=None,
        help="Optional cap on how many classes to keep, taking the first N classes in sorted order. Useful for debug sweeps.",
    )
    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=None,
        help="Optional cap on how many source images/priors to use per class. Useful for debug sweeps.",
    )

    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--overwrite_prior", action="store_true")
    parser.add_argument(
        "--prior_only",
        action="store_true",
        help="Only build/reuse DDIM priors under --prior_cache_dir and skip stage-2 target-domain generation.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if device.startswith("cpu") and dtype != torch.float32:
        print("[warn] CPU only supports fp32 reliably, overriding dtype to fp32.")
        dtype = torch.float32
    return dtype


def _infer_output_hw_from_prior(prior_obj: dict, default_resolution: int) -> Tuple[int, int, int, int]:
    paper_height = prior_obj.get("paper_height", None)
    paper_width = prior_obj.get("paper_width", None)
    model_height = prior_obj.get("model_height", None)
    model_width = prior_obj.get("model_width", None)
    if (
        isinstance(paper_height, int)
        and isinstance(paper_width, int)
        and isinstance(model_height, int)
        and isinstance(model_width, int)
        and paper_height > 0
        and paper_width > 0
        and model_height > 0
        and model_width > 0
    ):
        return model_height, model_width, paper_height, paper_width

    latents = prior_obj.get("latents", None)
    if torch.is_tensor(latents) and latents.ndim >= 4:
        model_height = int(latents.shape[-2]) * VAE_SPATIAL_FACTOR
        model_width = int(latents.shape[-1]) * VAE_SPATIAL_FACTOR
        paper_height = prior_obj.get("processed_height", model_height)
        paper_width = prior_obj.get("processed_width", model_width)
        return model_height, model_width, paper_height, paper_width

    return default_resolution, default_resolution, default_resolution, default_resolution


def main():
    args = parse_args()
    if "{cls}" not in args.prompt_template:
        raise ValueError("--prompt_template must contain '{cls}', e.g. 'A {cls}'.")
    if not (0.0 <= args.inversion_end_ratio <= 1.0):
        raise ValueError("--inversion_end_ratio must be in [0, 1].")
    inversion_steps = args.inversion_num_inference_steps or args.num_inference_steps
    if args.resolution <= 0 or inversion_steps < 2:
        raise ValueError("Resolution must be positive and DDIM needs at least two steps.")
    if inversion_steps != args.num_inference_steps:
        raise ValueError("Inversion and denoising must use the same step count to match cached DDIM timesteps.")
    if args.center_crop:
        print("[warn] --center_crop is ignored; translation resizes the long side without cropping.")

    torch_dtype = _load_dtype(args.dtype, args.device)
    target_domains = [x.strip() for x in args.target_domains.split(",") if x.strip()]
    if not target_domains:
        raise ValueError("No target domains found in --target_domains.")
    check_target_order(args.checkpoint_dir, target_domains)
    data_root = _resolve_data_root(args)
    source_root = data_root / args.source_domain
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source domain path not found: {source_root}")
    if args.dataset == "minidomainnet":
        split_dir = _resolve_minidomainnet_split_dir(args)
        image_items = _load_minidomainnet_source_images(data_root, split_dir, args.source_domain)
    else:
        image_items = _scan_domain_images(source_root, strip_images_prefix=args.dataset == "office31")
    if not image_items:
        raise ValueError(f"No source images found under {source_root}")

    classes = sorted({cls for _, cls in image_items})
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    image_items = _limit_items_to_first_n_classes(image_items, args.max_classes)
    image_items = _limit_items_per_class(image_items, args.max_images_per_class)
    output_dir = Path(args.output_dir)
    prior_cache_dir = Path(args.prior_cache_dir) if args.prior_cache_dir else output_dir / "ddim_priors"
    checkpoint = checkpoint_identity(args.checkpoint_dir)
    cache_config = {
        "schema": 2, "model": args.pretrained_model_name_or_path,
        "vae": args.pretrained_vae_model_name_or_path, "checkpoint": checkpoint,
        "dataset": args.dataset, "source_domain": args.source_domain,
        "data_root": str(data_root.resolve()), "resolution": args.resolution,
        "inversion_steps": inversion_steps, "inversion_strength": args.inversion_end_ratio,
        "inversion_lora_scale": args.inversion_lora_scale,
        "guidance_scale": args.inversion_guidance_scale,
        "prompt_template": args.prompt_template, "negative_prompt": args.negative_prompt,
        "seed": args.seed, "dtype": args.dtype,
    }
    write_config(prior_cache_dir, cache_config, filename="cache_config.json", overwrite=args.overwrite_prior)
    fingerprint = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()
    run_config = {
        key: value for key, value in vars(args).items()
        if key not in {"output_dir", "prior_cache_dir", "overwrite_prior", "prior_only", "debug", "disable_tqdm"}
    }
    run_config.update({"schema": 2, "mode": "ddim_inversion", "classes": classes, "checkpoint": checkpoint})
    inventory = hashlib.sha256()
    for image_path, class_name in image_items:
        stat = image_path.stat()
        inventory.update(json.dumps([image_path.relative_to(source_root).as_posix(), class_name, stat.st_size, stat.st_mtime_ns]).encode())
    run_config["source_inventory"] = inventory.hexdigest()
    if not args.prior_only:
        write_config(output_dir, run_config)

    prior_meta = []
    inv_pipe = None
    image_tfm = _build_image_transform()
    for idx, (img_path, cls_name) in enumerate(image_items):
        relative = img_path.relative_to(source_root)
        image_id = stable_image_id(relative.as_posix())
        prior_path = prior_cache_dir / cls_name / f"{image_id}.pt"
        stat = img_path.stat()
        source_identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        prior_obj = None
        if prior_path.is_file() and not args.overwrite_prior:
            try:
                candidate = torch.load(prior_path, map_location="cpu", weights_only=True)
                if candidate.get("cache_fingerprint") == fingerprint and candidate.get("source_identity") == source_identity:
                    prior_obj = candidate
            except Exception as error:
                print(f"[prior] Rebuilding unreadable cache {prior_path}: {error}")
        if prior_obj is None:
            if inv_pipe is None:
                print("[stage 1/2] Loading source-semantic inversion model.")
                inv_pipe = _build_pipeline(args, torch_dtype)
                inject_style_lora_into_unet(
                    inv_pipe, args.checkpoint_dir, None,
                    lora_scale=args.inversion_lora_scale, semantic_scale=1.0, style_scale=0.0,
                    debug=args.debug,
                )
            with Image.open(img_path) as original:
                image, paper_height, paper_width = _resize_long_side_to_resolution(original.convert("RGB"), args.resolution)
            image_tensor, model_height, model_width = _pad_tensor_for_model(
                image_tfm(image).unsqueeze(0), VAE_SPATIAL_FACTOR
            )
            prompt = args.prompt_template.format(cls=prompt_class(cls_name))
            image_seed = (args.seed + int(image_id, 16)) % (2**63 - 1)
            generator = torch.Generator(device=inv_pipe.device).manual_seed(image_seed)
            latents = ddim_invert_sdxl(
                inv_pipe, image_tensor, prompt, inversion_steps, args.inversion_end_ratio,
                args.inversion_guidance_scale, args.negative_prompt, generator=generator,
            )
            _, endpoint = inversion_endpoint(
                DDIMScheduler.from_config(inv_pipe.scheduler.config),
                inversion_steps, args.inversion_end_ratio, device=inv_pipe.device,
            )
            prior_obj = {
                "latents": latents.detach().to("cpu", dtype=torch.float32),
                "class_name": cls_name, "prompt": prompt, "source_image": str(img_path),
                "source_identity": source_identity, "source_relative": relative.as_posix(),
                "image_id": image_id, "seed": image_seed, "cache_fingerprint": fingerprint,
                "inversion_timestep": endpoint, "paper_height": paper_height,
                "paper_width": paper_width, "model_height": model_height, "model_width": model_width,
            }
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_prior = prior_path.with_suffix(".pt.tmp")
            torch.save(prior_obj, temporary_prior)
            temporary_prior.replace(prior_path)
        prior_meta.append((img_path, cls_name, prior_path))
        if (idx + 1) % 50 == 0:
            print(f"[prior] {idx + 1}/{len(image_items)} ready")
    if inv_pipe is not None:
        del inv_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.prior_only:
        print(f"[done] DDIM priors: {prior_cache_dir}")
        return

    for style_idx, target_domain in enumerate(target_domains):
        print(f"[stage 2/2] Target {style_idx + 1}/{len(target_domains)}: {target_domain}")
        pipe = _build_pipeline(args, torch_dtype)
        inject_style_lora_into_unet(
            pipe, args.checkpoint_dir, style_idx, lora_scale=args.lora_scale,
            semantic_scale=args.semantic_scale, style_scale=args.style_scale, debug=args.debug,
        )
        style_dir = output_dir / _sanitize_name(target_domain)
        (style_dir / "images").mkdir(parents=True, exist_ok=True)
        begin_target(style_dir)
        with (style_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
            for prior_idx, (_, cls_name, prior_path) in enumerate(prior_meta):
                prior_obj = torch.load(prior_path, map_location="cpu", weights_only=True)
                if args.generation_prompt_mode == "prior":
                    prompt = prior_obj["prompt"]
                else:
                    prompt = args.generation_prompt_template.format(cls=prompt_class(cls_name))
                class_dir = style_dir / "images" / cls_name
                class_dir.mkdir(parents=True, exist_ok=True)
                out_path = class_dir / f"{prior_obj['image_id']}.png"
                latents = prior_obj["latents"].to(device=pipe.device, dtype=next(pipe.unet.parameters()).dtype)
                model_height, model_width, paper_height, paper_width = _infer_output_hw_from_prior(prior_obj, args.resolution)
                seed = args.seed + style_idx * 100000 + prior_idx
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                pipe.scheduler.prior_timestep = prior_obj["inversion_timestep"]
                result = pipe(
                    prompt=prompt, negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.generation_guidance_scale,
                    latents=latents, height=model_height, width=model_width, generator=generator,
                )
                image = result.images[0]
                if paper_width != model_width or paper_height != model_height:
                    image = image.crop((0, 0, paper_width, paper_height))
                image.save(out_path)
                write_record(metadata, {
                    "image": out_path.relative_to(style_dir).as_posix(),
                    "class_name": cls_name, "label": class_to_idx[cls_name],
                    "source_image": prior_obj["source_relative"], "target_domain": target_domain,
                    "style_index": style_idx, "mode": "ddim_inversion",
                    "inversion_strength": args.inversion_end_ratio,
                    "inversion_timestep": prior_obj["inversion_timestep"],
                    "adapter_strength": args.lora_scale,
                    "semantic_scale": args.semantic_scale, "style_scale": args.style_scale,
                    "seed": seed, "inversion_seed": prior_obj["seed"], "prompt": prompt,
                })
                print(f"[gen][{target_domain}] {prior_idx + 1}/{len(prior_meta)} -> {out_path}")
        complete_target(style_dir, len(prior_meta), {cls for _, cls, _ in prior_meta})
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[done] DDIM inversion and target-domain synthesis completed.")


if __name__ == "__main__":
    main()
