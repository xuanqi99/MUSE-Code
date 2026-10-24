#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for MUSE from the Diffusers SDXL DreamBooth LoRA training example:
# https://github.com/huggingface/diffusers/blob/v0.24.0/examples/dreambooth/train_dreambooth_lora_sdxl.py

import argparse
import gc
import itertools
import json
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch.utils.checkpoint
import transformers
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
try:
    from diffusers.loaders import LoraLoaderMixin
except ImportError:
    from diffusers.loaders import StableDiffusionLoraLoaderMixin as LoraLoaderMixin
# from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import random

try:
    from ldm.modules.diffusionmodules.openaimodel_enc import Encoder4
except Exception:
    Encoder4 = None

GLOBAL_TIME = 0
CURRENT_TARGET_INDEX = -1


def set_current_target_index(target_idx: int):
    global CURRENT_TARGET_INDEX
    CURRENT_TARGET_INDEX = int(target_idx)


def set_params_requires_grad(params, requires_grad: bool):
    # Deduplicate by object identity to avoid repeated toggles for shared refs.
    seen = set()
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        p.requires_grad_(requires_grad)


def unet_lora_state_dict(unet):
    """Keep the shared factors and every target core in the checkpoint."""
    state_dict = {}
    for name, module in unet.named_modules():
        layer = getattr(module, "lora_layer", None)
        if layer is not None:
            for key, value in layer.state_dict().items():
                state_dict[f"unet.{name}.lora.{key}"] = value
    if not state_dict:
        raise ValueError("The U-Net has no MUSE adapter parameters to save.")
    return state_dict


def save_muse_weights(save_directory, unet_lora_layers,
                      text_encoder_lora_layers=None, text_encoder_2_lora_layers=None):
    from safetensors.torch import save_file

    state_dict = dict(unet_lora_layers)
    for prefix, layers in (("text_encoder", text_encoder_lora_layers),
                           ("text_encoder_2", text_encoder_2_lora_layers)):
        if layers:
            state_dict.update({f"{prefix}.{key}": value for key, value in layers.items()})
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state_dict.items()},
        str(Path(save_directory) / "pytorch_lora_weights.safetensors"),
        metadata={"format": "pt", "adapter": "muse"},
    )


def load_muse_weights(input_dir):
    from safetensors.torch import load_file

    input_dir = Path(input_dir)
    safe_path = input_dir / "pytorch_lora_weights.safetensors"
    if safe_path.is_file():
        state_dict = load_file(str(safe_path), device="cpu")
    else:
        state_dict = torch.load(
            input_dir / "pytorch_lora_weights.bin", map_location="cpu", weights_only=True
        )
    normalized = {}
    for key, value in state_dict.items():
        # Earlier Diffusers saves could add a second U-Net prefix.
        while key.startswith("unet.unet."):
            key = key[len("unet."):]
        if key in normalized:
            raise ValueError(f"Duplicate adapter tensor in checkpoint: {key}")
        normalized[key] = value
    return normalized


def restore_muse_unet(unet, state_dict):
    expected = set(unet_lora_state_dict(unet))
    received = {key for key in state_dict if key.startswith("unet.")}
    if expected != received:
        raise ValueError(
            "MUSE checkpoint does not match this model. "
            f"Missing tensors: {sorted(expected - received)}; "
            f"unexpected tensors: {sorted(received - expected)}"
        )
    for name, module in unet.named_modules():
        layer = getattr(module, "lora_layer", None)
        if layer is not None:
            prefix = f"unet.{name}.lora."
            layer.load_state_dict(
                {key[len(prefix):]: value for key, value in state_dict.items()
                 if key.startswith(prefix)},
                strict=True,
            )


def save_muse_config(output_dir, args, target_domains):
    config = {
        "dataset": args.dataset,
        "source_domain": args.source_domain,
        "target_domains": list(target_domains),
        "rank": args.rank,
        "style_rank": args.style_rank,
        "seed": args.seed,
    }
    with (Path(output_dir) / "muse_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

def target_layer_from_sd_name(k):
  # They use slightly different naming schemes for attn processors vs the rest
  if '.processor.to_' in k:
    target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
    target_layer = target_layer.replace("to_out", "to_out[0]")
  else:
    target_layer = k.split(".lora.")[0]
  # Replace '.1.' with '[1]' and so on:
  for i in range(10):
    target_layer = target_layer.replace(f".{i}", f"[{i}]")
  # Return (skipping the first 'unet.' in this case):
  return target_layer

class DiagnalLayer3(nn.Module):
    def __init__(
        self,
        rank: int = 4,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.omegas_weights = nn.Parameter(torch.zeros(rank, rank, device=device, dtype=dtype))

    def forward(self, x, t):
        return x + x @ self.omegas_weights * t  # element-wise multiplication


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        style_rank: Optional[int] = None,
        num_style_domains: int = 1,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        style_rank = int(rank if style_rank is None else style_rank)
        num_style_domains = int(max(1, num_style_domains))

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        self.style_down = nn.Linear(in_features, style_rank, bias=False, device=device, dtype=dtype)
        self.style_up = nn.Linear(style_rank, out_features, bias=False, device=device, dtype=dtype)
        self.style_cores = nn.Parameter(
            torch.zeros(num_style_domains, style_rank, style_rank, device=device, dtype=dtype)
        )

        self.network_alpha = network_alpha
        self.rank = rank
        self.style_rank = style_rank
        self.num_style_domains = num_style_domains
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.style_down.weight, std=1 / max(1, style_rank))
        nn.init.normal_(self.style_up.weight, std=(1 / max(1, style_rank)) * 1e-3)
        with torch.no_grad():
            self.style_cores.normal_(mean=0.0, std=1 / max(1, style_rank))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        hidden_states = hidden_states.to(dtype)

        sem_delta = self.up(self.down(hidden_states))

        sty_delta = 0.0
        target_idx = int(CURRENT_TARGET_INDEX)
        if 0 <= target_idx < self.num_style_domains:
            style_hidden = self.style_down(hidden_states)
            core = self.style_cores[target_idx]
            style_hidden = style_hidden @ core.T
            sty_delta = self.style_up(style_hidden)

        up_hidden_states = sem_delta + sty_delta

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

def ensure_lora_compatible_linear(module: nn.Module) -> nn.Module:
    if hasattr(module, "set_lora_layer"):
        return module
    module.lora_layer = None
    module._forward_lora_orig = module.forward

    def _set_lora_layer(self, lora_layer):
        self.lora_layer = lora_layer

    def _get_lora_layer(self):
        return getattr(self, "lora_layer", None)

    def _lora_forward(self, hidden_states, *args, **kwargs):
        output = self._forward_lora_orig(hidden_states, *args, **kwargs)
        lora_layer = getattr(self, "lora_layer", None)
        if lora_layer is not None:
            output = output + lora_layer(hidden_states)
        return output

    module.set_lora_layer = _set_lora_layer.__get__(module, module.__class__)
    module.get_lora_layer = _get_lora_layer.__get__(module, module.__class__)
    module.forward = _lora_forward.__get__(module, module.__class__)
    return module
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)


# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    instance_prompt=str,
    validation_prompt=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = "widget:\n" if images else ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"""
        - text: '{validation_prompt if validation_prompt else ' ' }'
          output:
            url:
                "image_{i}.png"
        """

    yaml = f"""
---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
    """

    model_card = f"""
# MUSE SDXL LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained with the MUSE fine-tuning pipeline.

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MUSE SDXL fine-tuning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from a model registry or local path.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from a model registry or local path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the dataset hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--theta_weight",
        type=float,
        default=1.0,
        help="The weight of ",
    )
    parser.add_argument(
        "--randomimage",
        type=float,
        default=0.75,
        help="The ",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=False,  #  text encoder
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_theta",
        type=float,
        default=1,
        help="lr_theta.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to a remote model registry.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use when pushing to a remote model registry.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "TensorBoard log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help=("The semantic LoRA rank (source-shared branch)."),
    )
    parser.add_argument(
        "--source_domain",
        type=str,
        default="Clipart",
        help="Subdirectory name of the source domain under instance_data_dir.",
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        default="Real World",
        help="Subdirectory name of the target domain under instance_data_dir.",
    )
    parser.add_argument(
        "--target_domains",
        type=str,
        default=None,
        help="Comma-separated target domain names for multi-target UDA. If not set, OfficeHome uses all domains except source_domain.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="officehome",
        choices=["officehome", "office31", "minidomainnet"],
        help="Which dataset to use for domain images.",
    )

    parser.add_argument(
        "--officehome_root",
        type=str,
        default=None,
        help="Root directory for OfficeHome dataset (used as fallback for --instance_data_dir).",
    )
    parser.add_argument(
        "--minidomainnet_split_dir",
        type=str,
        default=None,
        help=(
            "Directory that contains miniDomainNet train/test split txt files "
            "(e.g. domainnet/splits_mini). When --dataset=minidomainnet, *_train.txt "
            "files are used to avoid mixing evaluation images into fine-tuning."
        ),
    )

    # NEW: concept-token related args
    parser.add_argument(
        "--use_concept_tokens",
        action="store_true",
        default=False,
        help="Use Encoder4 to learn per-image concept tokens and inject into cross-attention.",
    )
    parser.add_argument(
        "--concept_token_count",
        type=int,
        default=5,
        help="Number of learned concept tokens per image (latent_unit in Encoder4).",
    )
    parser.add_argument(
        "--encoder_channels",
        type=int,
        default=128,
        help="Base channels (d) of Encoder4.",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-4,
        help="Learning rate for Encoder4 parameters.",
    )
    parser.add_argument(
        "--encoder_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for Encoder4 parameters.",
    )
    parser.add_argument(
        "--style_rank",
        type=int,
        default=64,
        help="Style adapter rank for target-style branch. Default is 64.",
    )
    parser.add_argument(
        "--freeze_semantic_on_target",
        dest="freeze_semantic_on_target",
        action="store_true",
        help=(
            "Freeze semantic LoRA down/up during target-branch backward, "
            "so target loss only updates style_down/style_up/style_cores."
        ),
    )
    parser.add_argument(
        "--no_freeze_semantic_on_target",
        dest="freeze_semantic_on_target",
        action="store_false",
        help="Disable semantic freezing on target branch (for ablation).",
    )
    parser.add_argument(
        "--style_loss_weights",
        type=str,
        default=None,
        help="Comma-separated lambda_d for target domains. Length should match number of target domains.",
    )
    parser.add_argument(
        "--orth_lambda",
        type=float,
        default=1e-4,
        help="Weight for cross-target style orthogonality regularization.",
    )
    parser.add_argument(
        "--style_core_l2_lambda",
        type=float,
        default=1e-5,
        help="L2 regularization weight for each target style core to stabilize multi-target disentanglement.",
    )
    parser.add_argument(
        "--enable_adaptive_target_sampling",
        dest="enable_adaptive_target_sampling",
        action="store_true",
        help="Enable difficulty-adaptive target-domain sampling using EMA target losses.",
    )
    parser.add_argument(
        "--disable_adaptive_target_sampling",
        dest="enable_adaptive_target_sampling",
        action="store_false",
        help="Disable difficulty-adaptive target-domain sampling and use static sampling.",
    )
    parser.add_argument(
        "--target_sampling_alpha",
        type=float,
        default=1.0,
        help="Exponent alpha in adaptive sampling probability p_d  (EMA_loss_d + eps)^alpha.",
    )
    parser.add_argument(
        "--target_sampling_ema",
        type=float,
        default=0.9,
        help="EMA momentum for per-domain target losses used by adaptive sampling.",
    )
    parser.add_argument(
        "--target_sampling_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for adaptive sampling.",
    )
    parser.add_argument(
        "--target_sampling_update_steps",
        type=int,
        default=20,
        help="Update interval (in optimizer steps) for adaptive target sampling probabilities.",
    )
    parser.add_argument(
        "--token_strength_src",
        type=float,
        default=1.0,
        help="Scaling factor applied to source-domain concept tokens before concatenation.",
    )
    parser.add_argument(
        "--token_strength_tgt",
        type=float,
        default=1.0,
        help="Scaling factor applied to target-domain concept tokens before concatenation.",
    )
    # /
    parser.add_argument(
        "--gen_concept_image",
        type=str,
        default=None,
        help="Path to a concept image used at generation time to extract concept tokens.",
    )
    parser.add_argument(
        "--gen_num_tokens",
        type=int,
        default=None,
        help="How many concept tokens to use at generation time. If None, use all.",
    )
    parser.add_argument(
        "--gen_token_indices",
        type=str,
        default=None,
        help="Comma-separated indices of concept tokens to use at generation time, e.g. '0,2,4'. Overrides gen_num_tokens if set.",
    )
    parser.add_argument(
        "--gen_token_strength",
        type=float,
        default=1.0,
        help="Scaling factor for generation-time concept tokens.",
    )
    parser.add_argument(
        "--inference_time_t",
        type=float,
        default=1.0,
        help="Time conditioning t in [0,1] to apply to time-aware LoRA at inference.",
    )
    parser.add_argument(
        "--use_concepts_in_validation",
        action="store_true",
        help="If set and gen_concept_image is provided, use concept tokens during validation image generation.",
    )

    parser.set_defaults(
        freeze_semantic_on_target=True,
        enable_adaptive_target_sampling=True,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # NEW: relax / auto-fill dataset_name / instance_data_dir constraint for folder-style datasets
    if args.dataset in {"officehome", "office31", "minidomainnet"}:
        if args.dataset_name is not None and args.instance_data_dir is not None:
            raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

        if args.dataset_name is None and args.instance_data_dir is None:
            # Prefer explicit environment/CLI roots, then portable package-relative locations.
            if args.dataset == "officehome":
                candidate_roots = [
                    os.environ.get("INSTANCE_DIR"),
                    os.environ.get("OFFICEHOME_ROOT"),
                    args.officehome_root,
                    str(PACKAGE_ROOT / "data" / "OfficeHomeDataset_10072016"),
                    str(PACKAGE_ROOT / "data" / "OfficeHome"),
                ]
            elif args.dataset == "office31":
                candidate_roots = [
                    os.environ.get("INSTANCE_DIR"),
                    os.environ.get("OFFICE31_ROOT"),
                    str(PACKAGE_ROOT / "data" / "Office-31"),
                ]
            else:
                candidate_roots = [
                    os.environ.get("INSTANCE_DIR"),
                    os.environ.get("MINIDOMAINNET_ROOT"),
                    str(PACKAGE_ROOT / "data" / "minidomainnet" / "domainnet"),
                ]
            candidate_roots = [c for c in candidate_roots if c]  # drop None/empty

            picked = None
            for c in candidate_roots:
                try:
                    p = Path(c)
                except Exception:
                    continue
                if p.exists() and p.is_dir():
                    picked = str(p)
                    break

            if picked is not None:
                args.instance_data_dir = picked
                warnings.warn(
                    f"`--instance_data_dir` not provided; auto-using {args.dataset} root: {args.instance_data_dir}"
                )
            else:
                raise ValueError(
                    "Specify either `--dataset_name` or `--instance_data_dir`.\n"
                    f"Tried these {args.dataset} roots (none existed):\n"
                    + "\n".join([f"  - {c}" for c in candidate_roots])
                )

        if args.dataset == "minidomainnet" and args.minidomainnet_split_dir is None:
            candidate_split_dirs = [
                os.environ.get("MINIDOMAINNET_SPLIT_DIR"),
                str(Path(args.instance_data_dir) / "splits_mini") if args.instance_data_dir else None,
                str(PACKAGE_ROOT / "data" / "minidomainnet" / "domainnet" / "splits_mini"),
            ]
            candidate_split_dirs = [c for c in candidate_split_dirs if c]

            picked_split = None
            for c in candidate_split_dirs:
                try:
                    p = Path(c)
                except Exception:
                    continue
                if p.exists() and p.is_dir():
                    picked_split = str(p)
                    break

            if picked_split is not None:
                args.minidomainnet_split_dir = picked_split
                warnings.warn(
                    "`--minidomainnet_split_dir` not provided; "
                    f"auto-using split dir: {args.minidomainnet_split_dir}"
                )
            else:
                raise ValueError(
                    "miniDomainNet fine-tuning expects the standard splits_mini directory so that "
                    "only *_train.txt entries are used during training.\n"
                    "Tried these split roots (none existed):\n"
                    + "\n".join([f"  - {c}" for c in candidate_split_dirs])
                )
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args

#### update domain!!!! [begin]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def doublePath(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain dataset directory not found: {path}")
    arr = [item for item in path.rglob("*") if _is_image_file(item)]
    arr.sort()
    if len(arr) == 0:
        raise RuntimeError(f"No images found under domain dataset directory: {path}")
    return arr


def _domain_split_stems(domain_name: str):
    stems = []
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


def resolve_domain_train_split(split_dir, domain_name):
    split_dir = Path(split_dir)
    for stem in _domain_split_stems(domain_name):
        candidate = split_dir / f"{stem}_train.txt"
        if candidate.exists():
            return candidate
    tried = [str(split_dir / f"{stem}_train.txt") for stem in _domain_split_stems(domain_name)]
    raise FileNotFoundError(
        f"Could not find train split txt for domain '{domain_name}'. Tried:\n"
        + "\n".join([f"  - {p}" for p in tried])
    )


def paths_from_train_split(dataset_root, split_dir, domain_name):
    dataset_root = Path(dataset_root)
    split_file = resolve_domain_train_split(split_dir, domain_name)
    arr = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path = line.split()[0]
            img_path = dataset_root / rel_path
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image listed in split file does not exist:\n"
                    f"  split_file = {split_file}\n"
                    f"  rel_path   = {rel_path}\n"
                    f"  resolved   = {img_path}"
                )
            if _is_image_file(img_path):
                arr.append(img_path)
    if len(arr) == 0:
        raise RuntimeError(f"No valid train images found from split file: {split_file}")
    return arr

class MUSEDomainDataset():
    def __init__(
        self,
        instance_data_root=None,
        instance_prompt=None,
        class_prompts=None,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        randomimage=0.75,
        source_domain="Clipart",
        target_domain="Real World",
        target_domains=None,
        train_split_dir=None,
        # Optional direct image paths and captions.
        image_paths1=None,
        image_paths2=None,
        captions1=None,
        captions2=None,
    ):
        """
        If image_paths1/2 & captions1/2 are provided, use them directly.
        Otherwise, fall back to OfficeHome-style directory structure using instance_data_root.
        """
        self.size = size
        self.randomimage = randomimage
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt

        Image.MAX_IMAGE_PIXELS = None

        # Direct image and caption lists bypass the domain-folder scan.
        self.use_direct_lists = (
            image_paths1 is not None
            and image_paths2 is not None
            and captions1 is not None
            and captions2 is not None
        )

        if self.use_direct_lists:
            # Expect lists of path strings and caption strings
            self.instance_image_paths1 = [Path(p) for p in image_paths1]
            self.instance_image_paths2 = [Path(p) for p in image_paths2]
            self.custom_instance_prompts1 = list(captions1)
            self.custom_instance_prompts2 = list(captions2)
            self.num_instance_images1 = len(self.instance_image_paths1)
            self.num_instance_images2 = len(self.instance_image_paths2)
            self.target_domain_indices = [0 for _ in range(self.num_instance_images2)]
            self.num_target_domains = 1
            self._length = max(self.num_instance_images1, self.num_instance_images2)
            self.available_target_domain_indices = [0] if self.num_instance_images2 > 0 else []
            self.target_sampling_probs = torch.ones(self.num_target_domains, dtype=torch.float32)
        else:
            # Original OfficeHome domain-folder logic
            self.instance_root = Path(instance_data_root)
            self.train_split_dir = Path(train_split_dir) if train_split_dir is not None else None
            self.instance_data_root1 = self.instance_root / source_domain
            if target_domains is None:
                target_domains = [target_domain]
            target_domains = [td for td in target_domains if td and td != source_domain]
            if not target_domains:
                target_domains = [target_domain]
            self.target_domains = target_domains

            if self.train_split_dir is not None:
                self.instance_image_paths1 = paths_from_train_split(
                    self.instance_root, self.train_split_dir, source_domain
                )
            else:
                self.instance_image_paths1 = doublePath(self.instance_data_root1)
            self.instance_image_paths2 = []
            self.target_domain_indices = []
            self.target_domain_paths = []
            for domain_idx, domain_name in enumerate(self.target_domains):
                if self.train_split_dir is not None:
                    domain_paths = paths_from_train_split(
                        self.instance_root, self.train_split_dir, domain_name
                    )
                else:
                    domain_root = self.instance_root / domain_name
                    domain_paths = doublePath(domain_root)
                self.target_domain_paths.append(domain_paths)
                self.instance_image_paths2.extend(domain_paths)
                self.target_domain_indices.extend([domain_idx] * len(domain_paths))

            self.available_target_domain_indices = [
                domain_idx for domain_idx, domain_paths in enumerate(self.target_domain_paths) if len(domain_paths) > 0
            ]

            if len(self.instance_image_paths2) == 0:
                raise ValueError("No target images found for configured target domains.")

            self.custom_instance_prompts1 = [
                f"A {path.parent.name.replace('_', ' ').lower()}"
                for path in self.instance_image_paths1
            ]
            self.custom_instance_prompts2 = "An image"
            self.num_target_domains = len(self.target_domains)

            self.num_instance_images1 = len(self.instance_image_paths1)
            self.num_instance_images2 = len(self.instance_image_paths2)

            self._length = max(self.num_instance_images1, self.num_instance_images2)

            self.target_sampling_probs = torch.zeros(self.num_target_domains, dtype=torch.float32)
            for domain_idx in self.available_target_domain_indices:
                self.target_sampling_probs[domain_idx] = 1.0

        prob_sum = float(self.target_sampling_probs.sum().item())
        if prob_sum <= 0:
            self.target_sampling_probs.fill_(1.0 / max(1, self.num_target_domains))
        else:
            self.target_sampling_probs /= prob_sum
        self.target_sampling_probs.share_memory_()

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                #
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def set_target_sampling_probs(self, probs):
        if probs is None:
            return
        probs_t = torch.tensor(probs, dtype=torch.float32)
        if probs_t.numel() != self.num_target_domains:
            raise ValueError(
                f"Expected {self.num_target_domains} sampling probs, got {probs_t.numel()}"
            )
        probs_t = torch.clamp(probs_t, min=0.0)
        if self.available_target_domain_indices:
            mask = torch.zeros_like(probs_t)
            mask[self.available_target_domain_indices] = 1.0
            probs_t = probs_t * mask
        s = float(probs_t.sum().item())
        if s <= 0:
            probs_t.zero_()
            if self.available_target_domain_indices:
                uniform = 1.0 / float(len(self.available_target_domain_indices))
                for idx in self.available_target_domain_indices:
                    probs_t[idx] = uniform
            else:
                probs_t.fill_(1.0 / max(1, self.num_target_domains))
        else:
            probs_t /= s
        self.target_sampling_probs.copy_(probs_t)

    def get_target_sampling_probs(self):
        return self.target_sampling_probs.clone()

    def __getitem__(self, index):
        example = {}
        instance_image_path1 = self.instance_image_paths1[index % self.num_instance_images1]
        instance_image_path2 = self.instance_image_paths2[index % self.num_instance_images2]
        instance_image1 = Image.open(instance_image_path1)
        instance_image2 = Image.open(instance_image_path2)

        instance_image1 = exif_transpose(instance_image1)
        instance_image2 = exif_transpose(instance_image2)

        if not instance_image1.mode == "RGB":
            instance_image1 = instance_image1.convert("RGB")
        if not instance_image2.mode == "RGB":
            instance_image2 = instance_image2.convert("RGB")

        example["instance_images1"] = self.image_transforms(instance_image1)
        example["instance_images2"] = self.image_transforms(instance_image2)

        # When using direct lists, custom_instance_prompts1 is already
        # per-image caption (e.g., "a aeroplane"), and custom_instance_prompts2 too.
        if self.use_direct_lists:
            example["instance_prompt1"] = self.custom_instance_prompts1[index % self.num_instance_images1]
            example["instance_prompt2"] = self.custom_instance_prompts2[index % self.num_instance_images2]
            example["target_domain_idx"] = 0
        else:
            if random.random() > self.randomimage:
                example["instance_prompt1"] = self.custom_instance_prompts1[index % self.num_instance_images1]
            else:
                example["instance_prompt1"] = self.custom_instance_prompts2
            example["instance_prompt2"] = self.custom_instance_prompts2

            # Multi-target balanced sampling:
            # 1) sample a target domain with externally-updated probabilities
            # 2) sample one image within that domain
            sampled_domain_idx = int(torch.multinomial(self.target_sampling_probs, 1).item())
            if sampled_domain_idx not in self.available_target_domain_indices:
                sampled_domain_idx = random.choice(self.available_target_domain_indices)
            sampled_domain_paths = self.target_domain_paths[sampled_domain_idx]
            sampled_local_idx = random.randrange(len(sampled_domain_paths))

            sampled_path = sampled_domain_paths[sampled_local_idx]
            sampled_image = Image.open(sampled_path)
            sampled_image = exif_transpose(sampled_image)
            if not sampled_image.mode == "RGB":
                sampled_image = sampled_image.convert("RGB")
            example["instance_images2"] = self.image_transforms(sampled_image)
            example["target_domain_idx"] = sampled_domain_idx

        return example

#### update domain!!!! [/end]


def collate_fn(examples, with_prior_preservation=False):
    pixel_values1 = [example["instance_images1"] for example in examples]
    prompts1 = [example["instance_prompt1"] for example in examples]

    pixel_values2 = [example["instance_images2"] for example in examples]
    prompts2 = [example["instance_prompt2"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    # if with_prior_preservation:
    #     pixel_values += [example["class_images"] for example in examples]
    #     prompts += [example["class_prompt"] for example in examples]

    pixel_values1 = torch.stack(pixel_values1)
    pixel_values1 = pixel_values1.to(memory_format=torch.contiguous_format).float()
    pixel_values2 = torch.stack(pixel_values2)
    pixel_values2 = pixel_values2.to(memory_format=torch.contiguous_format).float()
    target_domain_idx = torch.tensor([example["target_domain_idx"] for example in examples], dtype=torch.long)

    # batch = {"pixel_values": pixel_values, "prompts": prompts}
    batch = {
        "pixel_values1": pixel_values1,
        "prompts1": prompts1,
        "pixel_values2": pixel_values2,
        "prompts2": prompts2,
        "target_domain_idx": target_domain_idx,
    }
    return batch


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

#### update domain!!!! [/begin]

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.officehome_prompts = ['drill', 'exit sign', 'bottle', 'glasses', 'computer', 'file cabinet', 'shelf', 'toys', 'sink', 'laptop', 'kettle', 'folder', 'keyboard', 'flipflops', 'pencil', 'bed', 'hammer', 'toothbrush', 'couch', 'bike', 'postit notes', 'mug', 'webcam', 'desk lamp', 'telephone', 'helmet', 'mouse', 'pen', 'monitor', 'mop', 'sneakers', 'notebook', 'backpack', 'alarm clock', 'push pin', 'paper clip', 'batteries', 'radio', 'fan', 'ruler', 'pan', 'screwdriver', 'trash can', 'printer', 'speaker', 'eraser', 'bucket', 'chair', 'calendar', 'calculator', 'flowers', 'lamp shade', 'spoon', 'candles', 'clipboards', 'scissors', 'tv', 'curtains', 'fork', 'soda', 'table', 'knives', 'oven', 'refrigerator', 'marker']
        self.prompt = "a %s"
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt % self.officehome_prompts[index % len(self.officehome_prompts)]
        example["index"] = index
        return example

#### update domain!!!! [/end]

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# ====  ====

def _preprocess_concept_image(img_path: str, resolution: int, center_crop: bool):
    img = Image.open(img_path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tfm(img).unsqueeze(0)  # [1,3,H,W]

def _parse_token_indices(spec: Optional[str], max_count: int):
    if not spec:
        return list(range(max_count))
    idxs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 0 or i >= max_count:
            continue
        idxs.append(i)
    if not idxs:
        idxs = list(range(max_count))
    return idxs

def _apply_time_conditioned_lora_to_pipeline(pipeline, lora_state_dict: dict, time_t: float):
    #  lora ( down/up/diagnal) dW  pipeline.unet
    for k, v in lora_state_dict.items():
        if not k.endswith("down.weight"):
            continue
        a1 = v
        up_key = k.replace("down.weight", "up.weight")
        diag_key = k.replace("down.weight", "diagnal.omegas_weights")
        if up_key not in lora_state_dict or diag_key not in lora_state_dict:
            continue
        b1 = lora_state_dict[up_key]
        omegas_weights = lora_state_dict[diag_key]
        eye = torch.eye(omegas_weights.shape[0], device=a1.device, dtype=a1.dtype)
        matrix = omegas_weights * time_t + eye
        dW = b1 @ (matrix.T) @ a1
        target_layer = target_layer_from_sd_name(k)
        #  dW
        eval(f'pipeline.unet.{target_layer}').weight.data += dW.to(
            eval(f'pipeline.unet.{target_layer}').weight.data.dtype
        ).to(pipeline.device)

def _load_concept_encoder_for_inference(pipeline, args, device):
    if Encoder4 is None:
        raise ImportError(
            "Encoder4 is unavailable. Install/enable `ldm.modules.diffusionmodules.openaimodel_enc` "
            "before using concept-token inference."
        )
    #  pipeline.encode_prompt
    pe, _ = pipeline.encode_prompt(
        prompt=["x"],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    embed_dim = pe.shape[-1]
    enc = Encoder4(
        d=args.encoder_channels,
        context_dim=embed_dim,
        latent_unit=args.concept_token_count,
        num_channels=3,
    ).to(device, dtype=torch.float32)
    enc_path = os.path.join(args.output_dir, "concept_encoder.pt")
    if os.path.exists(enc_path):
        sd = torch.load(enc_path, map_location="cpu")
        enc.load_state_dict(sd, strict=False)
    enc.eval()
    return enc, embed_dim

# ====  ====

def main(args):
    officehome_all_domains = ["Art", "Clipart", "Product", "Real World"]
    if args.target_domains is not None:
        target_domain_names = [d.strip() for d in args.target_domains.split(",") if d.strip()]
    elif args.dataset.lower() == "officehome":
        target_domain_names = [d for d in officehome_all_domains if d != args.source_domain]
    else:
        target_domain_names = [args.target_domain]
    target_domain_names = [d for d in target_domain_names if d != args.source_domain]
    if len(target_domain_names) == 0:
        target_domain_names = [args.target_domain]
    sorted_target_domain_names = sorted(target_domain_names, key=lambda x: x.lower())

    if args.style_loss_weights:
        style_loss_weights = [float(x.strip()) for x in args.style_loss_weights.split(",") if x.strip()]
    else:
        style_loss_weights = [1.0 for _ in target_domain_names]
    if len(style_loss_weights) != len(target_domain_names):
        raise ValueError("Length of --style_loss_weights must match number of target domains.")
    style_weight_logs = {
        f"style_w_{name.replace(' ', '_').lower()}": float(style_loss_weights[idx])
        for idx, name in enumerate(target_domain_names)
    }
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See  for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_parameters1 = []
    unet_lora_parameters2 = []
    unet_semantic_parameters = []
    unet_style_parameters = []

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        # if "attn2" not in attn_processor_name:
        #     continue
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        ensure_lora_compatible_linear(attn_module.to_q).set_lora_layer(
             LoRALinearLayer(
                 in_features=attn_module.to_q.in_features,
                 out_features=attn_module.to_q.out_features,
                 rank=args.rank,
                 style_rank=args.style_rank,
                 num_style_domains=len(target_domain_names),
             )
         )
        ensure_lora_compatible_linear(attn_module.to_k).set_lora_layer(
             LoRALinearLayer(
                 in_features=attn_module.to_k.in_features,
                 out_features=attn_module.to_k.out_features,
                 rank=args.rank,
                 style_rank=args.style_rank,
                 num_style_domains=len(target_domain_names),
             )
         )
        ensure_lora_compatible_linear(attn_module.to_v).set_lora_layer(
             LoRALinearLayer(
                 in_features=attn_module.to_v.in_features,
                 out_features=attn_module.to_v.out_features,
                 rank=args.rank,
                 style_rank=args.style_rank,
                 num_style_domains=len(target_domain_names),
             )
         )
        ensure_lora_compatible_linear(attn_module.to_out[0]).set_lora_layer(
             LoRALinearLayer(
                 in_features=attn_module.to_out[0].in_features,
                 out_features=attn_module.to_out[0].out_features,
                 rank=args.rank,
                 style_rank=args.style_rank,
                 num_style_domains=len(target_domain_names),
             )
         )

        # Accumulate the LoRA params to optimize.

        unet_lora_parameters1.extend(attn_module.to_q.lora_layer.down.parameters())
        unet_lora_parameters1.extend(attn_module.to_k.lora_layer.down.parameters())
        unet_lora_parameters1.extend(attn_module.to_v.lora_layer.down.parameters())
        unet_lora_parameters1.extend(attn_module.to_q.lora_layer.up.parameters())
        unet_lora_parameters1.extend(attn_module.to_k.lora_layer.up.parameters())
        unet_lora_parameters1.extend(attn_module.to_v.lora_layer.up.parameters())
        unet_lora_parameters1.extend(attn_module.to_q.lora_layer.style_down.parameters())
        unet_lora_parameters1.extend(attn_module.to_k.lora_layer.style_down.parameters())
        unet_lora_parameters1.extend(attn_module.to_v.lora_layer.style_down.parameters())
        unet_lora_parameters1.extend(attn_module.to_q.lora_layer.style_up.parameters())
        unet_lora_parameters1.extend(attn_module.to_k.lora_layer.style_up.parameters())
        unet_lora_parameters1.extend(attn_module.to_v.lora_layer.style_up.parameters())
        unet_lora_parameters2.append(attn_module.to_q.lora_layer.style_cores)
        unet_lora_parameters2.append(attn_module.to_k.lora_layer.style_cores)
        unet_lora_parameters2.append(attn_module.to_v.lora_layer.style_cores)
        unet_lora_parameters1.extend(attn_module.to_out[0].lora_layer.down.parameters())
        unet_lora_parameters1.extend(attn_module.to_out[0].lora_layer.up.parameters())
        unet_lora_parameters1.extend(attn_module.to_out[0].lora_layer.style_down.parameters())
        unet_lora_parameters1.extend(attn_module.to_out[0].lora_layer.style_up.parameters())
        unet_lora_parameters2.append(attn_module.to_out[0].lora_layer.style_cores)

        # Keep explicit parameter groups for branch-wise gradient control in ablations.
        unet_semantic_parameters.extend(attn_module.to_q.lora_layer.down.parameters())
        unet_semantic_parameters.extend(attn_module.to_k.lora_layer.down.parameters())
        unet_semantic_parameters.extend(attn_module.to_v.lora_layer.down.parameters())
        unet_semantic_parameters.extend(attn_module.to_q.lora_layer.up.parameters())
        unet_semantic_parameters.extend(attn_module.to_k.lora_layer.up.parameters())
        unet_semantic_parameters.extend(attn_module.to_v.lora_layer.up.parameters())
        unet_semantic_parameters.extend(attn_module.to_out[0].lora_layer.down.parameters())
        unet_semantic_parameters.extend(attn_module.to_out[0].lora_layer.up.parameters())

        unet_style_parameters.extend(attn_module.to_q.lora_layer.style_down.parameters())
        unet_style_parameters.extend(attn_module.to_k.lora_layer.style_down.parameters())
        unet_style_parameters.extend(attn_module.to_v.lora_layer.style_down.parameters())
        unet_style_parameters.extend(attn_module.to_q.lora_layer.style_up.parameters())
        unet_style_parameters.extend(attn_module.to_k.lora_layer.style_up.parameters())
        unet_style_parameters.extend(attn_module.to_v.lora_layer.style_up.parameters())
        unet_style_parameters.extend(attn_module.to_out[0].lora_layer.style_down.parameters())
        unet_style_parameters.extend(attn_module.to_out[0].lora_layer.style_up.parameters())
        unet_style_parameters.append(attn_module.to_q.lora_layer.style_cores)
        unet_style_parameters.append(attn_module.to_k.lora_layer.style_cores)
        unet_style_parameters.append(attn_module.to_v.lora_layer.style_cores)
        unet_style_parameters.append(attn_module.to_out[0].lora_layer.style_cores)
    # The text encoder comes from  transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=args.rank
        )
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            text_encoder_two, dtype=torch.float32, rank=args.rank
        )
        # NEW: define optimizer groups for text encoders
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "lr": args.text_encoder_lr,
            "weight_decay": args.adam_weight_decay_text_encoder,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "lr": args.text_encoder_lr,
            "weight_decay": args.adam_weight_decay_text_encoder,
        }

    # NEW: Instantiate Encoder4 for per-image concept tokens (lazy-dimensioned to match prompt embeddings)
    concept_encoder = None
    prompt_embed_dim = None
    if args.use_concept_tokens:
        if Encoder4 is None:
            raise ImportError(
                "`--use_concept_tokens` was enabled, but Encoder4 could not be imported from "
                "`ldm.modules.diffusionmodules.openaimodel_enc`."
            )
        # Compute the joint prompt embedding dim (concat of 2 text encoders)
        with torch.no_grad():
            dummy_prompt = [args.instance_prompt]
            _pe, _ = encode_prompt(
                [text_encoder_one.to(accelerator.device, dtype=weight_dtype),
                 text_encoder_two.to(accelerator.device, dtype=weight_dtype)],
                [tokenizer_one, tokenizer_two],
                dummy_prompt,
            )
        prompt_embed_dim = _pe.shape[-1]
        # Build Encoder4 to output concept_token_count tokens, each of size prompt_embed_dim
        concept_encoder = Encoder4(
            d=args.encoder_channels,
            context_dim=prompt_embed_dim,
            latent_unit=args.concept_token_count,
            num_channels=3,
        ).to(accelerator.device, dtype=torch.float32)
        concept_encoder.train()

    # custom save/load hooks for LoRA and Encoder4
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None
            # ...existing code up to model loop...
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_lora_state_dict(model)
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                # NEW: persist Encoder4 weights if present
                elif concept_encoder is not None and isinstance(model, type(accelerator.unwrap_model(concept_encoder))):
                    enc_path = os.path.join(output_dir, "concept_encoder.pt")
                    torch.save(model.state_dict(), enc_path)
                else:
                    pass
                weights.pop()
            # NEW:  concept_encoder isinstance
            if concept_encoder is not None:
                try:
                    ce = accelerator.unwrap_model(concept_encoder)
                except Exception:
                    ce = concept_encoder
                torch.save(ce.state_dict(), os.path.join(output_dir, "concept_encoder.pt"))
            # ...existing code to save_lora_weights...
            save_muse_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )
            save_muse_config(output_dir, args, target_domain_names)

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None
        concept_encoder_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            elif concept_encoder is not None and isinstance(model, type(accelerator.unwrap_model(concept_encoder))):
                concept_encoder_ = model
            else:
                pass

        config_path = Path(input_dir) / "muse_config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
            if saved_config.get("target_domains") != target_domain_names:
                raise ValueError("Checkpoint target-domain order differs from --target_domains.")
        lora_state_dict = load_muse_weights(input_dir)
        restore_muse_unet(unet_, lora_state_dict)
        # ...existing code for text encoders...
        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        if text_encoder_state_dict:
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=None,
                text_encoder=text_encoder_one_,
            )
        text_encoder_2_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
        }
        if text_encoder_2_state_dict:
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=None,
                text_encoder=text_encoder_two_,
                prefix="text_encoder_2",
            )
        # NEW:  concept_encoder models
        enc_path = os.path.join(input_dir, "concept_encoder.pt")
        if concept_encoder_ is not None and os.path.exists(enc_path):
            try:
                ce = accelerator.unwrap_model(concept_encoder)
            except Exception:
                ce = concept_encoder
            sd = torch.load(enc_path, map_location="cpu")
            ce.load_state_dict(sd, strict=False)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ...existing code (TF32, scale_lr)...

    # Optimization parameter groups
    unet_lora_parameters_with_lr1 = {
        "params": unet_lora_parameters1,
        "lr": args.learning_rate,
    }
    unet_lora_parameters_with_lr2 = {
        "params": unet_lora_parameters2,
        "lr": args.learning_rate * args.lr_theta,
    }
    # ...existing code building params_to_optimize for text encoders...

    # NEW: add Encoder4 params to optimizer if enabled
    if args.use_concept_tokens and concept_encoder is not None:
        encoder_parameters_with_lr = {
            "params": concept_encoder.parameters(),
            "weight_decay": args.encoder_weight_decay,
            "lr": args.encoder_lr,
        }
        if args.train_text_encoder:
            params_to_optimize = [
                unet_lora_parameters_with_lr1,
                unet_lora_parameters_with_lr2,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,
                encoder_parameters_with_lr,
            ]
        else:
            params_to_optimize = [
                unet_lora_parameters_with_lr1,
                unet_lora_parameters_with_lr2,
                encoder_parameters_with_lr,
            ]
    else:
        # Ensure params_to_optimize is defined even without concept tokens
        if args.train_text_encoder:
            params_to_optimize = [
                unet_lora_parameters_with_lr1,
                unet_lora_parameters_with_lr2,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,
            ]
        else:
            params_to_optimize = [
                unet_lora_parameters_with_lr1,
                unet_lora_parameters_with_lr2,
            ]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:

    # Folder-style multi-target datasets: Office-Home, Office-31, miniDomainNet.
    train_dataset = MUSEDomainDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompts=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        randomimage=args.randomimage,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        target_domains=target_domain_names,
        train_split_dir=(
            args.minidomainnet_split_dir if args.dataset.lower() == "minidomainnet" else None
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (
            args.crops_coords_top_left_h,
            args.crops_coords_top_left_w,
        )
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # Handle instance prompt.
    instance_time_ids = compute_time_ids()

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts1:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
        ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        if not args.train_text_encoder:
            (
                class_prompt_hidden_states,
                class_pooled_prompt_embeds,
            ) = compute_text_embeddings(args.class_prompt, text_encoders, tokenizers)

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts1 and not train_dataset.custom_instance_prompts2:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    add_time_ids = instance_time_ids
    if args.with_prior_preservation:
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)

    if not train_dataset.custom_instance_prompts1 and not train_dataset.custom_instance_prompts2:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            if args.with_prior_preservation:
                prompt_embeds = torch.cat(
                    [prompt_embeds, class_prompt_hidden_states], dim=0
                )
                unet_add_text_embeds = torch.cat(
                    [unet_add_text_embeds, class_pooled_prompt_embeds], dim=0
                )
        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt)
            tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt)
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt)
                class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt)
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        if args.use_concept_tokens and concept_encoder is not None:
            (  #  concept_encoder
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
                concept_encoder,
            ) = accelerator.prepare(
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
                concept_encoder,
            )
        else:
            (
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
    else:
        if args.use_concept_tokens and concept_encoder is not None:
            (  #  concept_encoder

                unet,
                optimizer,
                train_dataloader,
                lr_scheduler,
                concept_encoder,
            ) = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler, concept_encoder)
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-lora-sd-xl", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Target domains = {target_domain_names}")
    logger.info(f"  freeze_semantic_on_target (default) = {args.freeze_semantic_on_target}")
    logger.info(
        f"  adaptive_target_sampling = {args.enable_adaptive_target_sampling}, "
        f"alpha={args.target_sampling_alpha}, ema={args.target_sampling_ema}"
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    running_domain_counts = [0 for _ in target_domain_names]
    running_target_samples = 0
    ema_domain_losses = [1.0 for _ in target_domain_names]
    current_sampling_probs = [
        float(x)
        for x in train_dataset.get_target_sampling_probs().tolist()
    ]

    global GLOBAL_TIME
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            text_encoder_one.text_model.embeddings.requires_grad_(True)
            text_encoder_two.text_model.embeddings.requires_grad_(True)

        if args.use_concept_tokens and concept_encoder is not None:
            concept_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.freeze_semantic_on_target:
                    # Ensure source branch always updates semantic adapters.
                    set_params_requires_grad(unet_semantic_parameters, True)

                pixel_values1 = batch["pixel_values1"].to(dtype=vae.dtype)
                prompts1 = batch["prompts1"]

                pixel_values2 = batch["pixel_values2"].to(dtype=vae.dtype)
                prompts2 = batch["prompts2"]
                target_domain_idx = batch["target_domain_idx"].to(accelerator.device)

                # Pass 1: source domain (t=0)
                GLOBAL_TIME = 0
                set_current_target_index(-1)
                prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                    prompts1, text_encoders, tokenizers
                )
                # NEW: build concept tokens from source images and concatenate
                if args.use_concept_tokens and concept_encoder is not None:
                    bsz = pixel_values1.size(0)
                    # Encoder4 expects float32
                    src_tokens = concept_encoder(pixel_values1.to(torch.float32))  # [B, latent_unit * D]
                    src_tokens = src_tokens.view(bsz, args.concept_token_count, prompt_embeds.shape[-1]).to(prompt_embeds.dtype)
                    src_tokens = src_tokens * args.token_strength_src
                # Convert to latents
                model_input = vae.encode(pixel_values1).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                ).long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Repeat counts for time ids and text embeds
                if not train_dataset.custom_instance_prompts1:
                    elems_to_repeat_text_embeds = bsz // 2 if args.with_prior_preservation else bsz
                    elems_to_repeat_time_ids = bsz // 2 if args.with_prior_preservation else bsz
                else:
                    elems_to_repeat_text_embeds = 1
                    elems_to_repeat_time_ids = bsz // 2 if args.with_prior_preservation else bsz

                if not args.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    # NEW: concat source concept tokens
                    if args.use_concept_tokens and concept_encoder is not None:
                        src_tokens_rep = src_tokens.repeat(elems_to_repeat_text_embeds, 1, 1)
                        prompt_embeds_input = torch.cat([prompt_embeds_input, src_tokens_rep], dim=1)
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                    ).sample

                # Compute loss1

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss1 = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    mse_loss_weights = base_weight + 1 if noise_scheduler.config.prediction_type == "v_prediction" else base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss1 = loss.mean()

                # Backward source branch immediately to keep checkpoint recomputation path consistent.
                accelerator.backward(loss1)

                # Pass 2: target domains (style branch, grouped by target idx)
                GLOBAL_TIME = 1
                if args.freeze_semantic_on_target:
                    set_params_requires_grad(unet_semantic_parameters, False)
                    set_params_requires_grad(unet_style_parameters, True)

                unique_target_idx = torch.unique(target_domain_idx)
                loss2 = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
                dom_loss_logs = {
                    f"dom_loss_{name.replace(' ', '_').lower()}": 0.0
                    for name in target_domain_names
                }
                dom_loss_logs.update(
                    {
                        f"dom_loss_w_{name.replace(' ', '_').lower()}": 0.0
                        for name in target_domain_names
                    }
                )
                sample_logs = {
                    f"sample_ratio_{name.replace(' ', '_').lower()}": 0.0
                    for name in target_domain_names
                }
                sample_logs.update(
                    {
                        f"sample_count_{name.replace(' ', '_').lower()}": 0.0
                        for name in target_domain_names
                    }
                )
                batch_domain_counts = torch.bincount(
                    target_domain_idx, minlength=len(target_domain_names)
                )
                batch_size_target = max(1, int(target_domain_idx.numel()))
                running_target_samples += int(target_domain_idx.numel())

                running_sample_logs = {
                    f"running_sample_ratio_{name.replace(' ', '_').lower()}": 0.0
                    for name in target_domain_names
                }
                for dom_idx, domain_name in enumerate(target_domain_names):
                    domain_slug = domain_name.replace(" ", "_").lower()
                    cnt = int(batch_domain_counts[dom_idx].item())
                    running_domain_counts[dom_idx] += cnt
                    sample_logs[f"sample_count_{domain_slug}"] = float(cnt)
                    sample_logs[f"sample_ratio_{domain_slug}"] = float(cnt / batch_size_target)
                    if running_target_samples > 0:
                        running_sample_logs[f"running_sample_ratio_{domain_slug}"] = float(
                            running_domain_counts[dom_idx] / running_target_samples
                        )

                for dom_idx_t in unique_target_idx:
                    dom_idx = int(dom_idx_t.item())
                    if dom_idx < 0 or dom_idx >= len(target_domain_names):
                        continue
                    set_current_target_index(dom_idx)
                    mask = (target_domain_idx == dom_idx_t)
                    if not torch.any(mask):
                        continue

                    pixel_values2_d = pixel_values2[mask]
                    prompts2_d = [p for i, p in enumerate(prompts2) if bool(mask[i].item())]

                    prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                        prompts2_d, text_encoders, tokenizers
                    )
                    bsz_d = pixel_values2_d.size(0)

                    if args.use_concept_tokens and concept_encoder is not None:
                        tgt_tokens = concept_encoder(pixel_values2_d.to(torch.float32))
                        tgt_tokens = tgt_tokens.view(bsz_d, args.concept_token_count, prompt_embeds.shape[-1]).to(prompt_embeds.dtype)
                        tgt_tokens = tgt_tokens * args.token_strength_tgt

                    model_input_d = vae.encode(pixel_values2_d).latent_dist.sample()
                    model_input_d = model_input_d * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input_d = model_input_d.to(weight_dtype)

                    noise_d = torch.randn_like(model_input_d)
                    timesteps_d = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz_d,), device=model_input_d.device
                    ).long()
                    noisy_model_input_d = noise_scheduler.add_noise(model_input_d, noise_d, timesteps_d)

                    elems_to_repeat_text_embeds = 1
                    elems_to_repeat_time_ids = bsz_d

                    if not args.train_text_encoder:
                        unet_added_conditions = {
                            "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                            "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                        }
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        if args.use_concept_tokens and concept_encoder is not None:
                            prompt_embeds_input = torch.cat([prompt_embeds_input, tgt_tokens], dim=1)
                        model_pred_d = unet(
                            noisy_model_input_d, timesteps_d, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                        ).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target_d = noise_d
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target_d = noise_scheduler.get_velocity(model_input_d, noise_d, timesteps_d)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.snr_gamma is None:
                        dom_loss = F.mse_loss(model_pred_d.float(), target_d.float(), reduction="mean")
                    else:
                        snr = compute_snr(noise_scheduler, timesteps_d)
                        base_weight = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps_d)], dim=1).min(dim=1)[0] / snr
                        )
                        mse_loss_weights = base_weight + 1 if noise_scheduler.config.prediction_type == "v_prediction" else base_weight
                        _loss_tmp = F.mse_loss(model_pred_d.float(), target_d.float(), reduction="none")
                        _loss_tmp = _loss_tmp.mean(dim=list(range(1, len(_loss_tmp.shape)))) * mse_loss_weights
                        dom_loss = _loss_tmp.mean()

                    dom_key = f"dom_loss_{target_domain_names[dom_idx].replace(' ', '_').lower()}"
                    dom_loss_logs[dom_key] = dom_loss.detach().item()

                    weighted_dom_loss = style_loss_weights[dom_idx] * dom_loss
                    dom_w_key = f"dom_loss_w_{target_domain_names[dom_idx].replace(' ', '_').lower()}"
                    dom_loss_logs[dom_w_key] = weighted_dom_loss.detach().item()
                    ema_domain_losses[dom_idx] = (
                        args.target_sampling_ema * ema_domain_losses[dom_idx]
                        + (1.0 - args.target_sampling_ema) * dom_loss.detach().item()
                    )
                    loss2 = loss2 + weighted_dom_loss.detach()
                    accelerator.backward(weighted_dom_loss)

                set_current_target_index(-1)

                if args.freeze_semantic_on_target:
                    # Restore default trainability for next source pass.
                    set_params_requires_grad(unet_semantic_parameters, True)

                loss_orth = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
                loss_style_core_l2 = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
                if len(target_domain_names) > 1:
                    for core_param in unet_lora_parameters2:
                        # core_param shape: [D, r, r]
                        for i in range(len(target_domain_names)):
                            for j in range(i + 1, len(target_domain_names)):
                                kij = torch.matmul(core_param[i].transpose(0, 1), core_param[j])
                                loss_orth = loss_orth + torch.sum(kij * kij)
                for core_param in unet_lora_parameters2:
                    loss_style_core_l2 = loss_style_core_l2 + torch.sum(core_param * core_param)

                loss_orth_weighted = args.orth_lambda * loss_orth
                loss_style_core_l2_weighted = args.style_core_l2_lambda * loss_style_core_l2
                accelerator.backward(loss_orth_weighted)
                accelerator.backward(loss_style_core_l2_weighted)

                loss = (loss1 + loss2 + loss_orth_weighted + loss_style_core_l2_weighted).detach()
                loss1 = loss1.detach()
                loss2 = loss2.detach()
                loss_orth = loss_orth.detach()
                loss_style_core_l2 = loss_style_core_l2.detach()
                if accelerator.sync_gradients:
                    # include Encoder4 params in clipping
                    params_to_clip = (
                        itertools.chain(
                            unet_lora_parameters1,
                            unet_lora_parameters2,
                            text_lora_parameters_one,
                            text_lora_parameters_two,
                            concept_encoder.parameters() if (args.use_concept_tokens and concept_encoder is not None) else []
                        )
                        if args.train_text_encoder
                        else itertools.chain(
                            unet_lora_parameters1,
                            unet_lora_parameters2,
                            concept_encoder.parameters() if (args.use_concept_tokens and concept_encoder is not None) else []
                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if (
                    accelerator.sync_gradients
                    and args.enable_adaptive_target_sampling
                    and len(target_domain_names) > 1
                    and global_step > 0
                    and (global_step % max(1, args.target_sampling_update_steps) == 0)
                ):
                    scores = []
                    for dom_idx in range(len(target_domain_names)):
                        if dom_idx in train_dataset.available_target_domain_indices:
                            s = max(ema_domain_losses[dom_idx], args.target_sampling_eps)
                            scores.append(math.pow(s, args.target_sampling_alpha))
                        else:
                            scores.append(0.0)
                    denom = sum(scores)
                    if denom <= 0:
                        fallback = [
                            1.0 if i in train_dataset.available_target_domain_indices else 0.0
                            for i in range(len(target_domain_names))
                        ]
                        denom = max(sum(fallback), 1.0)
                        new_probs = [x / denom for x in fallback]
                    else:
                        new_probs = [x / denom for x in scores]
                    train_dataset.set_target_sampling_probs(new_probs)
                    current_sampling_probs = new_probs

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


            if (accelerator.sync_gradients and global_step > 0
                    and global_step % 1000 == 0
                    and args.validation_prompt and args.num_validation_images > 0):

                GLOBAL_TIME = 1
                set_current_target_index(0)

                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )

                if not args.train_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        revision=args.revision,
                    )
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder_2",
                        revision=args.revision,
                    )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    if args.seed is not None
                    else None
                )

                #  token
                if args.use_concepts_in_validation and args.gen_concept_image:
                    #
                    enc, embed_dim = _load_concept_encoder_for_inference(pipeline, args, accelerator.device)

                    img_t = _preprocess_concept_image(
                        args.gen_concept_image, args.resolution, args.center_crop
                    ).to(accelerator.device, dtype=torch.float32)

                    with torch.no_grad():
                        toks = enc(img_t)  # [1, latent_unit * embed_dim]

                    toks = toks.view(1, args.concept_token_count, embed_dim).to(
                        accelerator.device, dtype=torch.float32
                    )

                    # selection: indices > num_tokens > all
                    if args.gen_token_indices:
                        sel_idx = _parse_token_indices(args.gen_token_indices, args.concept_token_count)
                    elif args.gen_num_tokens is not None:
                        sel_idx = list(range(max(0, min(args.gen_num_tokens, args.concept_token_count))))
                    else:
                        sel_idx = list(range(args.concept_token_count))

                    toks = toks[:, sel_idx, :] * args.gen_token_strength

                    images = []
                    for _ in range(args.num_validation_images):
                        prompt_embeds, pooled = pipeline.encode_prompt(
                            prompt=[args.validation_prompt],
                            device=accelerator.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        combined = torch.cat([prompt_embeds, toks.to(prompt_embeds.dtype)], dim=1)
                        img = pipeline(
                            prompt_embeds=combined,
                            pooled_prompt_embeds=pooled,
                            generator=generator,
                        ).images[0]
                        images.append(img)
                else:
                    pipeline_args = {"prompt": args.validation_prompt}
                    images = [
                        pipeline(**pipeline_args, generator=generator).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}"
                                    )
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                set_current_target_index(-1)
                torch.cuda.empty_cache()


            logs = {
                "loss": loss.detach().item(),
                "loss1": loss1.detach().item(),
                "loss2": loss2.detach().item(),
                "loss_orth": loss_orth.detach().item(),
                "loss_style_core_l2": loss_style_core_l2.detach().item(),
                "GLOBAL_TIME": GLOBAL_TIME,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            ordered_domain_logs = {}
            for domain_name in sorted_target_domain_names:
                domain_slug = domain_name.replace(" ", "_").lower()
                dom_idx_sorted = target_domain_names.index(domain_name)
                ordered_domain_logs[f"dom_loss_{domain_slug}"] = dom_loss_logs.get(f"dom_loss_{domain_slug}", 0.0)
                ordered_domain_logs[f"dom_loss_w_{domain_slug}"] = dom_loss_logs.get(f"dom_loss_w_{domain_slug}", 0.0)
                ordered_domain_logs[f"style_w_{domain_slug}"] = style_weight_logs.get(f"style_w_{domain_slug}", 0.0)
                ordered_domain_logs[f"sample_count_{domain_slug}"] = sample_logs.get(f"sample_count_{domain_slug}", 0.0)
                ordered_domain_logs[f"sample_ratio_{domain_slug}"] = sample_logs.get(f"sample_ratio_{domain_slug}", 0.0)
                ordered_domain_logs[f"running_sample_ratio_{domain_slug}"] = running_sample_logs.get(
                    f"running_sample_ratio_{domain_slug}", 0.0
                )
                ordered_domain_logs[f"ema_dom_loss_{domain_slug}"] = float(ema_domain_losses[dom_idx_sorted])
                ordered_domain_logs[f"sampling_prob_{domain_slug}"] = float(current_sampling_probs[dom_idx_sorted])
            logs.update(ordered_domain_logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_lora_state_dict(unet)

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = text_encoder_lora_state_dict(
                text_encoder_one.to(torch.float32)
            )
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = text_encoder_lora_state_dict(
                text_encoder_two.to(torch.float32)
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        save_muse_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        save_muse_config(args.output_dir, args, target_domain_names)

        #  concept_encoder
        if args.use_concept_tokens and concept_encoder is not None:
            try:
                ce = accelerator.unwrap_model(concept_encoder).to(torch.float32)
            except Exception:
                ce = concept_encoder
            torch.save(ce.state_dict(), os.path.join(args.output_dir, "concept_encoder.pt"))

        # remove unuse models for save GPU memory
        unet = unet.cpu()
        text_encoder_one = text_encoder_one.cpu()
        text_encoder_two = text_encoder_two.cpu()
        del unet, text_encoder_one, text_encoder_two
        del optimizer
        if args.train_text_encoder:
            del text_encoder_lora_layers, text_encoder_2_lora_layers

        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            from MUSE.generate_pure_noise import inject_style_lora_into_unet

            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            scheduler_args = {}
            if pipeline.scheduler.config.get("variance_type") in {"learned", "learned_range"}:
                scheduler_args["variance_type"] = "fixed_small"
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config, **scheduler_args
            )
            inject_style_lora_into_unet(
                pipeline, args.output_dir, target_style_idx=0,
                lora_scale=1.0, semantic_scale=1.0, style_scale=1.0,
            )
            if args.train_text_encoder:
                saved_weights = load_muse_weights(args.output_dir)
                for prefix, encoder in (("text_encoder", pipeline.text_encoder),
                                         ("text_encoder_2", pipeline.text_encoder_2)):
                    LoraLoaderMixin.load_lora_into_text_encoder(
                        {key: value for key, value in saved_weights.items()
                         if key.startswith(prefix + ".")},
                        network_alphas=None, text_encoder=encoder, prefix=prefix,
                    )
            pipeline = pipeline.to(accelerator.device)
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if args.seed is not None
                else None
            )
            images = [
                pipeline(
                    args.validation_prompt, num_inference_steps=25, generator=generator
                ).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
