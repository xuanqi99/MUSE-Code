# MUSE: Revisiting Diffusion Fine-Tuning for Unsupervised Domain Adaptation

**NeurIPS 2026** · Xuan Qi, Yi Wei, Daniele Berardini, Vito Paolo Pastore, Vittorio Murino

[Paper](https://arxiv.org/abs/2609.33716) · [Project page](https://xuanqi99.github.io/MUSE/) · [Code](https://github.com/xuanqi99/MUSE-Code)

MUSE fine-tunes one diffusion model on a labeled source domain and several unlabeled target domains. A shared semantic adapter learns source class information; target-specific style adapters capture the appearance of each target. The model generates labeled bridge images for downstream UDA.

This repository covers **Office-31, Office-Home, and miniDomainNet**.

## Method

![MUSE pipeline](https://xuanqi99.github.io/MUSE/assets/pipeline.webp)

The SDXL U-Net, VAE, and text encoders remain frozen in the main configuration. Attention projections receive a shared semantic update and a style update with shared input/output projections and one private core per target domain.

Each iteration has two passes. The source pass updates the semantic branch using labeled source images. The target pass freezes that branch and updates the style parameters on unlabeled target images. Core regularization and adaptive target sampling follow `MUSE/train_MUSE.py`. Both passes run in the same training command.

## Installation

Use Python 3.10 or newer. Install a matching PyTorch/torchvision build for your CUDA environment using the [PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The source package did not include a version lock. No exact PyTorch/CUDA combination has been verified for this release, so dependency versions remain unpinned. The trainer checks for Diffusers 0.24.0.dev0 or newer; this is a source-level minimum, not a tested compatibility range. Syntax and utility checks were run with Python 3.12.7. Model imports and GPU execution have not been verified.

Prepare [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) locally. The optional [fp16 VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) can be passed with `--pretrained_vae_model_name_or_path`. The paper's 8-bit Adam recipe also requires:

```bash
python -m pip install bitsandbytes
```

Models and datasets are obtained separately under their respective terms. Training and generation commands below expect a CUDA GPU.

## Datasets

Obtain [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), and miniDomainNet following the [dataset authors' instructions](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#minidomainnet). Keep the original class folder names.

```text
data/
  Office-31/
    amazon/<class>/*.jpg
    dslr/<class>/*.jpg
    webcam/<class>/*.jpg
  OfficeHome/
    Art/<class>/*.jpg
    Clipart/<class>/*.jpg
    Product/<class>/*.jpg
    Real World/<class>/*.jpg
  minidomainnet/domainnet/
    clipart/<class>/*.jpg
    painting/<class>/*.jpg
    real/<class>/*.jpg
    sketch/<class>/*.jpg
    splits_mini/{clipart,painting,real,sketch}_{train,test}.txt
```

Office-31 also accepts the original `<domain>/images/<class>/` layout for diffusion training and generation. For downstream UDA, point the real-domain arguments to the directory containing class folders.

| Dataset | Classes | Source-to-multiple-target tasks |
|---|---:|---|
| Office-31 | 31 | `A2DW`, `D2AW`, `W2AD` |
| Office-Home | 65 | `A2CPR`, `C2APR`, `P2ACR`, `R2ACP` |
| miniDomainNet | 126 | `C2PRS`, `P2CRS`, `R2CPS`, `S2CPR` |

Domain abbreviations follow the directory order above: A/D/W, A/C/P/R, and C/P/R/S. MiniDomainNet fine-tuning and inversion use only `*_train.txt`. Downstream training uses the target training split, with evaluation on the separate target test split.

Diffusion training resizes images and applies the trainer's crop. DDIM translation preserves the source aspect ratio, resizes the long side, and pads to the VAE spatial factor before inference. Class-conditioned synthesis produces square images by default.

## Diffusion fine-tuning

Run from the repository root. This command uses the diffusion optimization settings reported in the paper:

```bash
python MUSE/muse.py train \
  --dataset office31 --task A2DW --data_root data/Office-31 \
  --pretrained_model_name_or_path models/sdxl-base \
  --output_root outputs --instance_prompt 'A [CLASS]' \
  --resolution 1024 --rank 32 --style_rank 64 \
  --train_batch_size 8 --gradient_accumulation_steps 1 \
  --max_train_steps 10000 --learning_rate 5e-5 --lr_theta 100 \
  --optimizer AdamW --use_8bit_adam \
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --adam_weight_decay 1e-4 --max_grad_norm 1.0 \
  --lr_scheduler constant --lr_warmup_steps 0 \
  --mixed_precision fp16 --gradient_checkpointing --seed 0
```

For Office-Home, use `--dataset officehome --task P2ACR --data_root data/OfficeHome`. For miniDomainNet, use `--dataset minidomainnet --task S2CPR --data_root data/minidomainnet/domainnet`. Pass `--minidomainnet_split_dir` if `splits_mini` is stored elsewhere. Repeat for the source tasks in the table.

Private cores use `learning_rate × lr_theta`; semantic factors and shared style projections use `learning_rate`. The supplied code retains `--randomimage 0.75`, which replaces some source class prompts with a generic prompt. This behavior has been preserved; the command is not a newly verified reproduction of the published scores.

Checkpoints are saved every 1,000 updates under `outputs/muse_office31_A2DW/checkpoint-<step>/`. They contain adapter weights, optimizer state, and `muse_config.json`, which records the target-branch order. Append `--resume_from_checkpoint latest` to the same command to resume. Keep the same source, target order, and model configuration. Adaptive sampling statistics are reinitialized on resume, so continuation is not bit-identical.

## Generate the bridge datasets

For each target domain, generation uses these 30 groups:

| Generation | Inversion strength | Adapter strength | Groups |
|---|---|---|---:|
| DDIM inversion | 0.6, 0.7, 0.8, 0.9, 1.0 | 0.6, 0.7, 0.8, 0.9, 1.0 | 25 |
| Class-conditioned | — | 0.6, 0.7, 0.8, 0.9, 1.0 | 5 |

```bash
python MUSE/muse.py bridge \
  --dataset office31 --task A2DW --data_root data/Office-31 \
  --pretrained_model_name_or_path models/sdxl-base \
  --checkpoint_dir outputs/muse_office31_A2DW/checkpoint-10000 \
  --output_root generated \
  --num_inference_steps 50 --num_inversion_steps 50 \
  --num_images_per_class 50 --max_images_per_class 0 --seed 0
```

Use the corresponding dataset, task, data root, and checkpoint for Office-Home and miniDomainNet. Add `--dry_run` to print all generation and merge commands without loading models or writing outputs.

Adapter strength scales the fused semantic and target-style update through `--lora_scale`. The existing branch multipliers remain `--semantic_scale 0.8 --style_scale 2.0`. Source inversion uses a fixed semantic adapter strength of 1.0, allowing its five caches to be reused across generation strengths. DDIM denoising starts at the matching inversion timestep. Both step counts must match.

Guidance defaults are 1.0 for inversion and DDIM translation, and 7.5 for class-conditioned synthesis. Seeds are fixed across strength groups. Each DDIM group translates every source training image once; `--max_images_per_class` sets an optional cap. Each class-conditioned group generates 50 images per class. With source count `N_c` and class-conditioned count `P`, a target bridge contains `25 × N_c + 5 × P` images for class `c`.

The 30-group bridge has not been evaluated against the published tables below.

```text
generated/bridge_office31_A2DW/
  plan.json
  groups/
    ddim_inv_0.6_adapter_0.6/dslr/
      images/<class>/*.png
      metadata.jsonl
      completion.json
    ...
    class_condition_adapter_1.0/dslr/...
  inversion_cache/
  uda/
    dslr/
      images/<class>/<group>/*.png
      metadata.jsonl
      classes.json
    webcam/...
  bridge.json
```

Each target's `uda/<target>/images` is a single ImageFolder containing all 30 groups. The merge uses hard links and verifies group completion, metadata, configuration, and class vocabulary. Metadata records labels, generation mode, strengths, and group-image paths. Group metadata retains prompts, seeds, and DDIM source-image paths. Use a new output directory when changing settings.

To rebuild the merge after generation:

```bash
python MUSE/prepare_bridge.py --root generated/bridge_office31_A2DW
```

## Downstream UDA

Train separately for each target using its pooled bridge as labeled source data and real images as unlabeled target data.

ELS with the supplied CDAN, MCC, and SAM training code:

```bash
python UDA/ELS/cdan_mcc_sdat.py data/ImageFolder \
  --source-train-dir generated/bridge_office31_A2DW/uda/dslr/images \
  --target-train-dir data/Office-31/dslr \
  --target-test-dir data/Office-31/dslr \
  --arch resnet50 --epochs 20 --iters-per-epoch 1000 \
  --eps 0.1 --seed 0 --gpu 0 --log logs/els_office31_A2D
```

The fixed epsilon is an executable example, not a newly established paper setting. The original ELS epsilon sweep remains available through `--eps 1`. To evaluate a fixed-epsilon run, repeat the command with `--phase test` and the same epsilon and log directory.

SSRT with the supplied torchvision ViT implementation:

```bash
python UDA/SSRT/ssrt_sdat.py \
  --source-train-dir generated/bridge_office31_A2DW/uda/dslr/images \
  --target-train-dir data/Office-31/dslr \
  --target-test-dir data/Office-31/dslr \
  --arch vit_b_16 --epochs 20 --iters-per-epoch 1000 \
  --seed 0 --gpu 0 --log logs/ssrt_office31_A2D
```

Append `--phase test --checkpoint logs/ssrt_office31_A2D/best.pth` to evaluate. This implementation includes self-refinement, domain adversarial loss, and safe training; the filename does not imply a SAM/SDAT optimizer.

For Office-Home, use the matching bridge and real target class directories. For miniDomainNet, use the matching bridge and add these target arguments to either entrypoint, replacing `real` with the chosen target:

```bash
--target-train-dir data/minidomainnet/domainnet/real \
--target-test-dir data/minidomainnet/domainnet/real \
--target-train-list data/minidomainnet/domainnet/splits_mini/real_train.txt \
--target-test-list data/minidomainnet/domainnet/splits_mini/real_test.txt
```

Both entrypoints use target evaluation accuracy to select the best checkpoint. Target labels enter that evaluation, not training losses. Report this selection rule when comparing results. The supplied package has no standalone MCC training entrypoint.

## Main paper results

Average target top-1 accuracy (%) from the [main paper tables](https://arxiv.org/html/2609.33716v1#S4). These are published MUSE results, not scores reproduced in this environment.

| Dataset | MCC + MUSE | ELS + MUSE | SSRT + MUSE |
|---|---:|---:|---:|
| Office-31 | 91.77 | 92.36 | 95.53 |
| Office-Home | 76.53 | 77.57 | 88.15 |
| miniDomainNet | 64.02 | 65.04 | 84.37 |

Fine-tune once per source task, generate its target bridges, and train one UDA model per target. Report the mean over 6 Office-31, 12 Office-Home, and 12 miniDomainNet transfers. Full reproduction still requires the original downstream experiment configurations and model environments; these commands alone do not establish those scores.

No pretrained MUSE checkpoint is included or currently linked for download.

## Checks

```bash
python -m compileall -q MUSE UDA tests
python -m unittest discover -s tests -v
```

These checks cover source syntax, the strength grid, command arguments, generation bookkeeping, and bridge assembly without downloading models or running training.

## Citation

```bibtex
@inproceedings{qi2026muse,
  title={Revisiting Diffusion Fine-Tuning for Unsupervised Domain Adaptation},
  author={Qi, Xuan and Wei, Yi and Berardini, Daniele and Pastore, Vito Paolo and Murino, Vittorio},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments

This code builds on SDXL, Hugging Face Diffusers, Transfer-Learning-Library, SDAT, ELS, and SSRT. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources, retained notices, and the ELS redistribution permission.
