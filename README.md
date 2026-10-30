# MUSE: Revisiting Diffusion Fine-Tuning for Unsupervised Domain Adaptation

**NeurIPS 2026**

Xuan Qi, Yi Wei, Daniele Berardini, Vito Paolo Pastore, Vittorio Murino

[Paper](https://arxiv.org/abs/2609.33716) · [Project page](https://xuanqi99.github.io/MUSE/) · [Code](https://github.com/xuanqi99/MUSE-Code)

MUSE uses shared semantic and target-specific style adapters to generate bridge data for unsupervised domain adaptation.

## Setup

Use Python 3.10+, PyTorch with CUDA, and a local [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model.

```bash
python -m pip install -r requirements.txt
python -m pip install bitsandbytes
```

## Data

Keep images under `<data_root>/<domain>/<class>/`. Office-31 also supports `<domain>/images/<class>/`.

| Dataset | Domain folders | Tasks |
|---|---|---|
| [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/) | `amazon`, `dslr`, `webcam` | `A2DW`, `D2AW`, `W2AD` |
| [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) | `Art`, `Clipart`, `Product`, `Real World` | `A2CPR`, `C2APR`, `P2ACR`, `R2ACP` |
| [miniDomainNet](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#minidomainnet) | `clipart`, `painting`, `real`, `sketch` | `C2PRS`, `P2CRS`, `R2CPS`, `S2CPR` |

For miniDomainNet, put the official `*_train.txt` and `*_test.txt` files in `<data_root>/splits_mini/`.

## Fine-tuning

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

Change `--dataset`, `--task`, and `--data_root` for the other datasets. Checkpoints are saved in `outputs/muse_<dataset>_<task>/checkpoint-<step>/`. Add `--resume_from_checkpoint latest` to resume.

## Bridge generation

```bash
python MUSE/muse.py bridge \
  --dataset office31 --task A2DW --data_root data/Office-31 \
  --pretrained_model_name_or_path models/sdxl-base \
  --checkpoint_dir outputs/muse_office31_A2DW/checkpoint-10000 \
  --output_root generated \
  --num_inference_steps 50 --num_inversion_steps 50 \
  --num_images_per_class 50 --max_images_per_class 0 --seed 0
```

`lora_scale=1.0` and `style_scale=1.0` remain fixed. Inversion strength and `semantic_scale` each take **0.6, 0.7, 0.8, 0.9, 1.0**.

Each `(inversion strength, semantic_scale)` bridge combines DDIM-translated images with class-conditioned images at the same `semantic_scale`. This gives **25 separate bridge datasets per target domain**. The five class-conditioned sets are generated once and reused across inversion strengths. The default uses all source training images for DDIM and 50 class-conditioned images per class.

Each bridge is saved under `uda/<target>/inv_<strength>_semantic_<scale>/images/`.

Use a new output directory for this configuration. Add `--dry_run` to preview generation commands.

## UDA training

```bash
python MUSE/train_uda.py \
  --bridge_root generated/bridge_office31_A2DW \
  --data_root data/Office-31 --method els --log_root logs \
  --arch resnet50 --epochs 20 --iters-per-epoch 1000 \
  --eps 0.1 --seed 0 --gpu 0
```

This trains one model for each bridge and target, with separate logs and checkpoints. For SSRT, use `--method ssrt --arch vit_b_16` and omit `--eps`. Add `--dry_run` to preview the jobs. MiniDomainNet uses the official target train/test splits.

Checkpoints are selected by target evaluation accuracy.

## Main results

Average target accuracy (%) reported in the paper:

| Dataset | MCC + MUSE | ELS + MUSE | SSRT + MUSE |
|---|---:|---:|---:|
| Office-31 | 91.77 | 92.36 | 95.53 |
| Office-Home | 76.53 | 77.57 | 88.15 |
| miniDomainNet | 64.02 | 65.04 | 84.37 |

## Citation

```bibtex
@inproceedings{qi2026muse,
  title={Revisiting Diffusion Fine-Tuning for Unsupervised Domain Adaptation},
  author={Qi, Xuan and Wei, Yi and Berardini, Daniele and Pastore, Vito Paolo and Murino, Vittorio},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

Built on SDXL, Diffusers, Transfer-Learning-Library, SDAT, ELS, and SSRT. Sources and license notices are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
