# Third-party code and assets

The existing code repository supplies the root MIT license for the authors' contributions. It does not replace the terms of reused code or model weights.

| Component | Source | Notice |
|---|---|---|
| SDXL training scaffolding in `MUSE/train_MUSE.py` | [Diffusers DreamBooth SDXL example](https://github.com/huggingface/diffusers/blob/v0.24.0/examples/dreambooth/train_dreambooth_lora_sdxl.py) | Copyright 2023 The HuggingFace Inc. team; [Apache 2.0](licenses/diffusers-Apache-2.0.txt). Modified for MUSE adapters, optimization, datasets, and checkpoint handling. |
| Transfer utilities in `UDA/ELS/common` and `UDA/ELS/dalib` | [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) | [MIT notice](UDA/ELS/licenses/Transfer-Learning-Library.txt) |
| SAM and downstream training scaffolding | [SDAT](https://github.com/val-iisc/SDAT) | [MIT notice](UDA/ELS/licenses/SDAT.txt) |
| SSRT reference code | [SSRT](https://github.com/tsun/SSRT) | Copyright 2022 tsun; [MIT notice](UDA/SSRT/LICENSE) |
| ELS-specific modifications | [Environment-Label-Smoothing](https://github.com/yfzhang114/Environment-Label-Smoothing) | Redistributed with permission, confirmed by the repository owner on 2026-10-24. The owner confirmed that source attribution and acknowledgment satisfy the notice requirements. The root MIT license does not relicense upstream contributions. |

The method illustration is linked from the authors' project page. Datasets, generated bridge images, and pretrained weights are not included. SDXL weights have separate [model terms](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
