import argparse
import json
import logging
import os
import os.path as osp
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

UDA_DIR = Path(__file__).resolve().parent.parent
if str(UDA_DIR) not in sys.path:
    sys.path.insert(0, str(UDA_DIR))
from imagefolder_splits import load_imagefolder, remap_class_indices


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


class ForeverDataIterator:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            return next(self.iter)

    def __len__(self):
        return len(self.data_loader)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        cudnn.benchmark = True
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _resolve_imagefolder_root(path_str: str) -> str:
    path = Path(path_str)
    images_child = path / "images"
    if images_child.is_dir():
        return str(images_child)
    if path.is_dir():
        return str(path)
    if path.name.lower() == "images" and path.parent.is_dir():
        return str(path.parent)
    raise FileNotFoundError(
        f"ImageFolder root not found: {path}. Expected the directory itself or an 'images' child."
    )


class AlignedImageFolder(Dataset):
    def __init__(self, base_dataset: ImageFolder, ref_class_to_idx: dict):
        self.base = base_dataset
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        self.loader = base_dataset.loader
        self.ref_class_to_idx = ref_class_to_idx

        remap = remap_class_indices(base_dataset.class_to_idx, ref_class_to_idx)

        self.samples = []
        dropped = 0
        for path, old_idx in base_dataset.samples:
            if old_idx in remap:
                self.samples.append((path, remap[old_idx]))
            else:
                dropped += 1

        if dropped > 0:
            warnings.warn(f"Dropped {dropped} samples that could not be remapped to source classes.")
        if len(self.samples) == 0:
            raise ValueError(
                "After class remapping, no samples remain. Check class naming consistency between source and target."
            )

        self.targets = [t for _, t in self.samples]
        self.classes = [None] * len(ref_class_to_idx)
        for cls_name, idx in ref_class_to_idx.items():
            self.classes[idx] = cls_name
        self.class_to_idx = dict(ref_class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        image = self.loader(path)
        image = self.transform(image) if self.transform is not None else image
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, idx


def build_train_transform(args: argparse.Namespace, use_rand_aug: bool = False) -> T.Compose:
    transforms = [T.Resize((args.resize_size, args.resize_size))]
    if use_rand_aug and args.rand_aug:
        transforms.append(T.RandAugment(num_ops=args.rand_aug_num_ops, magnitude=args.rand_aug_magnitude))
    transforms.append(T.RandomResizedCrop(args.crop_size))
    if args.color_jitter > 0:
        jitter = args.color_jitter
        transforms.append(T.ColorJitter(jitter, jitter, jitter, min(0.5, jitter)))
    if not args.no_hflip:
        transforms.append(T.RandomHorizontalFlip())
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=args.norm_mean, std=args.norm_std),
        ]
    )
    if args.random_erasing > 0:
        transforms.append(T.RandomErasing(p=args.random_erasing))
    return T.Compose(transforms)


def build_val_transform(args: argparse.Namespace) -> T.Compose:
    return T.Compose(
        [
            T.Resize((args.resize_size, args.resize_size)),
            T.CenterCrop(args.crop_size),
            T.ToTensor(),
            T.Normalize(mean=args.norm_mean, std=args.norm_std),
        ]
    )


def get_custom_path_datasets(args: argparse.Namespace):
    train_transform = build_train_transform(args, use_rand_aug=False)
    target_train_transform = build_train_transform(args, use_rand_aug=True)
    val_transform = build_val_transform(args)

    source_train_dirs = []
    if args.source_train_dir:
        source_train_dirs.append(args.source_train_dir)
    if args.source_train_dirs:
        source_train_dirs.extend(args.source_train_dirs)
    source_train_dirs = list(dict.fromkeys(source_train_dirs))
    if not source_train_dirs:
        raise ValueError("At least one source synthetic directory is required.")
    if not args.target_train_dir or not args.target_test_dir:
        raise ValueError("Both --target-train-dir and --target-test-dir are required.")

    resolved_source_train_dirs = [_resolve_imagefolder_root(p) for p in source_train_dirs]
    resolved_target_train_dir = _resolve_imagefolder_root(args.target_train_dir)
    resolved_target_test_dir = _resolve_imagefolder_root(args.target_test_dir)

    raw_source_datasets = [ImageFolder(src_dir, transform=train_transform) for src_dir in resolved_source_train_dirs]
    ref_dataset = raw_source_datasets[0]
    ref_class_to_idx = ref_dataset.class_to_idx
    class_names = ref_dataset.classes
    num_classes = len(class_names)

    aligned_source_datasets = [AlignedImageFolder(ds, ref_class_to_idx) for ds in raw_source_datasets]
    if len(aligned_source_datasets) == 1:
        train_source_dataset: Dataset = aligned_source_datasets[0]
    else:
        train_source_dataset = ConcatDataset(aligned_source_datasets)

    raw_target_train = load_imagefolder(
        resolved_target_train_dir, transform=target_train_transform, split_file=args.target_train_list
    )
    raw_target_test = load_imagefolder(
        resolved_target_test_dir, transform=val_transform, split_file=args.target_test_list
    )

    train_target_dataset = AlignedImageFolder(raw_target_train, ref_class_to_idx)
    test_dataset = AlignedImageFolder(raw_target_test, ref_class_to_idx)

    return (
        train_source_dataset,
        train_target_dataset,
        test_dataset,
        num_classes,
        class_names,
        train_transform,
        target_train_transform,
        val_transform,
    )


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, coeff: float):
        ctx.coeff = coeff
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 0.1, max_iters: int = 1000):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.max_iters = max_iters
        self.iter_num = 0

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )
        self.iter_num += 1
        return GradientReverseFunction.apply(input_tensor, coeff)


class TorchvisionViTBackbone(nn.Module):
    def __init__(self, arch: str, pretrained: bool, sr_alpha: float, sr_layers: Sequence[int], logger: logging.Logger):
        super().__init__()
        self.logger = logger
        self.arch = arch
        self.sr_layers = list(sr_layers)
        self.sr_alpha = sr_alpha
        self.sr_alpha_adap = sr_alpha
        self.embed_dim = 768

        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except Exception as exc:
            raise RuntimeError(
                "torchvision with vit_b_16 support is required for SSRT. "
                "Please use a torchvision build that includes Vision Transformer."
            ) from exc

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None

        try:
            self.model = vit_b_16(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the ViT backbone. Cache its torchvision pretrained weights, "
                "or explicitly pass --pretrained-backbone false to train from scratch."
            ) from exc

        self.embed_dim = getattr(self.model, "hidden_dim", 768)

    def _get_target_mix(self, tensor: torch.Tensor, source_batch_size: int) -> torch.Tensor:
        total_batch = tensor.shape[0]
        target_start = source_batch_size
        idx = torch.flip(torch.arange(target_start, total_batch, device=tensor.device), dims=[0])
        mixed = tensor[target_start:] + (tensor[idx] - tensor[target_start:]).detach() * self.sr_alpha_adap
        return mixed

    def forward_features(self, x: torch.Tensor, source_batch_size: int) -> torch.Tensor:
        if self.training and self.sr_layers:
            perturb_layer = random.choice(self.sr_layers)
        else:
            perturb_layer = None

        if perturb_layer == -1:
            mixed = self._get_target_mix(x, source_batch_size)
            x = torch.cat([x, mixed], dim=0)

        y = self.model._process_input(x)
        cls_tokens = self.model.class_token.expand(y.shape[0], -1, -1)
        y = torch.cat((cls_tokens, y), dim=1)

        encoder = self.model.encoder
        y = y + encoder.pos_embedding
        y = encoder.dropout(y)

        for layer_idx, layer in enumerate(encoder.layers):
            if self.training and perturb_layer == layer_idx:
                mixed = self._get_target_mix(y, source_batch_size)
                y = torch.cat([y, mixed], dim=0)
            y = layer(y)

        y = encoder.ln(y)
        return y[:, 0]


class SSRTNet(nn.Module):
    def __init__(self, num_classes: int, args: argparse.Namespace, logger: logging.Logger):
        super().__init__()
        arch = args.arch.lower()
        if arch not in {"vit_b_16", "vit_base_patch16_224"}:
            raise ValueError(
                f"Unsupported SSRT backbone '{args.arch}'. This script currently supports vit_b_16 / "
                "vit_base_patch16_224."
            )

        self.backbone = TorchvisionViTBackbone(
            arch=args.arch,
            pretrained=args.pretrained_backbone,
            sr_alpha=args.sr_alpha,
            sr_layers=args.sr_layers,
            logger=logger,
        )
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=args.adap_adjust_T)
        feature_dim = self.backbone.embed_dim
        self.use_bottleneck = args.use_bottleneck

        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(feature_dim, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            classifier_dim = args.bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            classifier_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(args.hidden_dim, num_classes),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(classifier_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(args.hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.use_bottleneck and isinstance(self.bottleneck[0], nn.Linear):
            nn.init.normal_(self.bottleneck[0].weight, 0, 0.005)
            nn.init.constant_(self.bottleneck[0].bias, 0.1)
        for layer in (self.classifier[0], self.classifier[3], self.discriminator[0], self.discriminator[3]):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor, source_batch_size: int):
        features = self.backbone.forward_features(inputs, source_batch_size=source_batch_size)
        features = self.bottleneck(features)
        logits = self.classifier(features)
        domain_logits = self.discriminator(self.grl(features))
        return features, logits, domain_logits

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(inputs, source_batch_size=inputs.shape[0] // 2 if inputs.shape[0] > 1 else 0)
        features = self.bottleneck(features)
        return self.classifier(features)

    def get_parameter_groups(self, base_lr: float):
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.classifier.parameters(), "lr": base_lr},
            {"params": self.discriminator.parameters(), "lr": base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
        ]


class SSRTModel:
    def __init__(self, num_classes: int, args: argparse.Namespace, logger: logging.Logger):
        self.net = SSRTNet(num_classes=num_classes, args=args, logger=logger).to(args.device)
        self.args = args
        self.logger = logger
        self.iter_num = 0
        self.use_safe_training = args.use_safe_training
        self.sr_loss_weight = args.sr_loss_weight
        self.sr_loss_weight_adap = args.sr_loss_weight
        self.snap_shot = None
        self.restore = False
        self.r = 0.0
        self.r_period = args.adap_adjust_T
        self.r_phase = 0
        self.r_mag = 1.0
        self.adap_adjust_last_restore_iter = 0
        self.divs: List[float] = []
        self.divs_last_period: Optional[List[float]] = None
        self.bce_logits = nn.BCEWithLogitsLoss()

    def save_snapshot(self) -> None:
        self.snap_shot = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    def restore_snapshot(self) -> None:
        if self.snap_shot is None:
            return
        state_dict = {k: v.to(self.args.device) for k, v in self.snap_shot.items()}
        self.net.load_state_dict(state_dict, strict=False)
        self.adap_adjust_last_restore_iter = self.iter_num

    def get_adjust(self, iteration: int) -> float:
        if iteration >= self.r_period + self.r_phase:
            return self.r_mag
        return float(np.sin((iteration - self.r_phase) / self.r_period * np.pi / 2.0) * self.r_mag)

    def check_div_drop(self) -> None:
        flag = False
        for level in range(self.args.adap_adjust_L + 1):
            chunks = int(np.power(2, level))
            divs_split = np.array_split(np.array(self.divs, dtype=np.float32), chunks)
            divs_mean = [d.mean() for d in divs_split if len(d) > 0]
            if self.args.adap_adjust_append_last_subintervals and self.divs_last_period is not None:
                prev_split = np.array_split(np.array(self.divs_last_period, dtype=np.float32), chunks)
                prev_mean = [d.mean() for d in prev_split if len(d) > 0]
                if prev_mean:
                    divs_mean.insert(0, prev_mean[-1])
            for i in range(len(divs_mean) - 1):
                if divs_mean[i + 1] < divs_mean[i] - 1.0:
                    flag = True
        if self.r <= 0.1:
            flag = False
        if flag:
            self.restore = True
            self.r_phase = self.iter_num
            if self.iter_num - self.adap_adjust_last_restore_iter <= self.r_period:
                self.r_period *= 2

    def get_sr_loss(self, logits_target: torch.Tensor, logits_target_perturbed: torch.Tensor) -> torch.Tensor:
        prob1_t = F.softmax(logits_target.detach(), dim=1)
        prob2_t = F.softmax(logits_target_perturbed.detach(), dim=1)
        prob1 = F.softmax(logits_target, dim=1)
        log_prob1 = F.log_softmax(logits_target, dim=1)
        prob2 = F.softmax(logits_target_perturbed, dim=1)
        log_prob2 = F.log_softmax(logits_target_perturbed, dim=1)

        if random.random() <= self.args.sr_loss_p:
            mask = (prob1_t.max(dim=1)[0] > self.args.sr_epsilon).float()
            aug_loss = ((prob1 * (log_prob1 - log_prob2)).sum(dim=1) * mask).sum() / (mask.sum() + 1e-6)
        else:
            mask = (prob2_t.max(dim=1)[0] > self.args.sr_epsilon).float()
            aug_loss = ((prob2 * (log_prob2 - log_prob1)).sum(dim=1) * mask).sum() / (mask.sum() + 1e-6)

        if self.use_safe_training:
            self.r = self.get_adjust(self.iter_num)
            self.net.backbone.sr_alpha_adap = self.net.backbone.sr_alpha * self.r
            self.sr_loss_weight_adap = self.sr_loss_weight * self.r
            div_unique = int(prob1.argmax(dim=1).unique().shape[0])
            self.divs.append(div_unique)
            if (self.iter_num + 1) % self.args.adap_adjust_T == 0 and self.iter_num > 0:
                self.check_div_drop()
                if not self.restore:
                    self.divs_last_period = list(self.divs)

        return aug_loss

    def get_loss(self, inputs_source: torch.Tensor, inputs_target: torch.Tensor, labels_source: torch.Tensor):
        if self.use_safe_training:
            if self.restore and self.iter_num > 0 and self.sr_loss_weight > 0:
                self.restore_snapshot()
                self.restore = False
                self.logger.info("Iter %d: restored model snapshot (r=%.4f)", self.iter_num, self.r)
            if self.iter_num % self.args.adap_adjust_T == 0 and self.sr_loss_weight > 0:
                self.save_snapshot()
                self.divs = []
                self.logger.info("Iter %d: saved model snapshot (r=%.4f)", self.iter_num, self.r)

        source_batch_size = labels_source.size(0)
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        _, logits, domain_logits = self.net(inputs, source_batch_size=source_batch_size)

        classification_loss = F.cross_entropy(logits.narrow(0, 0, source_batch_size), labels_source)
        original_total = inputs.size(0)
        domain_labels = torch.cat(
            (
                torch.ones(inputs_source.shape[0], device=inputs.device, dtype=torch.float32),
                torch.zeros(inputs_target.shape[0], device=inputs.device, dtype=torch.float32),
            ),
            dim=0,
        )
        domain_loss = self.bce_logits(domain_logits.narrow(0, 0, original_total).squeeze(1), domain_labels) * 2.0

        total_loss = (
            self.args.classification_loss_weight * classification_loss
            + self.args.domain_loss_weight * domain_loss
        )

        sr_loss = torch.zeros((), device=inputs.device)
        if self.args.sr_loss_weight > 0 and logits.size(0) > original_total:
            logits_target = logits.narrow(0, source_batch_size, inputs_target.shape[0])
            logits_target_perturbed = logits.narrow(0, original_total, inputs_target.shape[0])
            sr_loss = self.get_sr_loss(logits_target, logits_target_perturbed)
            total_loss = total_loss + self.sr_loss_weight_adap * sr_loss

        mi_loss = torch.zeros((), device=inputs.device)
        if self.args.mi_loss_weight > 0:
            softmax_out = F.softmax(logits.narrow(0, source_batch_size, inputs_target.shape[0]), dim=1)
            entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-6), dim=1))
            marginal = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-marginal * torch.log(marginal + 1e-6))
            mi_loss = entropy_loss - gentropy_loss
            total_loss = total_loss + self.args.mi_loss_weight * mi_loss

        self.iter_num += 1
        metrics = {
            "cls": float(classification_loss.detach().item()),
            "domain": float(domain_loss.detach().item()),
            "sr": float(sr_loss.detach().item()),
            "mi": float(mi_loss.detach().item()),
        }
        return total_loss, metrics

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            logits = self.net.predict(inputs)
        return logits

    def train(self, mode: bool = True):
        self.net.train(mode)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict, strict=True)


def build_optimizer_and_scheduler(model: SSRTModel, args: argparse.Namespace):
    optimizer = SGD(
        model.net.get_parameter_groups(args.lr),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (1.0 + args.lr_gamma * float(step)) ** (-args.lr_decay),
    )
    return optimizer, lr_scheduler


@dataclass
class EvalResult:
    accuracy: float
    per_class_accuracy: float
    loss: float


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    correct = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    for true_label, pred_label in zip(y_true, y_pred):
        total[true_label] += 1
        if true_label == pred_label:
            correct[true_label] += 1
    valid = total > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(correct[valid] / total[valid]))


def evaluate(model: SSRTModel, data_loader: DataLoader, device: torch.device, num_classes: int) -> EvalResult:
    model.train(False)
    all_preds = []
    all_targets = []
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model.predict(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            losses.append(float(loss.item()))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    accuracy = float((y_pred == y_true).mean())
    per_class_accuracy = compute_per_class_accuracy(y_true, y_pred, num_classes=num_classes)
    return EvalResult(
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        loss=float(np.mean(losses)) if losses else 0.0,
    )


def train_one_epoch(
    model: SSRTModel,
    train_source_iter: ForeverDataIterator,
    train_target_iter: ForeverDataIterator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    model.train(True)
    epoch_metrics = {"loss": 0.0, "cls": 0.0, "domain": 0.0, "sr": 0.0, "mi": 0.0}
    start_time = time.time()

    for step in range(args.iters_per_epoch):
        source_batch = next(train_source_iter)
        target_batch = next(train_target_iter)

        inputs_source = source_batch[0].to(args.device, non_blocking=True)
        labels_source = source_batch[1].to(args.device, non_blocking=True)
        inputs_target = target_batch[0].to(args.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = model.get_loss(inputs_source, inputs_target, labels_source)
        total_loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), args.grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        epoch_metrics["loss"] += float(total_loss.item())
        for key, value in metrics.items():
            epoch_metrics[key] += value

        if step % args.print_freq == 0:
            logger.info(
                "Epoch %d [%d/%d] loss=%.4f cls=%.4f domain=%.4f sr=%.4f mi=%.4f lr=%.6f",
                epoch,
                step,
                args.iters_per_epoch,
                total_loss.item(),
                metrics["cls"],
                metrics["domain"],
                metrics["sr"],
                metrics["mi"],
                optimizer.param_groups[-1]["lr"],
            )

    elapsed = time.time() - start_time
    for key in epoch_metrics:
        epoch_metrics[key] /= max(1, args.iters_per_epoch)
    epoch_metrics["time"] = elapsed
    return epoch_metrics


def save_checkpoint(path: str, model: SSRTModel, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        },
        path,
    )


def main(args: argparse.Namespace):
    os.makedirs(args.log, exist_ok=True)
    logger = init_logging(args.log)
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    logger.info("Arguments: %s", args)
    logger.info(
        "[INFO] Using custom path datasets (ImageFolder). --data/--source/--target are kept only as metadata."
    )

    (
        train_source_dataset,
        train_target_dataset,
        test_dataset,
        num_classes,
        class_names,
        train_transform,
        target_train_transform,
        val_transform,
    ) = get_custom_path_datasets(args)

    logger.info("train_transform(source) = %s", train_transform)
    logger.info("train_transform(target) = %s", target_train_transform)
    logger.info("val_transform = %s", val_transform)
    logger.info(
        "Dataset sizes: source=%d target=%d test=%d num_classes=%d",
        len(train_source_dataset),
        len(train_target_dataset),
        len(test_dataset),
        num_classes,
    )

    with open(osp.join(args.log, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    args.class_names = class_names

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.phase == 'train' and (len(train_source_loader) == 0 or len(train_target_loader) == 0):
        raise ValueError("Source and target training sets must each contain at least --batch-size images.")

    if args.phase == 'test':
        args.pretrained_backbone = False
    model = SSRTModel(num_classes=num_classes, args=args, logger=logger)
    if args.phase == 'test':
        checkpoint_path = args.checkpoint or osp.join(args.log, 'best.pth')
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        saved_classes = checkpoint.get('args', {}).get('class_names')
        if saved_classes is not None and saved_classes != class_names:
            raise ValueError('Checkpoint class order does not match the source dataset.')
        model.load_state_dict(checkpoint['model'])
        result = evaluate(model, test_loader, device=args.device, num_classes=num_classes)
        logger.info('Test accuracy=%.4f per-class accuracy=%.4f', result.accuracy, result.per_class_accuracy)
        return result

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, args)

    best_acc = 0.0
    best_epoch = 0
    summary = []
    latest_ckpt = osp.join(args.log, "latest.pth")
    best_ckpt = osp.join(args.log, "best.pth")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_source_iter=train_source_iter,
            train_target_iter=train_target_iter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            args=args,
            logger=logger,
        )
        eval_result = evaluate(model, test_loader, device=args.device, num_classes=num_classes)
        summary.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "test_accuracy": eval_result.accuracy,
                "test_per_class_accuracy": eval_result.per_class_accuracy,
                "test_loss": eval_result.loss,
            }
        )

        logger.info(
            "Epoch %d done: train_loss=%.4f test_acc=%.4f test_bal_acc=%.4f test_loss=%.4f time=%.1fs",
            epoch,
            train_metrics["loss"],
            eval_result.accuracy,
            eval_result.per_class_accuracy,
            eval_result.loss,
            train_metrics["time"],
        )

        save_checkpoint(latest_ckpt, model, optimizer, epoch, best_acc, args)
        if eval_result.accuracy >= best_acc:
            best_acc = eval_result.accuracy
            best_epoch = epoch
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_acc, args)

        if epoch % max(1, args.eval_epoch) == 0:
            logger.info("Checkpoint status: best_acc=%.4f at epoch %d", best_acc, best_epoch)

    with open(osp.join(args.log, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training finished. best_acc=%.4f best_epoch=%d", best_acc, best_epoch)


def parse_args():
    parser = argparse.ArgumentParser("Synthetic-source SSRT training")
    parser.add_argument("--data", type=str, default="OfficeHome")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--source", nargs="+", default=None)
    parser.add_argument("--target", nargs="+", default=None)

    parser.add_argument("--source-train-dir", type=str, default="")
    parser.add_argument("--source-train-dirs", nargs="+", default=None)
    parser.add_argument("--target-train-dir", type=str, required=True)
    parser.add_argument("--target-test-dir", type=str, required=True)
    parser.add_argument("--target-train-list", type=str, default=None,
                        help="Official train split (image_path label), relative to the target domain root.")
    parser.add_argument("--target-test-list", type=str, default=None,
                        help="Official test split (image_path label), relative to the target domain root.")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SSRT checkpoint for --phase test; defaults to <log>/best.pth.")

    parser.add_argument("--arch", type=str, default="vit_b_16")
    parser.add_argument("--pretrained-backbone", type=str2bool, default=True)
    parser.add_argument("--use-bottleneck", type=str2bool, default=True)
    parser.add_argument("--bottleneck-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iters-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-epoch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--lr-gamma", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.75)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)

    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument("--mi-loss-weight", type=float, default=0.0)

    parser.add_argument("--sr-alpha", type=float, default=0.3)
    parser.add_argument("--sr-layers", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--sr-loss-p", type=float, default=0.5)
    parser.add_argument("--sr-loss-weight", type=float, default=0.2)
    parser.add_argument("--sr-epsilon", type=float, default=0.4)

    parser.add_argument("--use-safe-training", type=str2bool, default=True)
    parser.add_argument("--adap-adjust-T", type=int, default=1000)
    parser.add_argument("--adap-adjust-L", type=int, default=4)
    parser.add_argument("--adap-adjust-append-last-subintervals", type=str2bool, default=True)

    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--rand-aug", action="store_true")
    parser.add_argument("--rand-aug-num-ops", type=int, default=1)
    parser.add_argument("--rand-aug-magnitude", type=int, default=2)
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--norm-mean", nargs="+", type=float, default=(0.485, 0.456, 0.406))
    parser.add_argument("--norm-std", nargs="+", type=float, default=(0.229, 0.224, 0.225))

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
