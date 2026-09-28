# -*- coding: utf-8 -*-
"""
Full-supervised 4-class corneal segmentation training for SARMambaMGEFUNet
=========================================================================

Task
----
Pixel labels are integer IDs 0..3. The training code does NOT remap class IDs.

Important:
- The uploaded example mask contains exactly {0,1,2,3}.
- Background ID is configurable. By default background_index=-1 means:
  infer the background class from TRAIN masks by the class that dominates image borders.
- Loss supervises ALL 4 classes, including background. background_index is used only
  for reporting/saving foreground mean Dice/IoU.

Reproducible split
------------------
Before training starts, the program creates:
    split_dir/train_list.txt
    split_dir/val_list.txt
    split_dir/split_meta.json

If these files already exist, they are REUSED automatically.
Thus different models/test scripts can use exactly the same split.

Dependencies
------------
No pandas, scipy, sklearn, albumentations, torchvision.
Only Python stdlib + numpy + PIL + PyTorch.

Recommended project layout
--------------------------
project/
    train_cornea_SAR_MGEF.py
    model/
        UNet.py
        sar_mamba.py
        SIA.py
        MGEF.py

or UNet.py can be in the same directory as this training script.
"""

from __future__ import annotations

import os
import csv
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Current model
# =============================================================================

try:
    from model.UNet import SARMambaMGEFUNet, SARMambaMGEFConfig
except ImportError:
    from UNet import SARMambaMGEFUNet, SARMambaMGEFConfig


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Practical GPU behavior. Split reproducibility does NOT depend on this;
    # split generation uses its own local random.Random(seed).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Image/mask pairing
# =============================================================================

_SUPPORTED_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


@dataclass(frozen=True)
class PairRecord:
    image_name: str
    mask_name: str

    @property
    def stem(self) -> str:
        return os.path.splitext(self.image_name)[0]


def _files_by_stem(folder: str) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)

        if not os.path.isfile(full):
            continue

        stem, ext = os.path.splitext(name)

        if ext.lower() not in _SUPPORTED_EXTS:
            continue

        if stem in out:
            raise RuntimeError(
                f"Duplicate stem '{stem}' in {folder}: "
                f"{out[stem]} and {name}"
            )

        out[stem] = name

    return out


def list_paired_records(
    image_dir: str,
    mask_dir: str,
    strict: bool = True,
) -> List[PairRecord]:
    img_map = _files_by_stem(image_dir)
    mask_map = _files_by_stem(mask_dir)

    img_keys = set(img_map)
    mask_keys = set(mask_map)

    common = sorted(img_keys & mask_keys)
    only_img = sorted(img_keys - mask_keys)
    only_mask = sorted(mask_keys - img_keys)

    if not common:
        raise RuntimeError(
            "No image/mask pairs found.\n"
            f"image_dir={image_dir}\n"
            f"mask_dir ={mask_dir}\n"
            "Pairing rule: same filename stem."
        )

    if strict and (only_img or only_mask):
        raise RuntimeError(
            "Unpaired files detected.\n"
            f"Images without masks ({len(only_img)}): {only_img[:10]}\n"
            f"Masks without images ({len(only_mask)}): {only_mask[:10]}"
        )

    if only_img or only_mask:
        print(
            "[WARNING] Using paired intersection only: "
            f"unpaired_images={len(only_img)}, "
            f"unpaired_masks={len(only_mask)}"
        )

    return [
        PairRecord(
            image_name=img_map[k],
            mask_name=mask_map[k],
        )
        for k in common
    ]


# =============================================================================
# Persistent train/validation list generation
# =============================================================================

def _list_file_path(split_dir: str, split: str) -> str:
    return os.path.join(split_dir, f"{split}_list.txt")


def save_pair_list(
    path: str,
    records: Sequence[PairRecord],
) -> None:
    """
    Each line:
        image_filename<TAB>mask_filename

    Tab-separated names support image/mask files with different extensions.
    """
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        for r in records:
            f.write(f"{r.image_name}\t{r.mask_name}\n")


def load_pair_list(path: str) -> List[PairRecord]:
    records: List[PairRecord] = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise RuntimeError(
                    f"Invalid list format at {path}:{line_no}. "
                    "Expected: image_filename<TAB>mask_filename"
                )

            records.append(
                PairRecord(
                    image_name=parts[0],
                    mask_name=parts[1],
                )
            )

    if not records:
        raise RuntimeError(f"Empty split list: {path}")

    return records


def validate_split_records(
    image_dir: str,
    mask_dir: str,
    train_records: Sequence[PairRecord],
    val_records: Sequence[PairRecord],
) -> None:
    train_stems = {r.stem for r in train_records}
    val_stems = {r.stem for r in val_records}

    overlap = sorted(train_stems & val_stems)

    if overlap:
        raise RuntimeError(
            "Train/validation leakage in saved list files. "
            f"Examples: {overlap[:10]}"
        )

    for split_name, records in (
        ("train", train_records),
        ("val", val_records),
    ):
        for r in records:
            ipath = os.path.join(image_dir, r.image_name)
            mpath = os.path.join(mask_dir, r.mask_name)

            if not os.path.isfile(ipath):
                raise FileNotFoundError(
                    f"{split_name} image missing: {ipath}"
                )

            if not os.path.isfile(mpath):
                raise FileNotFoundError(
                    f"{split_name} mask missing: {mpath}"
                )


def prepare_or_load_split(
    all_records: Sequence[PairRecord],
    image_dir: str,
    mask_dir: str,
    split_dir: str,
    val_ratio: float,
    split_seed: int,
    force_resplit: bool,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    """
    Create the random split ONCE and persist it.

    Reproducibility policy:
    - If train_list.txt and val_list.txt both exist and force_resplit=False:
      reuse them exactly.
    - Otherwise regenerate using local random.Random(split_seed).
    """
    os.makedirs(split_dir, exist_ok=True)

    train_list_path = _list_file_path(split_dir, "train")
    val_list_path = _list_file_path(split_dir, "val")
    meta_path = os.path.join(split_dir, "split_meta.json")

    lists_exist = (
        os.path.isfile(train_list_path)
        and
        os.path.isfile(val_list_path)
    )

    if lists_exist and not force_resplit:
        train_records = load_pair_list(train_list_path)
        val_records = load_pair_list(val_list_path)

        validate_split_records(
            image_dir,
            mask_dir,
            train_records,
            val_records,
        )

        print("[SPLIT] Reusing existing split lists:")
        print("        ", train_list_path)
        print("        ", val_list_path)
        print(
            f"[SPLIT] train={len(train_records)}, "
            f"val={len(val_records)}"
        )

        return train_records, val_records

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0,1).")

    records = list(all_records)

    if len(records) < 2:
        raise RuntimeError(
            "At least two paired samples are required for train/val split."
        )

    rng = random.Random(int(split_seed))
    rng.shuffle(records)

    n_val = int(round(len(records) * float(val_ratio)))
    n_val = max(1, n_val)
    n_val = min(n_val, len(records) - 1)

    val_records = records[:n_val]
    train_records = records[n_val:]

    validate_split_records(
        image_dir,
        mask_dir,
        train_records,
        val_records,
    )

    save_pair_list(
        train_list_path,
        train_records,
    )
    save_pair_list(
        val_list_path,
        val_records,
    )

    meta = {
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "total_pairs": len(records),
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "image_dir": os.path.abspath(image_dir),
        "mask_dir": os.path.abspath(mask_dir),
        "train_list": os.path.abspath(train_list_path),
        "val_list": os.path.abspath(val_list_path),
    }

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[SPLIT] New split created and saved BEFORE training:")
    print("        ", train_list_path)
    print("        ", val_list_path)
    print("        ", meta_path)
    print(
        f"[SPLIT] seed={split_seed}, "
        f"train={len(train_records)}, "
        f"val={len(val_records)}"
    )

    return train_records, val_records


# =============================================================================
# Mask IO and validation
# =============================================================================

def load_mask_array(
    path: str,
    num_classes: int,
) -> np.ndarray:
    """
    Preserve integer label IDs exactly.

    IMPORTANT:
    We do NOT blindly call mask.convert("L") first, because a palette-mode mask
    could otherwise transform class indices through its color palette.
    """
    with Image.open(path) as m:
        arr = np.array(m)

    if arr.ndim == 3:
        # Accept RGB-like masks only when all channels contain the same IDs.
        if (
            arr.shape[2] >= 3
            and np.array_equal(arr[..., 0], arr[..., 1])
            and np.array_equal(arr[..., 0], arr[..., 2])
        ):
            arr = arr[..., 0]
        else:
            raise RuntimeError(
                "Color-coded mask detected, but this trainer expects "
                "integer class IDs 0..K-1. "
                f"Please provide indexed/grayscale labels: {path}"
            )

    if arr.ndim != 2:
        raise RuntimeError(
            f"Mask must be 2D after loading, got {arr.shape}: {path}"
        )

    arr = arr.astype(np.int64, copy=False)

    unique = np.unique(arr)

    invalid = unique[
        (unique < 0)
        | (unique >= num_classes)
    ]

    if len(invalid) > 0:
        raise RuntimeError(
            f"Invalid class IDs in mask {path}. "
            f"Found {unique.tolist()}, expected only "
            f"0..{num_classes - 1}."
        )

    return arr


def infer_background_index(
    mask_dir: str,
    train_records: Sequence[PairRecord],
    num_classes: int,
    max_masks: int = 128,
) -> Tuple[int, List[int]]:
    """
    Infer the background as the class that most frequently occurs on image borders.

    This is used only for "foreground mDice/mIoU" reporting/checkpoint selection.
    Loss still supervises ALL classes.
    """
    border_counts = np.zeros(
        (num_classes,),
        dtype=np.int64,
    )

    records = list(train_records)[:max_masks]

    for r in records:
        arr = load_mask_array(
            os.path.join(mask_dir, r.mask_name),
            num_classes=num_classes,
        )

        border = np.concatenate(
            [
                arr[0, :],
                arr[-1, :],
                arr[:, 0],
                arr[:, -1],
            ]
        )

        for c in range(num_classes):
            border_counts[c] += int(
                (border == c).sum()
            )

    bg = int(border_counts.argmax())

    return bg, border_counts.tolist()


def inspect_training_masks(
    mask_dir: str,
    train_records: Sequence[PairRecord],
    num_classes: int,
) -> List[int]:
    counts = np.zeros(
        (num_classes,),
        dtype=np.int64,
    )

    for i, r in enumerate(train_records):
        arr = load_mask_array(
            os.path.join(mask_dir, r.mask_name),
            num_classes=num_classes,
        )

        binc = np.bincount(
            arr.reshape(-1),
            minlength=num_classes,
        )
        counts += binc[:num_classes]

        if (i + 1) % 100 == 0:
            print(
                f"[MASK CHECK] {i + 1}/{len(train_records)}"
            )

    return counts.tolist()


# =============================================================================
# Augmentation
# =============================================================================

@dataclass
class AugCfg:
    train: bool = True

    # Pillow uses (width, height)
    base_resize: Tuple[int, int] = (256, 256)

    # Anatomically conservative geometry for anterior-segment OCT
    p_hflip: float = 0.5
    p_affine: float = 0.45
    rot_deg: float = 8.0
    scale_min: float = 0.95
    scale_max: float = 1.05
    trans_frac: float = 0.03
    shear_deg: float = 2.0

    # Image-only intensity augmentation
    p_brightness_contrast: float = 0.35
    brightness: float = 0.10
    contrast: float = 0.15

    p_gamma: float = 0.35
    gamma_min: float = 0.85
    gamma_max: float = 1.15

    p_noise: float = 0.15
    noise_std: float = 0.015

    in_ch: int = 1

    # Input in [0,1] -> roughly [-1,1]
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


def _gamma_adjust(
    img: Image.Image,
    gamma: float,
) -> Image.Image:
    arr = np.asarray(
        img,
        dtype=np.float32,
    ) / 255.0

    arr = np.power(
        np.clip(arr, 0.0, 1.0),
        gamma,
    )

    arr = np.clip(
        arr * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    return Image.fromarray(
        arr,
        mode=img.mode,
    )


def _add_gaussian_noise(
    img: Image.Image,
    std: float,
) -> Image.Image:
    arr = np.asarray(
        img,
        dtype=np.float32,
    ) / 255.0

    noise = np.random.normal(
        0.0,
        std,
        size=arr.shape,
    ).astype(np.float32)

    arr = np.clip(
        arr + noise,
        0.0,
        1.0,
    )

    arr = (
        arr * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        arr,
        mode=img.mode,
    )


def _random_affine(
    img: Image.Image,
    mask: Image.Image,
    cfg: AugCfg,
) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size

    angle = random.uniform(
        -cfg.rot_deg,
        cfg.rot_deg,
    )
    scale = random.uniform(
        cfg.scale_min,
        cfg.scale_max,
    )
    shear = math.radians(
        random.uniform(
            -cfg.shear_deg,
            cfg.shear_deg,
        )
    )

    dx = random.uniform(
        -cfg.trans_frac,
        cfg.trans_frac,
    ) * w

    dy = random.uniform(
        -cfg.trans_frac,
        cfg.trans_frac,
    ) * h

    theta = math.radians(angle)

    a = scale * math.cos(theta)
    b = -scale * math.sin(theta + shear)
    d = scale * math.sin(theta)
    e = scale * math.cos(theta + shear)

    cx, cy = w * 0.5, h * 0.5

    c = cx - a * cx - b * cy + dx
    f = cy - d * cx - e * cy + dy

    img = img.transform(
        (w, h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
        fillcolor=0,
    )

    mask = mask.transform(
        (w, h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.NEAREST,
        fillcolor=0,
    )

    return img, mask


class CorneaAugment:
    """
    Geometry is always synchronized between image and label.

    Deliberate differences from the old training script:
    - NO vertical flip: superior/inferior anatomy should not be inverted.
    - NO copy-paste: pasting complete cornea/chamber/lens anatomy from another
      eye can create anatomically inconsistent labels.
    - No random crop by default: preserve the complete anterior-segment anatomy.
    - Milder affine/intensity ranges.
    """

    def __init__(
        self,
        cfg: AugCfg,
    ) -> None:
        self.cfg = cfg

    def __call__(
        self,
        img: Image.Image,
        mask_arr: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        img = img.convert(
            "L" if cfg.in_ch == 1 else "RGB"
        )

        mask = Image.fromarray(
            mask_arr.astype(np.uint8),
            mode="L",
        )

        # Resize first; masks MUST use nearest interpolation.
        img = img.resize(
            cfg.base_resize,
            resample=Image.BILINEAR,
        )

        mask = mask.resize(
            cfg.base_resize,
            resample=Image.NEAREST,
        )

        if cfg.train:
            if random.random() < cfg.p_hflip:
                img = img.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
                mask = mask.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

            if random.random() < cfg.p_affine:
                img, mask = _random_affine(
                    img,
                    mask,
                    cfg,
                )

            if (
                random.random()
                < cfg.p_brightness_contrast
            ):
                if cfg.brightness > 0:
                    img = ImageEnhance.Brightness(
                        img
                    ).enhance(
                        1.0
                        + random.uniform(
                            -cfg.brightness,
                            cfg.brightness,
                        )
                    )

                if cfg.contrast > 0:
                    img = ImageEnhance.Contrast(
                        img
                    ).enhance(
                        1.0
                        + random.uniform(
                            -cfg.contrast,
                            cfg.contrast,
                        )
                    )

            if random.random() < cfg.p_gamma:
                img = _gamma_adjust(
                    img,
                    random.uniform(
                        cfg.gamma_min,
                        cfg.gamma_max,
                    ),
                )

            if random.random() < cfg.p_noise:
                img = _add_gaussian_noise(
                    img,
                    cfg.noise_std,
                )

        img_np = np.asarray(
            img,
            dtype=np.float32,
        ) / 255.0

        if cfg.in_ch == 1:
            if img_np.ndim == 3:
                img_np = img_np[..., 0]

            img_np = img_np[None, ...]

        else:
            if img_np.ndim == 2:
                img_np = np.repeat(
                    img_np[..., None],
                    3,
                    axis=2,
                )

            img_np = img_np.transpose(
                2, 0, 1
            )

        img_t = torch.from_numpy(
            np.ascontiguousarray(img_np)
        ).float()

        img_t = (
            img_t - cfg.normalize_mean
        ) / cfg.normalize_std

        mask_np = np.asarray(
            mask,
            dtype=np.int64,
        )

        mask_t = torch.from_numpy(
            np.ascontiguousarray(mask_np)
        ).long()

        return img_t, mask_t


# =============================================================================
# Dataset
# =============================================================================

class CorneaDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        records: Sequence[PairRecord],
        aug: CorneaAugment,
        num_classes: int,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.records = list(records)
        self.aug = aug
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.records[idx]

        ipath = os.path.join(
            self.image_dir,
            r.image_name,
        )
        mpath = os.path.join(
            self.mask_dir,
            r.mask_name,
        )

        with Image.open(ipath) as img0:
            img = img0.copy()

        mask_arr = load_mask_array(
            mpath,
            num_classes=self.num_classes,
        )

        if (
            img.size[0] != mask_arr.shape[1]
            or
            img.size[1] != mask_arr.shape[0]
        ):
            raise RuntimeError(
                "Image/mask spatial mismatch before resize:\n"
                f"image={ipath}, size={img.size}\n"
                f"mask ={mpath}, shape={mask_arr.shape}"
            )

        return self.aug(
            img,
            mask_arr,
        )


# =============================================================================
# Loss / metrics
# =============================================================================

class MultiClassDiceLoss(nn.Module):
    """
    Soft Dice over ALL classes.

    Unlike the old script, class 0 is NOT automatically ignored.
    This avoids silently assuming which ID represents background.
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        target_oh = F.one_hot(
            target.long(),
            num_classes=self.num_classes,
        ).permute(
            0, 3, 1, 2
        ).float()

        dims = (0, 2, 3)

        inter = (
            probs * target_oh
        ).sum(dims)

        denom = (
            probs + target_oh
        ).sum(dims)

        dice = (
            2.0 * inter + self.eps
        ) / (
            denom + self.eps
        )

        return 1.0 - dice.mean()


@torch.no_grad()
def confusion_matrix_update(
    confusion: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    valid = (
        (target >= 0)
        & (target < num_classes)
    )

    indices = (
        target[valid] * num_classes
        + pred[valid]
    )

    bincount = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    )

    confusion += bincount.reshape(
        num_classes,
        num_classes,
    ).to(
        confusion.device
    )


def metrics_from_confusion(
    confusion: torch.Tensor,
    background_index: int,
    eps: float = 1e-7,
):
    cm = confusion.double()

    tp = torch.diag(cm)
    pred_sum = cm.sum(dim=0)
    gt_sum = cm.sum(dim=1)

    dice = (
        2.0 * tp + eps
    ) / (
        pred_sum + gt_sum + eps
    )

    iou = (
        tp + eps
    ) / (
        pred_sum
        + gt_sum
        - tp
        + eps
    )

    keep = [
        c
        for c in range(cm.shape[0])
        if c != background_index
    ]

    mdice_fg = float(
        dice[keep].mean().item()
    )
    miou_fg = float(
        iou[keep].mean().item()
    )

    return (
        dice.cpu().tolist(),
        iou.cpu().tolist(),
        mdice_fg,
        miou_fg,
    )


def estimate_ce_weights(
    pixel_counts: Sequence[int],
) -> torch.Tensor:
    """
    Mild inverse-sqrt frequency weighting.
    Less extreme than direct inverse frequency.
    """
    counts = np.asarray(
        pixel_counts,
        dtype=np.float64,
    )

    freq = counts / max(
        float(counts.sum()),
        1.0,
    )

    weights = 1.0 / np.sqrt(
        np.clip(
            freq,
            1e-8,
            1.0,
        )
    )

    weights /= weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# =============================================================================
# LR schedule
# =============================================================================

class WarmupCosine:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 5e-6,
    ) -> None:
        self.opt = optimizer
        self.warmup_epochs = max(
            0,
            int(warmup_epochs),
        )
        self.max_epochs = int(max_epochs)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)

    def step(
        self,
        epoch: int,
    ) -> float:
        if (
            self.warmup_epochs > 0
            and epoch < self.warmup_epochs
        ):
            lr = (
                self.base_lr
                * float(epoch + 1)
                / float(self.warmup_epochs)
            )
        else:
            t = (
                float(
                    epoch - self.warmup_epochs
                )
                / float(
                    max(
                        1,
                        self.max_epochs
                        - self.warmup_epochs,
                    )
                )
            )

            t = min(
                max(t, 0.0),
                1.0,
            )

            lr = (
                self.min_lr
                + 0.5
                * (
                    self.base_lr
                    - self.min_lr
                )
                * (
                    1.0
                    + math.cos(
                        math.pi * t
                    )
                )
            )

        for pg in self.opt.param_groups:
            pg["lr"] = lr

        return lr


# =============================================================================
# Model
# =============================================================================

def build_model(
    args,
) -> SARMambaMGEFUNet:
    cfg = SARMambaMGEFConfig(
        in_ch=args.in_ch,
        num_classes=4,
        channels=(
            32,
            64,
            128,
            256,
            512,
        ),
        d_state=args.d_state,
        expand=args.expand,
        dt_rank=args.dt_rank,
        token_dim=args.token_dim,
        kmax=args.kmax,
        token_heads=args.token_heads,
        token_pool_hw=(
            args.token_pool,
            args.token_pool,
        ),
        route_topk=args.route_topk,
        route_mode=args.route_mode,
        sparse_route_train=False,
        sparse_route_eval=True,
        decoder_refine_blocks=args.decoder_refine_blocks,
    )

    model = SARMambaMGEFUNet(cfg)

    # Preserve SAR/SIA/MGEF intentional initialization.
    # Initialize only the final segmentation head.
    if isinstance(model.head, nn.Conv2d):
        nn.init.kaiming_normal_(
            model.head.weight,
            mode="fan_in",
            nonlinearity="linear",
        )

        if model.head.bias is not None:
            nn.init.zeros_(
                model.head.bias
            )

    return model


# =============================================================================
# Train / validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    background_index: int,
    scaler,
    amp: bool,
    grad_clip: float,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )

    for img, mask in loader:
        img = img.to(
            device,
            non_blocking=True,
        )

        mask = mask.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=amp
        ):
            logits = model(img)

        logits_fp32 = logits.float()

        loss = criterion(
            logits_fp32,
            mask,
        )

        if not torch.isfinite(loss):
            print(
                "[WARNING] Non-finite loss; "
                "batch skipped."
            )
            continue

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(
                optimizer
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(
            optimizer
        )
        scaler.update()

        pred = logits_fp32.argmax(
            dim=1
        )

        confusion_matrix_update(
            confusion,
            pred,
            mask,
            num_classes,
        )

        bs = img.size(0)

        total_loss += (
            float(loss.item())
            * bs
        )

        total_samples += bs

    (
        dice_pc,
        iou_pc,
        mdice_fg,
        miou_fg,
    ) = metrics_from_confusion(
        confusion,
        background_index=background_index,
    )

    return (
        total_loss
        / max(total_samples, 1),
        dice_pc,
        iou_pc,
        mdice_fg,
        miou_fg,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device,
    num_classes: int,
    background_index: int,
    amp: bool,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )

    for img, mask in loader:
        img = img.to(
            device,
            non_blocking=True,
        )
        mask = mask.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=amp
        ):
            logits = model(img)

        logits_fp32 = logits.float()

        loss = criterion(
            logits_fp32,
            mask,
        )

        pred = logits_fp32.argmax(
            dim=1
        )

        confusion_matrix_update(
            confusion,
            pred,
            mask,
            num_classes,
        )

        bs = img.size(0)

        total_loss += (
            float(loss.item())
            * bs
        )
        total_samples += bs

    (
        dice_pc,
        iou_pc,
        mdice_fg,
        miou_fg,
    ) = metrics_from_confusion(
        confusion,
        background_index=background_index,
    )

    return (
        total_loss
        / max(total_samples, 1),
        dice_pc,
        iou_pc,
        mdice_fg,
        miou_fg,
    )


# =============================================================================
# Unique checkpoint / CSV logging
# =============================================================================

def unique_path(
    folder: str,
    filename: str,
) -> str:
    base = Path(folder) / filename

    if not base.exists():
        return str(base)

    version = 2

    while True:
        p = base.with_name(
            f"{base.stem}_v{version}"
            f"{base.suffix}"
        )

        if not p.exists():
            return str(p)

        version += 1


def save_checkpoint_unique(
    save_dir: str,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    best_mdice: float,
    background_index: int,
    args,
) -> str:
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    path = unique_path(
        save_dir,
        filename,
    )

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_mdice_fg": float(best_mdice),
            "background_index": int(background_index),
            "num_classes": 4,
            "args": vars(args),
        },
        path,
    )

    return path


def write_history_csv(
    path: str,
    rows: Sequence[Dict],
) -> None:
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_mdice_fg",
        "train_miou_fg",
        "val_loss",
        "val_mdice_fg",
        "val_miou_fg",
        "val_dice_c0",
        "val_dice_c1",
        "val_dice_c2",
        "val_dice_c3",
        "val_iou_c0",
        "val_iou_c1",
        "val_iou_c2",
        "val_iou_c3",
        "best_val_mdice_fg",
        "epoch_seconds",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full-supervised 4-class corneal segmentation "
            "with SAR-Mamba + SIA + MGEF U-Net"
        )
    )

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--image_dir",
        type=str,
        default=r"",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=r"",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"",
    )

    # Split lists
    parser.add_argument(
        "--split_dir",
        type=str,
        default=r"",
        help=(
            "Where train_list.txt/val_list.txt are stored. "
            "Default: <out_dir>/split"
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--force_resplit",
        action="store_true",
        help="Regenerate and overwrite the saved split lists.",
    )
    parser.add_argument(
        "--allow_unpaired",
        action="store_true",
    )

    # ------------------------------------------------------------------
    # Task / input
    # ------------------------------------------------------------------
    parser.add_argument(
        "--in_ch",
        type=int,
        default=1,
        choices=[1, 3],
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
    )

    # -1 = infer from training-mask border pixels.
    parser.add_argument(
        "--background_index",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2, 3],
        help=(
            "-1 = infer from train-mask border pixels. "
            "Set explicitly if your label convention is known."
        ),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="0 is the safest default on Windows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
    )

    # Loss
    parser.add_argument(
        "--dice_w",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
    )

    # Optional periodic checkpoints
    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help="0 disables periodic checkpoints.",
    )

    # ------------------------------------------------------------------
    # Mamba / model
    # ------------------------------------------------------------------
    parser.add_argument(
        "--d_state",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--dt_rank",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--token_dim",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--token_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--token_pool",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--route_topk",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--route_mode",
        type=str,
        default="topk_st",
        choices=[
            "soft",
            "topk_st",
            "topk_hard",
        ],
    )
    parser.add_argument(
        "--decoder_refine_blocks",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    num_classes = 4

    set_seed(args.seed)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    split_dir = (
        args.split_dir
        if args.split_dir
        else os.path.join(
            args.out_dir,
            "split",
        )
    )

    # ------------------------------------------------------------------
    # Pair all data, then create/reuse split lists BEFORE training.
    # ------------------------------------------------------------------
    all_records = list_paired_records(
        args.image_dir,
        args.mask_dir,
        strict=not args.allow_unpaired,
    )

    train_records, val_records = prepare_or_load_split(
        all_records=all_records,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        split_dir=split_dir,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        force_resplit=args.force_resplit,
    )

    # ------------------------------------------------------------------
    # Validate class IDs on the TRAIN split before training.
    # ------------------------------------------------------------------
    pixel_counts = inspect_training_masks(
        args.mask_dir,
        train_records,
        num_classes=num_classes,
    )

    pixel_total = max(
        sum(pixel_counts),
        1,
    )

    print("[MASK CHECK] Train pixel distribution:")
    for c, count in enumerate(pixel_counts):
        print(
            f"  class {c}: "
            f"{count} "
            f"({100.0 * count / pixel_total:.3f}%)"
        )

    if args.background_index == -1:
        background_index, border_counts = infer_background_index(
            args.mask_dir,
            train_records,
            num_classes=num_classes,
        )

        print(
            "[MASK CHECK] Auto background inference "
            f"from border counts={border_counts} "
            f"-> background_index={background_index}"
        )
    else:
        background_index = int(
            args.background_index
        )

        print(
            "[MASK CHECK] User-specified "
            f"background_index={background_index}"
        )

    # Persist the inferred/selected task metadata beside split files.
    task_meta = {
        "num_classes": num_classes,
        "background_index": background_index,
        "train_pixel_counts": pixel_counts,
        "image_size": args.image_size,
        "in_ch": args.in_ch,
    }

    with open(
        os.path.join(
            split_dir,
            "task_meta.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            task_meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Dataset / augmentation
    # ------------------------------------------------------------------
    train_aug = CorneaAugment(
        AugCfg(
            train=True,
            base_resize=(
                args.image_size,
                args.image_size,
            ),
            in_ch=args.in_ch,
        )
    )

    val_aug = CorneaAugment(
        AugCfg(
            train=False,
            base_resize=(
                args.image_size,
                args.image_size,
            ),
            in_ch=args.in_ch,
        )
    )

    train_ds = CorneaDataset(
        args.image_dir,
        args.mask_dir,
        train_records,
        train_aug,
        num_classes=num_classes,
    )

    val_ds = CorneaDataset(
        args.image_dir,
        args.mask_dir,
        val_records,
        val_aug,
        num_classes=num_classes,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    # Model / objective
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = bool(
        args.amp
        and torch.cuda.is_available()
    )

    model = build_model(
        args
    ).to(device)

    ce_weight = None

    if args.use_class_weights:
        ce_weight = estimate_ce_weights(
            pixel_counts
        ).to(device)

        print(
            "[LOSS] CE class weights:",
            ce_weight.detach().cpu().tolist(),
        )

    ce = nn.CrossEntropyLoss(
        weight=ce_weight
    )

    dice_loss = MultiClassDiceLoss(
        num_classes=num_classes
    )

    def criterion(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            ce(
                logits,
                target,
            )
            + args.dice_w
            * dice_loss(
                logits,
                target,
            )
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = WarmupCosine(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    print("=" * 96)
    print("4-class Corneal Segmentation | FULL supervised")
    print("=" * 96)
    print("Device                 :", device)
    if torch.cuda.is_available():
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Total paired samples   :", len(all_records))
    print("Train samples          :", len(train_ds))
    print("Validation samples     :", len(val_ds))
    print("Train list             :", _list_file_path(split_dir, "train"))
    print("Validation list        :", _list_file_path(split_dir, "val"))
    print("Split seed             :", args.split_seed)
    print("Image size             :", args.image_size)
    print("Input channels         :", args.in_ch)
    print("Output classes         :", num_classes)
    print("Background index       :", background_index)
    print("Loss                   : CE + {:.3f} * Dice(all classes)".format(args.dice_w))
    print("Class-weighted CE      :", bool(args.use_class_weights))
    print("Augmentation           : HFlip + mild affine + brightness/contrast + gamma + noise")
    print("Vertical flip          : disabled")
    print("Copy-Paste             : disabled")
    print("AMP                    :", use_amp)
    print("Checkpoint policy      : save NEW file on each foreground-mDice improvement")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_mdice = -1.0
    best_epoch = -1

    history: List[Dict] = []

    history_path = os.path.join(
        args.out_dir,
        "training_history.csv",
    )

    for epoch in range(args.epochs):
        lr = scheduler.step(
            epoch
        )

        t0 = time.time()

        (
            train_loss,
            train_dice_pc,
            train_iou_pc,
            train_mdice_fg,
            train_miou_fg,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            background_index=background_index,
            scaler=scaler,
            amp=use_amp,
            grad_clip=args.grad_clip,
        )

        (
            val_loss,
            val_dice_pc,
            val_iou_pc,
            val_mdice_fg,
            val_miou_fg,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            background_index=background_index,
            amp=use_amp,
        )

        epoch_seconds = (
            time.time()
            - t0
        )

        dice_text = " ".join(
            [
                f"C{c}={val_dice_pc[c]:.4f}"
                for c in range(num_classes)
            ]
        )

        print(
            f"Ep {epoch + 1:03d}/{args.epochs} | "
            f"lr={lr:.3e} | "
            f"train_loss={train_loss:.4f} | "
            f"train_mDiceFG={train_mdice_fg:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mDiceFG={val_mdice_fg:.4f} | "
            f"val_mIoUFG={val_miou_fg:.4f} | "
            f"{dice_text} | "
            f"time={epoch_seconds:.1f}s"
        )

        # --------------------------------------------------------------
        # Every improvement gets a NEW checkpoint. No overwrite.
        # --------------------------------------------------------------
        if val_mdice_fg > best_mdice:
            best_mdice = val_mdice_fg
            best_epoch = epoch + 1

            filename = (
                f"SARMambaMGEFUNet_cornea4"
                f"_ep{epoch + 1:03d}"
                f"_mDiceFG{val_mdice_fg:.5f}"
                f"_mIoUFG{val_miou_fg:.5f}.pt"
            )

            path = save_checkpoint_unique(
                save_dir=args.out_dir,
                filename=filename,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_mdice=best_mdice,
                background_index=background_index,
                args=args,
            )

            print(
                "  [SAVE][NEW BEST]",
                path,
            )

        if (
            args.save_freq > 0
            and (epoch + 1) % args.save_freq == 0
        ):
            filename = (
                f"SARMambaMGEFUNet_cornea4"
                f"_periodic_ep{epoch + 1:03d}"
                f"_mDiceFG{val_mdice_fg:.5f}.pt"
            )

            path = save_checkpoint_unique(
                save_dir=args.out_dir,
                filename=filename,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_mdice=best_mdice,
                background_index=background_index,
                args=args,
            )

            print(
                "  [SAVE][PERIODIC]",
                path,
            )

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_mdice_fg": train_mdice_fg,
            "train_miou_fg": train_miou_fg,
            "val_loss": val_loss,
            "val_mdice_fg": val_mdice_fg,
            "val_miou_fg": val_miou_fg,
            "best_val_mdice_fg": best_mdice,
            "epoch_seconds": epoch_seconds,
        }

        for c in range(num_classes):
            row[f"val_dice_c{c}"] = val_dice_pc[c]
            row[f"val_iou_c{c}"] = val_iou_pc[c]

        history.append(
            row
        )

        write_history_csv(
            history_path,
            history,
        )

    print("=" * 96)
    print("Training finished.")
    print("Best epoch        :", best_epoch)
    print("Best val mDice FG :", best_mdice)
    print("Train list        :", _list_file_path(split_dir, "train"))
    print("Val list          :", _list_file_path(split_dir, "val"))
    print("History           :", history_path)
    print("=" * 96)


if __name__ == "__main__":
    main()
