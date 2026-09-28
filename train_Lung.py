# -*- coding: utf-8 -*-
"""
Supervised training for SARMambaMGEFUNet on the CSIRO Lung Segmentation Data Kit
================================================================================

Target data:
    Rusak F, Wang D, Arzhaeva Y.
    Lung Segmentation Data Kit. v1. CSIRO.
    DOI: 10.25919/5c49548be055

Design goals:
- Adapt the current SAR-Mamba + SIA + MGEF U-Net to 1-channel CXR input
  and 1-channel binary lung-mask output.
- No pandas / sklearn / albumentations / torchvision / scipy.
- Reuse only the dependency style already present in the old Lung script:
  numpy, PIL, tifffile, OpenCV, PyTorch.
- Generates persistent train_list.txt / val_list.txt before training and reuses them for exact reproducibility.
- Resize to 256x256 by default.
- Compute ONE normalization range from training images only, then apply the
  same range to both train and validation images.
- Mild paired geometric augmentation + mild image-only intensity augmentation.
- No Cutout that edits the ground-truth lung mask.
- Dice or BCE+Dice loss.
- Adam + cosine annealing by default, matching the old Lung training strategy.
- AMP supported; GradScaler is created once for the complete training run.
- Save a NEW .pt file whenever validation Dice improves; never overwrite.
- Optional periodic snapshots also use unique filenames.
- CSV logs use Python's built-in csv module only.

Expected project layout:
    project/
        train_lung_SAR_MGEF.py
        UNet.py
        sar_mamba.py
        SIA.py
        MGEF.py

If UNet.py is inside model/, change the import below to:
    from model.UNet import SARMambaMGEFUNet, SARMambaMGEFConfig
"""

from __future__ import annotations

import os
import glob
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tifffile as tiff
import cv2

# -------------------------------------------------------------------------
# Current network
# Prefer the project layout used by the current model:
#     model/UNet.py
#     model/sar_mamba.py
#     model/SIA.py
#     model/MGEF.py
# Fallback keeps direct-script layouts compatible.
# -------------------------------------------------------------------------
try:
    from model.UNet import SARMambaMGEFUNet, SARMambaMGEFConfig
except ImportError:
    from UNet import SARMambaMGEFUNet, SARMambaMGEFConfig


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep the same practical speed-oriented behavior as the old script.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# File IO
# =============================================================================

_SUPPORTED_EXTS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")


def _list_image_files(folder: str) -> List[str]:
    files: List[str] = []
    for ext in _SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def _read_raw(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path)
    else:
        arr = np.array(Image.open(path))

    return np.asarray(arr)


def _to_grayscale_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image array to float32 grayscale WITHOUT per-image normalization.
    """
    if arr.ndim == 3:
        # Use only the first 3 channels if alpha is present.
        arr = arr[..., :3]

        # cv2 expects RGB here because PIL/tifffile arrays are treated as RGB.
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image after conversion, got {arr.shape}")

    return arr.astype(np.float32)


def read_image_raw(path: str) -> np.ndarray:
    return _to_grayscale_float(_read_raw(path))


def read_mask_binary(path: str) -> np.ndarray:
    """
    Read a mask and convert it robustly to {0,1}.

    Threshold is based on half of the observed positive range instead of
    hard-coding 127, so 8-bit and 16-bit masks are both handled.
    """
    arr = _read_raw(path)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {arr.shape} from {path}")

    arr = arr.astype(np.float32)

    vmax = float(arr.max())
    vmin = float(arr.min())

    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)

    threshold = vmin + 0.5 * (vmax - vmin)
    return (arr > threshold).astype(np.float32)


# =============================================================================
# Pairing
# =============================================================================

def pair_images_masks(
    images_dir: str,
    masks_dir: str,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Pair images and masks by identical filename stem.

    strict=True avoids silently dropping unpaired data, which is safer for a
    small medical dataset.
    """
    imgs = _list_image_files(images_dir)
    msks = _list_image_files(masks_dir)

    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {images_dir}")

    if len(msks) == 0:
        raise RuntimeError(f"No masks found in: {masks_dir}")

    img_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in imgs
    }
    msk_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in msks
    }

    img_keys = set(img_map.keys())
    msk_keys = set(msk_map.keys())
    common = sorted(img_keys & msk_keys)

    only_img = sorted(img_keys - msk_keys)
    only_msk = sorted(msk_keys - img_keys)

    if len(common) == 0:
        raise RuntimeError(
            "No paired files found.\n"
            f"images_dir={images_dir}\n"
            f"masks_dir ={masks_dir}\n"
            "Pairing rule: identical basename/stem."
        )

    if strict and (only_img or only_msk):
        preview_img = only_img[:10]
        preview_msk = only_msk[:10]
        raise RuntimeError(
            "Unpaired Lung files were found.\n"
            f"Images without masks ({len(only_img)}): {preview_img}\n"
            f"Masks without images ({len(only_msk)}): {preview_msk}\n"
            "Fix pairing before training, or use --allow_unpaired 1 "
            "to train on the intersection only."
        )

    if only_img or only_msk:
        print(
            "[WARNING] Training on paired intersection only. "
            f"unpaired_images={len(only_img)}, unpaired_masks={len(only_msk)}"
        )

    return (
        [img_map[k] for k in common],
        [msk_map[k] for k in common],
    )


# =============================================================================
# Train / validation split helpers
# =============================================================================

def _canonical_path(path: str) -> str:
    """
    Canonical path comparison for Windows/Linux.
    On Windows, normcase also removes case-sensitivity differences.
    """
    return os.path.normcase(
        os.path.realpath(
            os.path.abspath(path)
        )
    )


def _same_data_folders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
) -> bool:
    return (
        _canonical_path(train_img_dir) == _canonical_path(val_img_dir)
        and
        _canonical_path(train_mask_dir) == _canonical_path(val_mask_dir)
    )


def infer_lung_source(path: str) -> str:
    """
    Infer the original source dataset from common Lung Data Kit filenames.

    JSRT commonly uses:
        JPCLNxxx
        JPCNNxxx

    Montgomery commonly uses:
        MCUCXR_xxxx_x

    Unknown names are kept in their own group instead of being discarded.
    """
    stem = os.path.splitext(
        os.path.basename(path)
    )[0].upper()

    if stem.startswith("JPCLN") or stem.startswith("JPCNN"):
        return "JSRT"

    if stem.startswith("MCUCXR"):
        return "Montgomery"

    return "Other"


def stratified_train_val_split(
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Deterministic source-aware split.

    The CSIRO Lung Segmentation Data Kit combines JSRT and Montgomery.
    When the user supplies ONE common Images folder and ONE common Masks folder,
    this function creates a train/validation split while approximately
    preserving the source-dataset proportions.

    No sklearn dependency is used.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError("Images and masks must have equal length.")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1.")

    groups: Dict[str, List[Tuple[str, str]]] = {}

    for img, msk in zip(image_paths, mask_paths):
        source = infer_lung_source(img)
        groups.setdefault(source, []).append((img, msk))

    rng = random.Random(int(seed))

    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []

    print("=" * 88)
    print("Automatic train/validation split")
    print(f"val_ratio={val_ratio:.3f}, split_seed={seed}")
    print("-" * 88)

    for source in sorted(groups.keys()):
        pairs = list(groups[source])
        rng.shuffle(pairs)

        n_total = len(pairs)

        if n_total <= 1:
            n_val = 0
        else:
            n_val = int(round(n_total * val_ratio))
            n_val = max(1, n_val)
            n_val = min(n_val, n_total - 1)

        val_part = pairs[:n_val]
        train_part = pairs[n_val:]

        val_pairs.extend(val_part)
        train_pairs.extend(train_part)

        print(
            f"{source:12s}: total={n_total:3d} "
            f"train={len(train_part):3d} val={len(val_part):3d}"
        )

    # Shuffle final sets so the source groups are not contiguous.
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise RuntimeError(
            "Automatic split produced an empty train or validation set. "
            "Check the dataset size and --val_ratio."
        )

    tr_imgs = [p[0] for p in train_pairs]
    tr_msks = [p[1] for p in train_pairs]
    va_imgs = [p[0] for p in val_pairs]
    va_msks = [p[1] for p in val_pairs]

    # Final hard safety check.
    train_keys = {
        os.path.splitext(os.path.basename(p))[0]
        for p in tr_imgs
    }
    val_keys = {
        os.path.splitext(os.path.basename(p))[0]
        for p in va_imgs
    }

    overlap = train_keys & val_keys

    if overlap:
        raise RuntimeError(
            "Internal split error: train/validation overlap remains. "
            f"Examples: {sorted(overlap)[:10]}"
        )

    print("-" * 88)
    print(
        f"TOTAL        : total={len(image_paths):3d} "
        f"train={len(tr_imgs):3d} val={len(va_imgs):3d}"
    )
    print("=" * 88)

    return tr_imgs, tr_msks, va_imgs, va_msks


def _split_list_paths(split_dir: str) -> Tuple[str, str, str]:
    """
    Return paths for persistent split artifacts.
    """
    return (
        os.path.join(split_dir, "train_list.txt"),
        os.path.join(split_dir, "val_list.txt"),
        os.path.join(split_dir, "split_meta.json"),
    )


def save_split_list(
    path: str,
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
) -> None:
    """
    Save one image/mask pair per line:

        absolute_image_path<TAB>absolute_mask_path<TAB>source

    Absolute paths make later model-comparison/test scripts reproduce exactly
    the same samples without having to guess the original folder layout.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError("image_paths and mask_paths must have equal length.")

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        for img, msk in zip(image_paths, mask_paths):
            f.write(
                f"{os.path.abspath(img)}\t"
                f"{os.path.abspath(msk)}\t"
                f"{infer_lung_source(img)}\n"
            )


def load_split_list(
    path: str,
) -> Tuple[List[str], List[str]]:
    """
    Load train_list.txt / val_list.txt generated by save_split_list().
    """
    image_paths: List[str] = []
    mask_paths: List[str] = []

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

            if len(parts) < 2:
                raise RuntimeError(
                    f"Invalid split list format at {path}:{line_no}. "
                    "Expected image_path<TAB>mask_path[<TAB>source]."
                )

            img = parts[0]
            msk = parts[1]

            if not os.path.isfile(img):
                raise FileNotFoundError(
                    f"Image recorded in split list does not exist:\n{img}"
                )

            if not os.path.isfile(msk):
                raise FileNotFoundError(
                    f"Mask recorded in split list does not exist:\n{msk}"
                )

            image_paths.append(img)
            mask_paths.append(msk)

    if len(image_paths) == 0:
        raise RuntimeError(f"Split list is empty: {path}")

    return image_paths, mask_paths


def _check_split_overlap(
    tr_imgs: Sequence[str],
    va_imgs: Sequence[str],
) -> None:
    """
    Ensure the same case stem never appears in both train and validation.
    """
    train_keys = {
        os.path.splitext(os.path.basename(p))[0]
        for p in tr_imgs
    }
    val_keys = {
        os.path.splitext(os.path.basename(p))[0]
        for p in va_imgs
    }

    overlap = sorted(train_keys & val_keys)

    if overlap:
        raise RuntimeError(
            "Train/validation leakage detected in split lists. "
            f"Examples: {overlap[:10]}"
        )


def save_split_artifacts(
    split_dir: str,
    tr_imgs: Sequence[str],
    tr_msks: Sequence[str],
    va_imgs: Sequence[str],
    va_msks: Sequence[str],
    split_seed: int,
    val_ratio: float,
    mode: str,
    args,
) -> Tuple[str, str, str]:
    """
    Persist the exact split so all later model-comparison/test runs can reuse it.
    """
    os.makedirs(split_dir, exist_ok=True)

    train_list_path, val_list_path, meta_path = _split_list_paths(split_dir)

    save_split_list(
        train_list_path,
        tr_imgs,
        tr_msks,
    )
    save_split_list(
        val_list_path,
        va_imgs,
        va_msks,
    )

    meta = {
        "mode": mode,
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "train_count": len(tr_imgs),
        "val_count": len(va_imgs),
        "train_img_dir": os.path.abspath(args.train_img_dir),
        "train_mask_dir": os.path.abspath(args.train_mask_dir),
        "val_img_dir": os.path.abspath(args.val_img_dir),
        "val_mask_dir": os.path.abspath(args.val_mask_dir),
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

    return train_list_path, val_list_path, meta_path


def prepare_or_load_split_lists(
    args,
    strict_pairing: bool,
) -> Tuple[List[str], List[str], List[str], List[str], str, str, str]:
    """
    Core reproducible split workflow.

    1) If train_list.txt and val_list.txt already exist and --force_resplit=0:
       reuse them exactly.
    2) Otherwise:
       - if train/val directories are the same complete Lung dataset:
         perform the existing source-aware random split (JSRT/Montgomery).
       - if explicit train/val directories are different:
         use those directories as the split.
    3) Save train_list.txt / val_list.txt / split_meta.json BEFORE training.
    """
    split_dir = args.split_dir

    if not split_dir:
        split_dir = os.path.join(
            args.save_dir,
            "split",
        )

    os.makedirs(split_dir, exist_ok=True)

    train_list_path, val_list_path, meta_path = _split_list_paths(split_dir)

    lists_exist = (
        os.path.isfile(train_list_path)
        and os.path.isfile(val_list_path)
    )

    if lists_exist and not bool(args.force_resplit):
        print("[SPLIT] Existing list files found. Reusing them exactly.")

        tr_imgs, tr_msks = load_split_list(train_list_path)
        va_imgs, va_msks = load_split_list(val_list_path)

        _check_split_overlap(
            tr_imgs,
            va_imgs,
        )

        print(f"[SPLIT] train={len(tr_imgs)}, val={len(va_imgs)}")
        print("[SPLIT] train list:", train_list_path)
        print("[SPLIT] val list  :", val_list_path)

        return (
            tr_imgs,
            tr_msks,
            va_imgs,
            va_msks,
            train_list_path,
            val_list_path,
            meta_path,
        )

    same_folders = _same_data_folders(
        args.train_img_dir,
        args.train_mask_dir,
        args.val_img_dir,
        args.val_mask_dir,
    )

    split_seed = (
        args.seed
        if args.split_seed is None
        else args.split_seed
    )

    if same_folders:
        print(
            "[SPLIT] Train and validation folders point to the same complete "
            "dataset. Creating a deterministic source-aware random split."
        )

        all_imgs, all_msks = pair_images_masks(
            args.train_img_dir,
            args.train_mask_dir,
            strict=strict_pairing,
        )

        tr_imgs, tr_msks, va_imgs, va_msks = stratified_train_val_split(
            all_imgs,
            all_msks,
            val_ratio=args.val_ratio,
            seed=split_seed,
        )

        split_mode = "auto_source_aware_random_split"

    else:
        print(
            "[SPLIT] Separate train/validation directories supplied. "
            "Saving those exact sets into persistent list files."
        )

        tr_imgs, tr_msks = pair_images_masks(
            args.train_img_dir,
            args.train_mask_dir,
            strict=strict_pairing,
        )

        va_imgs, va_msks = pair_images_masks(
            args.val_img_dir,
            args.val_mask_dir,
            strict=strict_pairing,
        )

        _check_split_overlap(
            tr_imgs,
            va_imgs,
        )

        split_mode = "explicit_directories"

    (
        train_list_path,
        val_list_path,
        meta_path,
    ) = save_split_artifacts(
        split_dir=split_dir,
        tr_imgs=tr_imgs,
        tr_msks=tr_msks,
        va_imgs=va_imgs,
        va_msks=va_msks,
        split_seed=split_seed,
        val_ratio=args.val_ratio,
        mode=split_mode,
        args=args,
    )

    print("[SPLIT] Split artifacts saved BEFORE training:")
    print("        train:", train_list_path)
    print("        val  :", val_list_path)
    print("        meta :", meta_path)

    return (
        tr_imgs,
        tr_msks,
        va_imgs,
        va_msks,
        train_list_path,
        val_list_path,
        meta_path,
    )


# =============================================================================
# Training-set-only global intensity normalization
# =============================================================================

def estimate_training_intensity_range(
    image_paths: Sequence[str],
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    sample_step: int = 32,
) -> Tuple[float, float]:
    """
    Estimate one global intensity range using TRAINING images only.

    Why not the old per-image 1-99 percentile normalization?
    ---------------------------------------------------------
    The CSIRO kit already harmonizes the constituent datasets. A separate
    percentile stretch for every image discards part of that common intensity
    calibration. Here we estimate one range from the training set and apply
    that exact range to train and validation images.

    To avoid loading all 2048x2048 images into RAM at once, a deterministic
    spatial subsample is collected from each image.
    """
    if len(image_paths) == 0:
        raise ValueError("No training images supplied for intensity statistics.")

    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("Invalid percentile range.")

    sample_step = max(int(sample_step), 1)

    chunks: List[np.ndarray] = []

    print(
        f"Estimating training intensity range from {len(image_paths)} images "
        f"(p{low_percentile:g}-p{high_percentile:g}, step={sample_step}) ..."
    )

    for i, path in enumerate(image_paths, start=1):
        img = read_image_raw(path)
        sample = img[::sample_step, ::sample_step].reshape(-1)
        chunks.append(sample.astype(np.float32, copy=False))

        if i % 50 == 0 or i == len(image_paths):
            print(f"  stats scan: {i}/{len(image_paths)}")

    values = np.concatenate(chunks, axis=0)

    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Safe fallback.
        lo = float(values.min())
        hi = float(values.max())

    if hi <= lo:
        raise RuntimeError(
            f"Could not estimate a valid intensity range: lo={lo}, hi={hi}"
        )

    print(f"Training intensity range: lo={lo:.6f}, hi={hi:.6f}")
    return lo, hi


def normalize_image(
    img: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    img = (img.astype(np.float32) - float(lo)) / (float(hi) - float(lo))
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


# =============================================================================
# Paired augmentation
# =============================================================================

class LungAugment:
    """
    CXR augmentation designed for lung-field segmentation.

    Important design choices:
    - Resize FIRST to the network resolution.
    - Image uses INTER_AREA for downsampling; mask uses INTER_NEAREST.
    - Mild affine transform: rotation, scaling, translation.
    - Horizontal flip is allowed.
    - No vertical flip.
    - Gamma / brightness-contrast / blur affect image only.
    - No Cutout that erases part of the ground-truth lung mask.
    """

    def __init__(
        self,
        img_size: int = 256,
        p_affine: float = 0.7,
        p_hflip: float = 0.5,
        p_gamma: float = 0.25,
        p_bc: float = 0.20,
        p_blur: float = 0.10,
        rotate_limit: float = 10.0,
        scale_limit: float = 0.10,
        shift_limit: float = 0.05,
    ) -> None:
        self.img_size = int(img_size)

        self.p_affine = float(p_affine)
        self.p_hflip = float(p_hflip)
        self.p_gamma = float(p_gamma)
        self.p_bc = float(p_bc)
        self.p_blur = float(p_blur)

        self.rotate_limit = float(rotate_limit)
        self.scale_limit = float(scale_limit)
        self.shift_limit = float(shift_limit)

    def _resize(
        self,
        img: np.ndarray,
        msk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img_r = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        msk_r = cv2.resize(
            msk,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return img_r, msk_r

    def _affine(
        self,
        img: np.ndarray,
        msk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape

        angle = random.uniform(
            -self.rotate_limit,
            self.rotate_limit,
        )
        scale = random.uniform(
            1.0 - self.scale_limit,
            1.0 + self.scale_limit,
        )
        tx = random.uniform(
            -self.shift_limit,
            self.shift_limit,
        ) * w
        ty = random.uniform(
            -self.shift_limit,
            self.shift_limit,
        ) * h

        matrix = cv2.getRotationMatrix2D(
            (w / 2.0, h / 2.0),
            angle,
            scale,
        )
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        img_w = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        msk_w = cv2.warpAffine(
            msk,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        return img_w, msk_w

    @staticmethod
    def _hflip(
        img: np.ndarray,
        msk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.ascontiguousarray(img[:, ::-1]),
            np.ascontiguousarray(msk[:, ::-1]),
        )

    @staticmethod
    def _gamma(img: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.85, 1.15)
        out = np.power(
            np.clip(img, 0.0, 1.0),
            gamma,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _brightness_contrast(img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.90, 1.10)
        beta = random.uniform(-0.04, 0.04)
        out = alpha * img + beta
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _blur(img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            img,
            (3, 3),
            sigmaX=0,
        ).astype(np.float32)

    def __call__(
        self,
        img: np.ndarray,
        msk: np.ndarray,
        train: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resize first: the source Data Kit is already standardized spatially,
        # and running every affine at 2048x2048 wastes CPU time.
        img, msk = self._resize(img, msk)

        if train and random.random() < self.p_affine:
            img, msk = self._affine(img, msk)

        if train and random.random() < self.p_hflip:
            img, msk = self._hflip(img, msk)

        if train and random.random() < self.p_gamma:
            img = self._gamma(img)

        if train and random.random() < self.p_bc:
            img = self._brightness_contrast(img)

        if train and random.random() < self.p_blur:
            img = self._blur(img)

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        msk = (msk > 0.5).astype(np.float32)

        return img, msk


# =============================================================================
# Dataset
# =============================================================================

class LungSegDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        augmenter: LungAugment,
        train: bool,
        norm_lo: float,
        norm_hi: float,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have equal length.")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.augmenter = augmenter
        self.train = bool(train)
        self.norm_lo = float(norm_lo)
        self.norm_hi = float(norm_hi)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = read_image_raw(self.image_paths[idx])
        msk = read_mask_binary(self.mask_paths[idx])

        if img.shape != msk.shape:
            raise RuntimeError(
                "Image/mask shape mismatch before preprocessing:\n"
                f"image={self.image_paths[idx]} shape={img.shape}\n"
                f"mask ={self.mask_paths[idx]} shape={msk.shape}"
            )

        img = normalize_image(
            img,
            lo=self.norm_lo,
            hi=self.norm_hi,
        )

        img, msk = self.augmenter(
            img,
            msk,
            train=self.train,
        )

        # [1,H,W]
        img = img[None, ...]
        msk = msk[None, ...]

        return (
            torch.from_numpy(img),
            torch.from_numpy(msk),
        )


# =============================================================================
# Losses / metrics
# =============================================================================

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (
            2.0 * inter + self.smooth
        ) / (
            denom + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Optional more strongly supervised binary objective.
    Default training keeps pure Dice for baseline comparability.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


@torch.no_grad()
def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - inter

    dice = (
        2.0 * inter + eps
    ) / (
        pred_sum + target_sum + eps
    )

    iou = (
        inter + eps
    ) / (
        union + eps
    )

    return (
        float(dice.mean().item()),
        float(iou.mean().item()),
    )


# =============================================================================
# Model
# =============================================================================

def build_model(args) -> Tuple[SARMambaMGEFUNet, SARMambaMGEFConfig]:
    """
    Lung segmentation:
        input  = [B,1,H,W]
        output = [B,1,H,W] logits
    """
    cfg = SARMambaMGEFConfig(
        in_ch=1,
        num_classes=1,
        channels=(
            args.c1,
            args.c2,
            args.c3,
            args.c4,
            args.cb,
        ),
        d_state=args.d_state,
        expand=args.expand,
        dt_rank=args.dt_rank,
        token_dim=args.token_dim,
        kmax=args.kmax,
        token_heads=args.token_heads,
        token_pool_hw=(args.token_pool, args.token_pool),
        route_topk=args.route_topk,
        route_mode=args.route_mode,
        sparse_route_train=False,
        sparse_route_eval=True,
        decoder_refine_blocks=args.decoder_refine_blocks,
    )

    model = SARMambaMGEFUNet(cfg)

    # Do not globally re-initialize the model. SAR/MGEF contains intentional
    # zero-initialized conditioning layers. Only initialize the final head.
    if isinstance(model.head, nn.Conv2d):
        nn.init.kaiming_normal_(
            model.head.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    return model, cfg


# =============================================================================
# Training / evaluation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler,
    amp: bool,
    grad_clip: float,
) -> Tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(
            device,
            non_blocking=True,
        )
        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(imgs)

        # Binary losses in FP32 for numerical stability.
        logits_fp32 = logits.float()
        masks_fp32 = masks.float()
        loss = loss_fn(
            logits_fp32,
            masks_fp32,
        )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        dice, iou = binary_metrics(
            logits_fp32.detach(),
            masks_fp32,
        )

        bs = imgs.size(0)

        total_loss += float(loss.item()) * bs
        total_dice += dice * bs
        total_iou += iou * bs
        n += bs

    n = max(n, 1)

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(
            device,
            non_blocking=True,
        )
        masks = masks.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(imgs)

        logits_fp32 = logits.float()
        masks_fp32 = masks.float()

        loss = loss_fn(
            logits_fp32,
            masks_fp32,
        )

        dice, iou = binary_metrics(
            logits_fp32,
            masks_fp32,
        )

        bs = imgs.size(0)

        total_loss += float(loss.item()) * bs
        total_dice += dice * bs
        total_iou += iou * bs
        n += bs

    n = max(n, 1)

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
    )


# =============================================================================
# Saving / logging
# =============================================================================

def unique_path(
    folder: str,
    filename: str,
) -> str:
    """
    Never overwrite an existing checkpoint.
    """
    base = Path(folder) / filename

    if not base.exists():
        return str(base)

    version = 2

    while True:
        candidate = base.with_name(
            f"{base.stem}_v{version}{base.suffix}"
        )
        if not candidate.exists():
            return str(candidate)
        version += 1


def save_checkpoint_unique(
    save_dir: str,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_dice: float,
    cfg,
    args,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    path = unique_path(
        save_dir,
        filename,
    )

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": float(best_val_dice),
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
            "args": vars(args),
        },
        path,
    )

    return path


def write_history_csv(
    csv_path: str,
    rows: Sequence[Dict],
) -> None:
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
        "best_val_dice",
        "epoch_seconds",
    ]

    with open(
        csv_path,
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
        description="Train SAR-Mamba+SIA+MGEF U-Net on CSIRO Lung Segmentation Data Kit"
    )

    # Data folders: preserve the explicit train/validation directory workflow
    # from the old Lung training script.
    parser.add_argument("--train_img_dir", type=str, default=r"")
    parser.add_argument("--train_mask_dir", type=str, default=r"")
    parser.add_argument("--val_img_dir", type=str, default=r"")
    parser.add_argument("--val_mask_dir", type=str, default=r"")

    # Data
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--allow_unpaired", type=int, default=0, choices=[0, 1])

    # If train and validation paths point to the SAME complete dataset,
    # automatically create a deterministic source-aware split.
    parser.add_argument(
        "--auto_split_if_same",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = auto split when train/val folders are identical.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Validation fraction used only by automatic splitting.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for automatic split; default uses --seed.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=r"",
        help=(
            "Directory used to save/reuse train_list.txt and val_list.txt. "
            "Default: <save_dir>/split"
        ),
    )
    parser.add_argument(
        "--force_resplit",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 = reuse existing list files if present; "
            "1 = regenerate split lists."
        ),
    )

    # Normalization estimated from TRAIN only.
    parser.add_argument("--norm_low_pct", type=float, default=0.5)
    parser.add_argument("--norm_high_pct", type=float, default=99.5)
    parser.add_argument("--norm_sample_step", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--loss",
        type=str,
        default="dice",
        choices=["dice", "bce_dice"],
        help="dice keeps closest comparability with the old Lung script.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP.",
    )

    # Save
    parser.add_argument("--save_dir", type=str, default=r"")
    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help="Optional periodic checkpoint every N epochs; 0 disables.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience; 0 disables early stopping.",
    )

    # Model
    parser.add_argument("--c1", type=int, default=32)
    parser.add_argument("--c2", type=int, default=64)
    parser.add_argument("--c3", type=int, default=128)
    parser.add_argument("--c4", type=int, default=256)
    parser.add_argument("--cb", type=int, default=512)

    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=float, default=1.5)
    parser.add_argument("--dt_rank", type=int, default=4)

    parser.add_argument("--token_dim", type=int, default=96)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_pool", type=int, default=8)

    parser.add_argument("--route_topk", type=int, default=2)
    parser.add_argument(
        "--route_mode",
        type=str,
        default="topk_st",
        choices=["soft", "topk_st", "topk_hard"],
    )
    parser.add_argument("--decoder_refine_blocks", type=int, default=2)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = bool(
        args.amp and torch.cuda.is_available()
    )

    # ---------------------------------------------------------------------
    # Pair data / create-or-reuse persistent train/validation list files
    # ---------------------------------------------------------------------
    strict_pairing = not bool(args.allow_unpaired)

    (
        tr_imgs,
        tr_msks,
        va_imgs,
        va_msks,
        train_list_path,
        val_list_path,
        split_meta_path,
    ) = prepare_or_load_split_lists(
        args=args,
        strict_pairing=strict_pairing,
    )

    # ---------------------------------------------------------------------
    # One training-derived normalization for BOTH sets
    # ---------------------------------------------------------------------
    norm_lo, norm_hi = estimate_training_intensity_range(
        tr_imgs,
        low_percentile=args.norm_low_pct,
        high_percentile=args.norm_high_pct,
        sample_step=args.norm_sample_step,
    )

    augmenter = LungAugment(
        img_size=args.img_size,
    )

    train_ds = LungSegDataset(
        tr_imgs,
        tr_msks,
        augmenter=augmenter,
        train=True,
        norm_lo=norm_lo,
        norm_hi=norm_hi,
    )

    val_ds = LungSegDataset(
        va_imgs,
        va_msks,
        augmenter=augmenter,
        train=False,
        norm_lo=norm_lo,
        norm_hi=norm_hi,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model, cfg = build_model(args)
    model = model.to(device)

    if args.loss == "dice":
        loss_fn = SoftDiceLoss()
    else:
        loss_fn = BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
        )

    # Default keeps the old Lung optimizer family while adding optional
    # weight_decay=0.0 for exact-like behavior.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    # Important fix vs old script:
    # GradScaler is created ONCE, not reset every epoch.
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    print("=" * 88)
    print("SAR-Mamba + SIA + MGEF Lung Segmentation")
    print("=" * 88)
    print("Device                 :", device)
    if torch.cuda.is_available():
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Train pairs            :", len(train_ds))
    print("Validation pairs       :", len(val_ds))
    print("Train list             :", train_list_path)
    print("Validation list        :", val_list_path)
    print("Split metadata         :", split_meta_path)
    print("Input size             :", args.img_size)
    print("Input/output channels  : 1 -> 1")
    print("Normalization          : training-global percentile")
    print("Normalization range    :", (norm_lo, norm_hi))
    print("Augmentation           : mild affine + HFlip + gamma/BC/blur")
    print("Cutout                 : disabled")
    print("Loss                   :", args.loss)
    print("Optimizer              : Adam")
    print("LR                     :", args.lr)
    print("Cosine min LR          :", args.min_lr)
    print("AMP                    :", use_amp)
    print("Batch size             :", args.batch_size)
    print("Epochs                 :", args.epochs)
    print("Checkpoint overwrite   : NEVER")
    print("=" * 88)

    best_dice = -1.0
    best_epoch = -1
    no_improve = 0

    history: List[Dict] = []
    history_path = os.path.join(
        args.save_dir,
        "training_history.csv",
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp=use_amp,
            grad_clip=args.grad_clip,
        )

        va_loss, va_dice, va_iou = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp=use_amp,
        )

        scheduler.step()

        epoch_seconds = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.3e} "
            f"train_loss={tr_loss:.4f} "
            f"train_dice={tr_dice:.4f} "
            f"train_iou={tr_iou:.4f} "
            f"val_loss={va_loss:.4f} "
            f"val_dice={va_dice:.4f} "
            f"val_iou={va_iou:.4f} "
            f"time={epoch_seconds:.1f}s"
        )

        improved = va_dice > best_dice

        if improved:
            best_dice = va_dice
            best_epoch = epoch
            no_improve = 0

            filename = (
                f"SARMambaMGEFUNet"
                f"_epoch{epoch:03d}"
                f"_valDice{va_dice:.5f}"
                f"_valIoU{va_iou:.5f}.pt"
            )

            ckpt_path = save_checkpoint_unique(
                save_dir=args.save_dir,
                filename=filename,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_dice=best_dice,
                cfg=cfg,
                args=args,
            )

            print(
                f"  [SAVE][NEW BEST] {ckpt_path}"
            )
        else:
            no_improve += 1

        # Optional periodic snapshot.
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            filename = (
                f"SARMambaMGEFUNet"
                f"_periodic_epoch{epoch:03d}"
                f"_valDice{va_dice:.5f}.pt"
            )

            periodic_path = save_checkpoint_unique(
                save_dir=args.save_dir,
                filename=filename,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_dice=best_dice,
                cfg=cfg,
                args=args,
            )

            print(
                f"  [SAVE][PERIODIC] {periodic_path}"
            )

        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": tr_loss,
                "train_dice": tr_dice,
                "train_iou": tr_iou,
                "val_loss": va_loss,
                "val_dice": va_dice,
                "val_iou": va_iou,
                "best_val_dice": best_dice,
                "epoch_seconds": epoch_seconds,
            }
        )

        # Standard-library CSV only. Updated every epoch.
        write_history_csv(
            history_path,
            history,
        )

        if (
            args.patience > 0
            and no_improve >= args.patience
        ):
            print(
                "Early stopping. "
                f"Best epoch={best_epoch}, "
                f"best val Dice={best_dice:.5f}"
            )
            break

    print("=" * 88)
    print(
        f"Done. Best epoch={best_epoch}, "
        f"best val Dice={best_dice:.5f}"
    )
    print("History:", history_path)
    print("Train list:", train_list_path)
    print("Val list  :", val_list_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
