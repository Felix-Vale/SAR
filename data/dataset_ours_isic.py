# -*- coding: utf-8 -*-
"""
ISIC Dataset for Ours (R-I Utility Mix)
=======================================

This dataset preserves the RTX-4060-Ti-friendly fair protocol already used
for the re-trained AdaMix baseline:

    original image
        -> deterministic Resize to image_size x image_size
        -> training geometric augmentation
        -> strong photometric augmentation
        -> ToTensor + ImageNet Normalize

The ONLY Ours-specific dataset extension is:
    reference_mode=True

Reference mode:
    - uses the labeled training subset only;
    - no random flip;
    - no random rotation;
    - no ColorJitter / AutoContrast / Equalize / GaussianBlur;
    - deterministic Resize only;
    - returns (image, GT label).

AdaMix and Ours must use the same normal training preprocessing for a fair
comparison.
"""

from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)

from data import transforms as T


class ISICDataset(Dataset):
    VALID_EXTS = {".png", ".jpg", ".jpeg"}

    MASK_DIR_HINTS = (
        "mask",
        "masks",
        "segmentation",
        "segmentations",
        "groundtruth",
        "ground_truth",
        "ground-truth",
        "gt",
    )

    MASK_SUFFIXES = (
        "_segmentation",
        "_mask",
        "_gt",
    )

    # Avoid recursively scanning the whole ISIC tree repeatedly in the
    # same main Python process.
    _INDEX_CACHE = {}

    def __init__(
        self,
        image_path="",
        stage="train",
        image_size=256,
        is_augmentation=False,
        labeled=True,
        percentage=0.1,
        validate_files=True,
        reference_mode=False,
    ):
        super().__init__()

        self.image_path = str(image_path)
        self.root = Path(image_path)
        self.image_size = int(image_size)
        self.stage = stage
        self.is_augmentation = bool(is_augmentation)
        self.reference_mode = bool(reference_mode)

        if not self.root.exists():
            raise FileNotFoundError(
                f"ISIC data root does not exist: {self.root}"
            )

        # Reference is defined only on the labeled TRAIN subset.
        if self.reference_mode and self.stage != "train":
            raise ValueError(
                "reference_mode=True is only valid for stage='train'."
            )

        # --------------------------------------------------------
        # List file
        # --------------------------------------------------------
        if self.stage == "train":
            list_path = self.root / "train.list"
        elif self.stage == "val":
            list_path = self.root / "val.list"
        else:
            list_path = self.root / "test.list"

        if not list_path.is_file():
            raise FileNotFoundError(
                f"List file not found: {list_path}"
            )

        with list_path.open("r", encoding="utf-8") as f:
            sample_list = [
                self._normalize_case_id(line.strip())
                for line in f
                if line.strip()
            ]

        # --------------------------------------------------------
        # Keep the SAME labeled/unlabeled split as AdaMix:
        # first percentage of train.list = labeled.
        # --------------------------------------------------------
        if self.stage == "train":
            split_index = int(len(sample_list) * percentage)

            if split_index <= 0:
                raise RuntimeError(
                    f"Labeled split is empty: total={len(sample_list)}, "
                    f"percentage={percentage}"
                )

            if labeled:
                self.sample_list = sample_list[:split_index]
            else:
                self.sample_list = sample_list[split_index:]
        else:
            self.sample_list = sample_list

        if len(self.sample_list) == 0:
            raise RuntimeError(
                f"No samples found: stage={stage}, "
                f"labeled={labeled}, percentage={percentage}"
            )

        # --------------------------------------------------------
        # Recursive image/mask index
        # --------------------------------------------------------
        cache_key = str(self.root.resolve()).lower()

        if cache_key not in self._INDEX_CACHE:
            self._INDEX_CACHE[cache_key] = self._build_file_index()

        self.image_index, self.mask_index = self._INDEX_CACHE[cache_key]

        # --------------------------------------------------------
        # FAIR SPEED PROTOCOL: Resize FIRST.
        # --------------------------------------------------------
        self.image_resize = Resize(
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        self.label_resize = Resize(
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        self.pre_transform = self.build_pre_transform()
        self.post_transform = self.build_post_transform()

        if self.is_augmentation and not self.reference_mode:
            self.augmentation = self.build_augmentation_transform()

        if validate_files:
            self._validate_dataset_files(max_report=30)

    # ============================================================
    # File discovery
    # ============================================================

    @classmethod
    def _normalize_case_id(cls, entry):
        name = Path(entry).name
        lower = name.lower()

        for ext in (".jpeg", ".jpg", ".png"):
            if lower.endswith(ext):
                name = name[:-len(ext)]
                break

        lower_name = name.lower()

        for suffix in cls.MASK_SUFFIXES:
            if lower_name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        return name

    @classmethod
    def _looks_like_mask(cls, path):
        stem_lower = path.stem.lower()

        if any(stem_lower.endswith(s) for s in cls.MASK_SUFFIXES):
            return True

        parent_parts = [p.lower() for p in path.parts[:-1]]

        for part in parent_parts:
            normalized = (
                part.replace(" ", "")
                .replace("-", "")
                .replace("_", "")
            )

            for hint in cls.MASK_DIR_HINTS:
                hint_norm = (
                    hint.replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                )

                if hint_norm and hint_norm in normalized:
                    return True

        return False

    def _build_file_index(self):
        image_index = {}
        mask_index = {}

        all_files = []

        for path in self.root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in self.VALID_EXTS
                and path.name.lower().startswith("isic_")
            ):
                all_files.append(path)

        if len(all_files) == 0:
            raise FileNotFoundError(
                "\nNo ISIC .png/.jpg/.jpeg files were found under:\n"
                f"{self.root}"
            )

        # Masks first.
        for path in all_files:
            if self._looks_like_mask(path):
                case = self._normalize_case_id(path.name)
                mask_index.setdefault(case, path)

        # Images.
        for path in all_files:
            if not self._looks_like_mask(path):
                case = self._normalize_case_id(path.name)
                image_index.setdefault(case, path)

        print("=" * 80)
        print("ISIC recursive file index (cached)")
        print("Data root      :", self.root)
        print("Image files    :", len(image_index))
        print("Mask files     :", len(mask_index))
        print("=" * 80)

        return image_index, mask_index

    def _validate_dataset_files(self, max_report=30):
        missing_images = [
            case
            for case in self.sample_list
            if case not in self.image_index
        ]

        missing_masks = [
            case
            for case in self.sample_list
            if case not in self.mask_index
        ]

        mode_name = (
            "reference"
            if self.reference_mode
            else self.stage
        )

        print(
            f"Dataset mode={mode_name}, samples={len(self.sample_list)}, "
            f"missing_images={len(missing_images)}, "
            f"missing_masks={len(missing_masks)}"
        )

        if missing_images or missing_masks:
            lines = [
                "",
                "ISIC list/data mismatch.",
                f"Data root: {self.root}",
                f"Mode: {mode_name}",
                f"Samples: {len(self.sample_list)}",
                f"Missing images: {len(missing_images)}",
                f"Missing masks: {len(missing_masks)}",
            ]

            if missing_images:
                lines.append(
                    "First missing image IDs: "
                    + ", ".join(missing_images[:max_report])
                )

            if missing_masks:
                lines.append(
                    "First missing mask IDs: "
                    + ", ".join(missing_masks[:max_report])
                )

            raise FileNotFoundError("\n".join(lines))

    # ============================================================
    # Data loading
    # ============================================================

    def __getitem__(self, item):
        case = self.sample_list[item]

        image_path = self.image_index.get(case)
        label_path = self.mask_index.get(case)

        if image_path is None:
            raise FileNotFoundError(
                f"Image not indexed for {case}"
            )

        if label_path is None:
            raise FileNotFoundError(
                f"Mask not indexed for {case}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(label_path) as lab:
            label = lab.convert("L")

        # Binary lesion mask: any positive value -> 1.
        label_np = np.asarray(label)
        label_np = (label_np > 0).astype(np.uint8)
        label = Image.fromarray(label_np, mode="L")

        # --------------------------------------------------------
        # First resize the original high-resolution ISIC image.
        # This is the SAME fair-speed protocol as the re-trained
        # AdaMix baseline.
        # --------------------------------------------------------
        image = self.image_resize(image)
        label = self.label_resize(label)

        # --------------------------------------------------------
        # Ours Reference mode:
        # deterministic labeled data only.
        # --------------------------------------------------------
        if self.reference_mode:
            image = self.post_transform(image)

            label = torch.from_numpy(
                np.asarray(label, dtype=np.uint8).copy()
            ).unsqueeze(0)

            return image, label

        # --------------------------------------------------------
        # Normal training mode:
        # EXACT SAME preprocessing/augmentation policy that should
        # also be used by the re-trained AdaMix baseline.
        # --------------------------------------------------------
        if self.stage == "train":
            image, label = self.pre_transform(image, label)

            imageA1 = deepcopy(image)
            imageA2 = deepcopy(image)

            if self.is_augmentation:
                # Preserve original AdaMix behavior:
                # A2 is generated by augmenting A1 again.
                imageA1, _ = self.augmentation(imageA1, label)
                imageA2, _ = self.augmentation(imageA1, label)

            image = self.post_transform(image)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)

            label = torch.from_numpy(
                np.asarray(label, dtype=np.uint8).copy()
            ).unsqueeze(0)

            return image, label, imageA1, imageA2

        # --------------------------------------------------------
        # Validation / test
        # --------------------------------------------------------
        image = self.post_transform(image)

        label = torch.from_numpy(
            np.asarray(label, dtype=np.uint8).copy()
        ).unsqueeze(0)

        return image, label

    def __len__(self):
        return len(self.sample_list)

    # ============================================================
    # Augmentations
    # ============================================================

    @staticmethod
    def build_augmentation_transform():
        return T.Compose([
            T.ColorJitter(0.5, 0.5, 0.5, 0.05),
            T.RandomPosterize(bits=5, p=0.2),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    @staticmethod
    def build_pre_transform():
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    @staticmethod
    def build_post_transform():
        # Resize has already been done.
        return Compose([
            ToTensor(),
            Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])
