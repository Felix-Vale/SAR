from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))
sys.path.append(os.path.dirname(os.path.dirname(_THIS_DIR)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Use exactly the same ISIC dataset implementation as the training script.
from data.dataset_ours_isic import ISICDataset
from wheels.torch_utils import seed_torch

# Same model as train_ISIC_full_supervised_amp_safe.py
from UNet2 import SARMambaMGEFUNet, SARMambaMGEFConfig

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError as exc:
    raise ImportError(
        "HD95 calculation requires scipy. Install it with: pip install scipy"
    ) from exc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SARMambaMGEFUNet on the ISIC validation split"
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"",
        help="Same ISIC root directory used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"",
        help="Path to the .pt model state_dict saved by the training script.",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--foreground_class", type=int, default=1)

    # Model arguments: defaults are intentionally identical to training.
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--channels", type=int, nargs=5, default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=float, default=1.5)
    parser.add_argument("--dt_rank", type=int, default=4)
    parser.add_argument("--token_dim", type=int, default=96)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--route_topk", type=int, default=2)
    parser.add_argument(
        "--route_mode",
        type=str,
        default="topk_st",
        choices=["soft", "topk_st", "topk_hard"],
    )
    parser.add_argument("--decoder_refine_blocks", type=int, default=2)

    parser.add_argument(
        "--output_csv",
        type=str,
        default="isic_val_metrics_per_case.csv",
        help="CSV file containing per-case metrics.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="isic_val_metrics_summary.csv",
        help="CSV file containing mean/std metrics.",
    )

    return parser.parse_args()


def make_val_loader(args) -> DataLoader:
    """Build validation loader exactly like the training script."""
    val_set = ISICDataset(
        image_path=args.data_path,
        stage="test",
        image_size=args.image_size,
        is_augmentation=False,
    )

    kwargs: Dict = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(
        dataset=val_set,
        shuffle=False,
        **kwargs,
    )

    print("Val samples :", len(val_set))
    print("Val batches :", len(val_loader))
    return val_loader


def build_model(args) -> SARMambaMGEFUNet:
    """Same model configuration as train_ISIC_full_supervised_amp_safe.py."""
    cfg = SARMambaMGEFConfig(
        in_ch=args.in_channels,
        num_classes=args.num_classes,
        channels=tuple(args.channels),
        d_state=args.d_state,
        expand=args.expand,
        dt_rank=args.dt_rank,
        token_dim=args.token_dim,
        kmax=args.kmax,
        token_heads=args.token_heads,
        route_topk=args.route_topk,
        route_mode=args.route_mode,
        decoder_refine_blocks=args.decoder_refine_blocks,
    )

    model = SARMambaMGEFUNet(cfg)

    # Kept for consistency with training construction. Loaded weights overwrite it.
    if isinstance(model.head, nn.Conv2d):
        nn.init.kaiming_normal_(
            model.head.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    return model


def normalize_label(label: torch.Tensor) -> torch.Tensor:
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)
    return label.long()


def load_weights(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    obj = torch.load(checkpoint_path, map_location="cpu")

    # Training script saves model.state_dict() directly. The extra branches below
    # only make this evaluator tolerant of other common checkpoint wrappers.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    else:
        state_dict = obj

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    # Handle DataParallel/DDP checkpoints if necessary.
    if len(state_dict) > 0 and all(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    print("Loaded checkpoint:", checkpoint_path)


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance between binary mask surfaces.

    Unit: pixels on the resized validation mask (e.g. 128x128 when image_size=128).

    Empty-mask policy:
      - both empty: 0
      - only one empty: image diagonal, a finite worst-case penalty
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    pred_any = bool(pred.any())
    target_any = bool(target.any())

    if not pred_any and not target_any:
        return 0.0

    if not pred_any or not target_any:
        h, w = pred.shape[-2:]
        return float(np.hypot(max(h - 1, 0), max(w - 1, 0)))

    # 8-connected erosion gives a 1-pixel-wide object surface.
    structure = np.ones((3, 3), dtype=bool)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ binary_erosion(target, structure=structure, border_value=0)

    # distance_transform_edt(~surface)[q] gives distance from q to nearest surface pixel.
    dt_to_target = distance_transform_edt(~target_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)

    d_pred_to_target = dt_to_target[pred_surface]
    d_target_to_pred = dt_to_pred[target_surface]

    distances = np.concatenate([d_pred_to_target, d_target_to_pred], axis=0)
    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))


def binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Foreground per-case Dice, IoU, Precision, Sensitivity and HD95."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, np.logical_not(target)).sum())
    fn = float(np.logical_and(np.logical_not(pred), target).sum())

    pred_sum = tp + fp
    target_sum = tp + fn
    union = tp + fp + fn

    # These definitions preserve the training script's eps behavior for Dice/IoU.
    dice = (2.0 * tp + eps) / (pred_sum + target_sum + eps)
    iou = (tp + eps) / (union + eps)

    # Empty denominator conventions:
    # precision=1 when there is no predicted foreground and no FP;
    # sensitivity=1 when GT has no foreground. ISIC lesion GT is normally non-empty.
    precision = (tp + eps) / (tp + fp + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "hd95": hd95_binary(pred, target),
    }


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args):
    model.eval()
    use_amp = bool(args.use_amp) and torch.cuda.is_available()

    rows: List[Dict] = []
    sample_index = 0

    for batch_idx, batch in enumerate(loader):
        if len(batch) < 2:
            raise RuntimeError("Validation batch must contain image and label.")

        # Exactly the same validation-view strategy as the training script:
        # batch[0] = base image, batch[1] = label.
        image = batch[0].to(device, non_blocking=True)
        label = batch[1].to(device, non_blocking=True)
        target = normalize_label(label)

        with autocast(enabled=use_amp):
            logits = model(image)

        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits found at validation batch {batch_idx}.")

        pred = torch.argmax(logits, dim=1)

        pred_fg = (pred == args.foreground_class).detach().cpu().numpy()
        target_fg = (target == args.foreground_class).detach().cpu().numpy()

        batch_size = pred_fg.shape[0]
        for i in range(batch_size):
            metrics = binary_metrics(pred_fg[i], target_fg[i])
            rows.append(
                {
                    "index": sample_index,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "sensitivity": metrics["sensitivity"],
                    "hd95": metrics["hd95"],
                }
            )
            sample_index += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"Evaluated batch {batch_idx + 1}/{len(loader)}")

    if not rows:
        raise RuntimeError("Validation loader produced no samples.")

    keys = ["dice", "iou", "precision", "sensitivity", "hd95"]
    values = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}

    summary = {
        "num_cases": len(rows),
        "dice_mean": float(values["dice"].mean()),
        "dice_std": float(values["dice"].std(ddof=0)),
        "iou_mean": float(values["iou"].mean()),
        "iou_std": float(values["iou"].std(ddof=0)),
        "precision_mean": float(values["precision"].mean()),
        "precision_std": float(values["precision"].std(ddof=0)),
        "sensitivity_mean": float(values["sensitivity"].mean()),
        "sensitivity_std": float(values["sensitivity"].std(ddof=0)),
        "hd95_mean": float(values["hd95"].mean()),
        "hd95_std": float(values["hd95"].std(ddof=0)),
    }

    return rows, summary


def main():
    args = get_args()
    seed_torch(args.seed)

    print("=" * 88)
    print("ISIC validation evaluation")
    print("=" * 88)
    print("Device            :", device)
    if torch.cuda.is_available():
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Data path         :", args.data_path)
    print("Image size        :", args.image_size)
    print("Foreground class  :", args.foreground_class)
    print("AMP               :", bool(args.use_amp) and torch.cuda.is_available())
    print("HD95 unit         : resized-image pixels")
    print("=" * 88)

    val_loader = make_val_loader(args)

    model = build_model(args).to(device)
    load_weights(model, args.checkpoint)

    per_case_rows, summary = evaluate(model, val_loader, args)

    write_csv(
        args.output_csv,
        per_case_rows,
        ["index", "dice", "iou", "precision", "sensitivity", "hd95"],
    )

    write_csv(
        args.summary_csv,
        [summary],
        [
            "num_cases",
            "dice_mean", "dice_std",
            "iou_mean", "iou_std",
            "precision_mean", "precision_std",
            "sensitivity_mean", "sensitivity_std",
            "hd95_mean", "hd95_std",
        ],
    )

    print("\n" + "=" * 88)
    print("FINAL ISIC VAL RESULTS (foreground class = {})".format(args.foreground_class))
    print("=" * 88)
    print("Cases : {}".format(summary["num_cases"]))
    print("Dice  : {:.6f} ± {:.6f}".format(summary["dice_mean"], summary["dice_std"]))
    print("IoU   : {:.6f} ± {:.6f}".format(summary["iou_mean"], summary["iou_std"]))
    print("Pre   : {:.6f} ± {:.6f}".format(summary["precision_mean"], summary["precision_std"]))
    print("Sen   : {:.6f} ± {:.6f}".format(summary["sensitivity_mean"], summary["sensitivity_std"]))
    print("HD95  : {:.6f} ± {:.6f} pixels".format(summary["hd95_mean"], summary["hd95_std"]))
    print("=" * 88)


if __name__ == "__main__":
    main()
