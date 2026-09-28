from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception as exc:
    raise ImportError(
        "This test script needs scipy for HD95. Install it with: pip install scipy"
    ) from exc


# =============================================================================
# Basic helpers
# =============================================================================

def seed_everything(seed: int = 1) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label: torch.Tensor) -> torch.Tensor:
    """
    Convert common segmentation-label layouts to [B,H,W] integer labels.
    """
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)

    # One-hot / multi-channel label fallback.
    elif label.dim() == 4 and label.shape[1] > 1:
        label = torch.argmax(label, dim=1)

    return label.long()


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Accept both normal state_dict and DataParallel/DDP state_dict.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Pure model.state_dict()
    if isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt

    # Common full-checkpoint formats
    elif isinstance(ckpt, dict):
        candidate_keys = (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
        )
        state_dict = None
        for key in candidate_keys:
            value = ckpt.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

        if state_dict is None:
            raise RuntimeError(
                "Could not find model weights in checkpoint. "
                f"Available keys: {list(ckpt.keys())[:30]}"
            )
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state_dict = strip_state_dict_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("[WARNING] Missing keys:")
        for k in missing:
            print("   ", k)

    if unexpected:
        print("[WARNING] Unexpected keys:")
        for k in unexpected:
            print("   ", k)

    if missing or unexpected:
        print(
            "[WARNING] The checkpoint/model configuration may not exactly match "
            "the training configuration."
        )
    else:
        print("[OK] Checkpoint loaded with an exact state_dict match.")


# =============================================================================
# Per-case segmentation metrics
# =============================================================================

def overlap_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> Tuple[float, float, float, float]:
    """
    Binary foreground metrics for ONE image.

    Returns:
        dice, iou, precision, sensitivity
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    # Keep mathematically sensible empty-mask handling.
    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        sensitivity = 1.0
        return dice, iou, precision, sensitivity

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (union + eps)

    if pred_sum == 0:
        precision = 0.0
    else:
        precision = (tp + eps) / (tp + fp + eps)

    if gt_sum == 0:
        # GT has no foreground but prediction is non-empty.
        sensitivity = 1.0
    else:
        sensitivity = (tp + eps) / (tp + fn + eps)

    return float(dice), float(iou), float(precision), float(sensitivity)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric 95th percentile Hausdorff distance in PIXELS for one 2-D mask.

    Empty-mask convention:
      both empty  -> 0
      one empty   -> image diagonal (finite worst-case penalty)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_any = bool(pred.any())
    gt_any = bool(gt.any())

    if not pred_any and not gt_any:
        return 0.0

    h, w = pred.shape[-2], pred.shape[-1]

    if pred_any != gt_any:
        return float(math.hypot(max(h - 1, 1), max(w - 1, 1)))

    pred_surface = _surface(pred)
    gt_surface = _surface(gt)

    # Distance to the nearest surface pixel.
    dist_to_gt = distance_transform_edt(~gt_surface)
    dist_to_pred = distance_transform_edt(~pred_surface)

    d_pred_gt = dist_to_gt[pred_surface]
    d_gt_pred = dist_to_pred[gt_surface]

    all_dist = np.concatenate([d_pred_gt, d_gt_pred], axis=0)

    if all_dist.size == 0:
        return 0.0

    return float(np.percentile(all_dist, 95))


# =============================================================================
# Runtime reuse of train_Lung_新版.py
# =============================================================================

def import_training_module(train_script: str):
    """
    Import the user's training file so the test code can reuse the SAME
    dataset/list/model construction functions instead of re-implementing them.
    """
    train_script = os.path.abspath(train_script)

    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"Training script not found: {train_script}")

    train_dir = os.path.dirname(train_script)
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    spec = importlib.util.spec_from_file_location("lung_train_runtime", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {train_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_train_args(train_module, checkpoint_path: str | None = None):
    """
    Recover the exact training arguments used by train_Lung_新版.py.

    Priority:
      1) training module get_args(), if it exposes one;
      2) checkpoint['args'], which train_Lung_新版.py saves in every full checkpoint.

    The current train_Lung_新版.py builds argparse inside main(), so importing the
    module does NOT expose those parsed arguments.  For this script, checkpoint
    recovery is therefore the normal path.
    """
    if hasattr(train_module, "get_args"):
        fn = train_module.get_args
        old_argv = sys.argv[:]
        try:
            sys.argv = [old_argv[0]]
            sig = inspect.signature(fn)
            if "known" in sig.parameters:
                return fn(known=True)
            return fn()
        finally:
            sys.argv = old_argv

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
            print("[OK] Recovered training arguments from checkpoint['args'].")
            return SimpleNamespace(**ckpt["args"])

    return None


def override_if_present(obj: Any, name: str, value: Any) -> None:
    if obj is not None and value is not None and hasattr(obj, name):
        setattr(obj, name, value)


def _build_val_loader_like_current_lung_training(train_module, train_args):
    """Reproduce the validation-loader block used inside train_Lung_新版.py main()."""
    if train_args is None:
        raise RuntimeError(
            "Training arguments are unavailable. The current train_Lung_新版.py "
            "creates argparse only inside main(), so evaluation needs a full "
            "checkpoint containing checkpoint['args']."
        )

    required = (
        "prepare_or_load_split_lists",
        "estimate_training_intensity_range",
        "LungAugment",
        "LungSegDataset",
    )
    missing = [name for name in required if not hasattr(train_module, name)]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct the validation loader because train_Lung_新版.py "
            f"is missing: {missing}"
        )

    strict_pairing = not bool(getattr(train_args, "allow_unpaired", 0))
    (
        tr_imgs,
        tr_msks,
        va_imgs,
        va_msks,
        train_list_path,
        val_list_path,
        _split_meta_path,
    ) = train_module.prepare_or_load_split_lists(
        args=train_args,
        strict_pairing=strict_pairing,
    )

    norm_lo, norm_hi = train_module.estimate_training_intensity_range(
        tr_imgs,
        low_percentile=train_args.norm_low_pct,
        high_percentile=train_args.norm_high_pct,
        sample_step=train_args.norm_sample_step,
    )

    augmenter = train_module.LungAugment(img_size=train_args.img_size)
    val_ds = train_module.LungSegDataset(
        va_imgs,
        va_msks,
        augmenter=augmenter,
        train=False,
        norm_lo=norm_lo,
        norm_hi=norm_hi,
    )

    loader_kwargs = {
        "batch_size": train_args.batch_size,
        "num_workers": train_args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if train_args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print("[OK] Reconstructed validation DataLoader using train_Lung_新版.py's exact workflow.")
    print("[OK] Train list:", train_list_path)
    print("[OK] Val list  :", val_list_path)
    print("[OK] Normalization range:", (norm_lo, norm_hi))
    return val_loader


def find_val_loader(train_module, train_args):
    """
    Prefer an exposed loader factory, then fall back to the exact validation
    construction used inside the current train_Lung_新版.py main().
    """
    candidate_names = (
        "get_data",
        "get_loaders",
        "get_loader",
        "build_dataloaders",
        "build_dataloader",
        "create_dataloaders",
        "create_dataloader",
    )

    last_error = None
    for name in candidate_names:
        fn = getattr(train_module, name, None)
        if not callable(fn):
            continue
        try:
            result = fn(train_args) if train_args is not None else fn()
        except TypeError as exc:
            last_error = exc
            continue

        if isinstance(result, dict):
            for key in ("val", "valid", "validation", "val_loader", "valid_loader"):
                if key in result:
                    print(f"[OK] Reusing {name}() -> ['{key}'] from training script.")
                    return result[key]

        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                print(f"[OK] Reusing {name}() second loader as validation loader.")
                return result[1]
            if len(result) == 1:
                print(f"[OK] Reusing sole loader returned by {name}().")
                return result[0]

        if hasattr(result, "__iter__") and hasattr(result, "dataset"):
            print(f"[OK] Reusing loader returned by {name}().")
            return result

    for name in ("val_loader", "valid_loader", "validation_loader"):
        loader = getattr(train_module, name, None)
        if loader is not None:
            print(f"[OK] Reusing module-level {name}.")
            return loader

    try:
        return _build_val_loader_like_current_lung_training(train_module, train_args)
    except Exception as exc:
        msg = (
            "Could not construct the validation DataLoader from train_Lung_新版.py."
            f"\nFallback reconstruction error: {type(exc).__name__}: {exc}"
        )
        if last_error is not None:
            msg += f"\nLast loader-factory error: {last_error}"
        raise RuntimeError(msg) from exc


def build_model_from_training(train_module, train_args):
    """Reuse the training script's model factory, including (model, cfg) returns."""
    candidate_names = ("build_model", "get_model", "create_model")

    for name in candidate_names:
        fn = getattr(train_module, name, None)
        if not callable(fn):
            continue

        try:
            result = fn(train_args) if train_args is not None else fn()
        except TypeError:
            try:
                result = fn()
            except TypeError:
                continue

        if isinstance(result, torch.nn.Module):
            print(f"[OK] Reusing {name}() from training script.")
            return result

        # Current train_Lung_新版.py: build_model(args) -> (model, cfg)
        if isinstance(result, (tuple, list)) and result:
            if isinstance(result[0], torch.nn.Module):
                print(f"[OK] Reusing {name}() first return value as model.")
                return result[0]

        if isinstance(result, dict):
            for key in ("model", "net", "network"):
                if isinstance(result.get(key), torch.nn.Module):
                    print(f"[OK] Reusing {name}()['{key}'] as model.")
                    return result[key]

    model = getattr(train_module, "model", None)
    if isinstance(model, torch.nn.Module):
        print("[OK] Reusing module-level model from training script.")
        return model

    raise RuntimeError(
        "Could not automatically build the model from train_Lung_新版.py. "
        "Expected build_model(args) to return a model or (model, cfg)."
    )


# =============================================================================
# Batch extraction
# =============================================================================

def extract_image_label(batch):
    """
    The validation loop in many segmentation projects returns a tuple/list:
        image, label, ...
    or a dict:
        {'image': ..., 'label': ...}

    We use ONLY the base validation image, never augmentation views.
    """
    if isinstance(batch, dict):
        image_keys = ("image", "img", "images", "data", "input")
        label_keys = ("label", "mask", "target", "gt", "seg")

        image = None
        label = None

        for k in image_keys:
            if k in batch:
                image = batch[k]
                break

        for k in label_keys:
            if k in batch:
                label = batch[k]
                break

        if image is None or label is None:
            raise RuntimeError(
                f"Cannot identify image/label in dict batch. Keys={list(batch.keys())}"
            )
        return image, label

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise RuntimeError("Validation batch must contain at least image and label.")
        return batch[0], batch[1]

    raise RuntimeError(f"Unsupported validation batch type: {type(batch)}")


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    """
    Supports:
      binary single-channel logits: [B,1,H,W] -> sigmoid > 0.5
      multi-class logits:           [B,C,H,W] -> argmax
    """
    if logits.dim() != 4:
        raise RuntimeError(f"Expected model output [B,C,H,W], got {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) >= 0.5).long()

    return torch.argmax(logits, dim=1)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    foreground_class: int = 1,
    use_amp: bool = True,
):
    model.eval()

    rows: List[Dict[str, float]] = []

    amp_enabled = bool(use_amp) and device.type == "cuda"

    case_index = 0

    for batch_idx, batch in enumerate(val_loader):
        image, label = extract_image_label(batch)

        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        target = normalize_label(label)

        with autocast(enabled=amp_enabled):
            logits = model(image)

        # Some networks may return (logits, aux) or a list of deep-supervision outputs.
        if isinstance(logits, dict):
            for key in ("logits", "out", "output", "pred"):
                if key in logits:
                    logits = logits[key]
                    break

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not torch.is_tensor(logits):
            raise RuntimeError(f"Unsupported model output type: {type(logits)}")

        logits = logits.float()
        pred = logits_to_prediction(logits)

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.shape != target_np.shape:
            raise RuntimeError(
                f"Prediction/GT shape mismatch: pred={pred_np.shape}, gt={target_np.shape}"
            )

        for i in range(pred_np.shape[0]):
            p = (pred_np[i] == foreground_class)
            g = (target_np[i] == foreground_class)

            dice, iou, pre, sen = overlap_metrics(p, g)
            hd95 = hd95_binary(p, g)

            row = {
                "index": int(case_index),
                "dice": dice,
                "iou": iou,
                "precision": pre,
                "sensitivity": sen,
                "hd95": hd95,
            }
            rows.append(row)

            print(
                f"[{case_index:04d}] "
                f"Dice={dice:.6f}  "
                f"IoU={iou:.6f}  "
                f"Pre={pre:.6f}  "
                f"Sen={sen:.6f}  "
                f"HD95={hd95:.6f}"
            )
            case_index += 1

    if not rows:
        raise RuntimeError("Validation loader produced zero samples.")

    return rows


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {"cases": len(rows)}

    mapping = {
        "dice": "dice",
        "iou": "iou",
        "precision": "precision",
        "sensitivity": "sensitivity",
        "hd95": "hd95",
    }

    for out_name, key in mapping.items():
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        summary[f"{out_name}_mean"] = float(np.mean(values))
        summary[f"{out_name}_std"] = float(np.std(values, ddof=0))

    return summary


def save_csv(rows, summary, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    per_case_path = os.path.join(output_dir, "lung_val_metrics_per_case.csv")
    with open(per_case_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "dice", "iou", "precision", "sensitivity", "hd95"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(output_dir, "lung_val_metrics_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])

        for metric in ("dice", "iou", "precision", "sensitivity", "hd95"):
            writer.writerow([
                metric,
                summary[f"{metric}_mean"],
                summary[f"{metric}_std"],
            ])

    print("[SAVE]", per_case_path)
    print("[SAVE]", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Lung validation data using the SAME validation-data/model "
            "construction functions from train_Lung_新版.py."
        )
    )

    parser.add_argument(
        "--train_script",
        type=str,
        default="train_Lung.py",
        help="Path to the Lung training script.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"",
        help="Path to trained .pt/.pth weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"F:\0_paper_code\Mamba_new\___out",
    )
    parser.add_argument("--foreground_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_amp", type=int, default=1, choices=[0, 1])

    # Optional overrides. They are applied ONLY if the training args expose
    # fields with these names.
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=None, help="Alias for training --img_size.")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--train_img_dir", type=str, default=None)
    parser.add_argument("--train_mask_dir", type=str, default=None)
    parser.add_argument("--val_img_dir", type=str, default=None)
    parser.add_argument("--val_mask_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("Lung validation performance evaluation")
    print("=" * 88)
    print("Device        :", device)
    print("Train script  :", os.path.abspath(args.train_script))
    print("Checkpoint    :", os.path.abspath(args.checkpoint))

    train_module = import_training_module(args.train_script)
    train_args = get_train_args(train_module, args.checkpoint)

    # Apply only explicit test-time CLI overrides.
    override_if_present(train_args, "data_path", args.data_path)
    override_if_present(train_args, "batch_size", args.batch_size)
    override_if_present(train_args, "num_workers", args.num_workers)

    requested_img_size = args.img_size if args.img_size is not None else args.image_size
    override_if_present(train_args, "img_size", requested_img_size)
    override_if_present(train_args, "image_size", requested_img_size)

    for name in ("split_dir", "train_img_dir", "train_mask_dir", "val_img_dir", "val_mask_dir"):
        override_if_present(train_args, name, getattr(args, name))

    val_loader = find_val_loader(train_module, train_args)
    model = build_model_from_training(train_module, train_args).to(device)

    print("Val samples   :", len(val_loader.dataset) if hasattr(val_loader, "dataset") else "unknown")
    print("Val batches   :", len(val_loader) if hasattr(val_loader, "__len__") else "unknown")

    load_weights(model, args.checkpoint, device)

    rows = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        foreground_class=args.foreground_class,
        use_amp=bool(args.use_amp),
    )

    summary = summarize(rows)

    print()
    print("=" * 88)
    print(f"FINAL LUNG VAL RESULTS (foreground class = {args.foreground_class})")
    print("=" * 88)
    print(f"Cases : {summary['cases']}")
    print(f"Dice  : {summary['dice_mean']:.6f} ± {summary['dice_std']:.6f}")
    print(f"IoU   : {summary['iou_mean']:.6f} ± {summary['iou_std']:.6f}")
    print(f"Pre   : {summary['precision_mean']:.6f} ± {summary['precision_std']:.6f}")
    print(f"Sen   : {summary['sensitivity_mean']:.6f} ± {summary['sensitivity_std']:.6f}")
    print(f"HD95  : {summary['hd95_mean']:.6f} ± {summary['hd95_std']:.6f} pixels")
    print("=" * 88)

    save_csv(rows, summary, args.output_dir)


if __name__ == "__main__":
    main()
