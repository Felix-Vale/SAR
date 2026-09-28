# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Evaluation for the private 4-class corneal dataset trained by train_cornea.py.

Key alignment with train_cornea.py
----------------------------------
1. Reuse split_dir/val_list.txt exactly; DO NOT resplit.
2. Reuse PairRecord/load_pair_list/validate_split_records.
3. Reuse CorneaDataset + CorneaAugment(train=False).
4. Same resize, channel conversion and normalization as training validation.
5. Same SARMambaMGEFUNet model configuration.
6. Prediction = argmax(logits, dim=1).
7. Background class is read from checkpoint/task_meta or inferred by the same
   train-mask border strategy.
8. The task is 4-class segmentation. Metrics are reported per class and as a
   foreground macro average excluding background.

Outputs
-------
- Terminal summary
- cornea_val_per_case.csv
- cornea_val_per_class_summary.csv
- cornea_val_foreground_summary.csv

HD95 unit
---------
Pixels in the resized validation resolution (default 256 x 256).

Per-case absent-class convention
--------------------------------
- GT empty AND prediction empty:
    Dice/IoU/Precision/Sensitivity/HD95 = NaN for that class/case and excluded
    from case-level statistics. This prevents absent structures from creating
    artificial perfect scores.
- Exactly one of GT/pred empty:
    overlap scores = 0 where mathematically appropriate; HD95 is assigned the
    image diagonal as a finite worst-case penalty.

Dataset-level Dice/IoU/Precision/Sensitivity are ALSO computed from a single
global confusion matrix, matching the aggregation style used by train_cornea.py.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception as exc:
    raise ImportError(
        "HD95 requires scipy. Install it in the current environment with:\n"
        "    pip install scipy"
    ) from exc


# =============================================================================
# Import the exact data/model utilities used by train_cornea.py
# =============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from train_cornea import (  # noqa: E402
    AugCfg,
    CorneaAugment,
    CorneaDataset,
    PairRecord,
    build_model,
    infer_background_index,
    load_pair_list,
    validate_split_records,
)


NUM_CLASSES = 4


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate train_cornea.py validation split with Dice/IoU/Pre/Sen/HD95."
    )

    p.add_argument("--checkpoint", type=str, default=r"")

    # Data locations. If omitted, first try checkpoint args.
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--mask_dir", type=str, default=None)
    p.add_argument("--split_dir", type=str, default=None)

    p.add_argument(
        "--output_dir",
        type=str,
        default=r"F:\0_paper_code\Mamba_new\___out",
    )

    # Optional overrides. Normally leave these unset so checkpoint config is used.
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--in_ch", type=int, choices=[1, 3], default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--background_index", type=int, choices=[-1, 0, 1, 2, 3], default=None)
    p.add_argument("--amp", action="store_true")

    return p.parse_args()


# =============================================================================
# Checkpoint / saved training config
# =============================================================================

def load_checkpoint(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "train_cornea.py saves a checkpoint dictionary, but this file is not a dict."
        )

    return ckpt


def _saved_args_dict(ckpt: Dict) -> Dict:
    args = ckpt.get("args", {})
    if isinstance(args, argparse.Namespace):
        return vars(args)
    if isinstance(args, dict):
        return dict(args)
    return {}


def choose(cli_value, saved: Dict, key: str, default):
    if cli_value is not None:
        return cli_value
    if key in saved and saved[key] is not None:
        return saved[key]
    return default


def build_runtime_args(cli, ckpt: Dict):
    """
    Reconstruct the model/data settings from the exact args saved by train_cornea.py.
    """
    saved = _saved_args_dict(ckpt)

    image_dir = choose(cli.image_dir, saved, "image_dir", None)
    mask_dir = choose(cli.mask_dir, saved, "mask_dir", None)

    # train_cornea.py uses split_dir if non-empty, otherwise out_dir/split.
    saved_split_dir = saved.get("split_dir")
    if cli.split_dir is not None:
        split_dir = cli.split_dir
    elif saved_split_dir:
        split_dir = saved_split_dir
    elif saved.get("out_dir"):
        split_dir = os.path.join(saved["out_dir"], "split")
    else:
        split_dir = None

    if image_dir is None:
        raise ValueError(
            "--image_dir is required because it could not be recovered from checkpoint args."
        )
    if mask_dir is None:
        raise ValueError(
            "--mask_dir is required because it could not be recovered from checkpoint args."
        )
    if split_dir is None:
        raise ValueError(
            "--split_dir is required because it could not be recovered from checkpoint args."
        )

    # Model arguments exactly matching build_model() in train_cornea.py.
    rt = SimpleNamespace(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_dir=split_dir,
        image_size=int(choose(cli.image_size, saved, "image_size", 256)),
        in_ch=int(choose(cli.in_ch, saved, "in_ch", 1)),
        batch_size=int(choose(cli.batch_size, saved, "batch_size", 4)),
        num_workers=int(choose(cli.num_workers, saved, "num_workers", 0)),
        d_state=int(saved.get("d_state", 16)),
        expand=float(saved.get("expand", 1.5)),
        dt_rank=int(saved.get("dt_rank", 4)),
        token_dim=int(saved.get("token_dim", 96)),
        kmax=int(saved.get("kmax", 8)),
        token_heads=int(saved.get("token_heads", 4)),
        token_pool=int(saved.get("token_pool", 8)),
        route_topk=int(saved.get("route_topk", 2)),
        route_mode=str(saved.get("route_mode", "topk_st")),
        decoder_refine_blocks=int(saved.get("decoder_refine_blocks", 2)),
    )

    return rt, saved


def resolve_background_index(
    cli,
    ckpt: Dict,
    split_dir: str,
    mask_dir: str,
    train_records: Sequence[PairRecord],
) -> int:
    if cli.background_index is not None and cli.background_index >= 0:
        return int(cli.background_index)

    if "background_index" in ckpt:
        bg = int(ckpt["background_index"])
        if 0 <= bg < NUM_CLASSES:
            return bg

    task_meta_path = os.path.join(split_dir, "task_meta.json")
    if os.path.isfile(task_meta_path):
        with open(task_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "background_index" in meta:
            bg = int(meta["background_index"])
            if 0 <= bg < NUM_CLASSES:
                return bg

    bg, border_counts = infer_background_index(
        mask_dir=mask_dir,
        train_records=train_records,
        num_classes=NUM_CLASSES,
    )
    print(
        "[BACKGROUND] checkpoint/task_meta had no valid background index; "
        f"re-inferred from TRAIN borders: counts={border_counts} -> class {bg}"
    )
    return int(bg)


def load_model_weights(model: torch.nn.Module, ckpt: Dict):
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # Fallback for a pure state_dict file.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            state = ckpt
        else:
            raise KeyError(
                "Checkpoint has no 'model_state_dict'. "
                f"Available keys: {list(ckpt.keys())[:30]}"
            )

    # DataParallel/DDP compatibility.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)


# =============================================================================
# Same saved val list + same validation preprocessing
# =============================================================================

def make_val_loader(rt):
    train_list = os.path.join(rt.split_dir, "train_list.txt")
    # val_list = os.path.join(rt.split_dir, "val_list.txt")
    val_list = os.path.join(rt.split_dir, "val_list.txt")


    if not os.path.isfile(train_list):
        raise FileNotFoundError(f"Missing training split list: {train_list}")
    if not os.path.isfile(val_list):
        raise FileNotFoundError(f"Missing validation split list: {val_list}")

    train_records = load_pair_list(train_list)
    val_records = load_pair_list(val_list)

    # Same safety check as train_cornea.py.
    validate_split_records(
        image_dir=rt.image_dir,
        mask_dir=rt.mask_dir,
        train_records=train_records,
        val_records=val_records,
    )

    val_aug = CorneaAugment(
        AugCfg(
            train=False,
            base_resize=(rt.image_size, rt.image_size),
            in_ch=rt.in_ch,
        )
    )

    val_ds = CorneaDataset(
        image_dir=rt.image_dir,
        mask_dir=rt.mask_dir,
        records=val_records,
        aug=val_aug,
        num_classes=NUM_CLASSES,
    )

    loader_kwargs = {
        "num_workers": rt.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if rt.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(
        val_ds,
        batch_size=rt.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_records, val_records, val_loader


# =============================================================================
# Metrics
# =============================================================================

def update_confusion(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray):
    valid = (gt >= 0) & (gt < NUM_CLASSES)
    idx = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    binc = np.bincount(idx.reshape(-1), minlength=NUM_CLASSES * NUM_CLASSES)
    cm += binc.reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_global_confusion(cm: np.ndarray, eps: float = 1e-7):
    cm = cm.astype(np.float64)

    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)
    gt_sum = cm.sum(axis=1)
    fp = pred_sum - tp
    fn = gt_sum - tp

    dice = (2 * tp + eps) / (pred_sum + gt_sum + eps)
    iou = (tp + eps) / (pred_sum + gt_sum - tp + eps)
    precision = (tp + eps) / (tp + fp + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, sensitivity


def _mask_surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    er = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    return np.logical_xor(mask, er)


def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    p_any = bool(pred.any())
    g_any = bool(gt.any())

    # Absent in both -> not a meaningful boundary measurement.
    if not p_any and not g_any:
        return float("nan")

    h, w = pred.shape
    if p_any != g_any:
        return float(math.hypot(max(h - 1, 1), max(w - 1, 1)))

    ps = _mask_surface(pred)
    gs = _mask_surface(gt)

    dist_to_g = distance_transform_edt(~gs)
    dist_to_p = distance_transform_edt(~ps)

    d1 = dist_to_g[ps]
    d2 = dist_to_p[gs]
    d = np.concatenate([d1, d2])

    if d.size == 0:
        return 0.0

    return float(np.percentile(d, 95))


def per_class_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_id: int,
    eps: float = 1e-7,
):
    p = pred == class_id
    g = gt == class_id

    p_sum = int(p.sum())
    g_sum = int(g.sum())

    # Do not treat complete absence as perfect segmentation for case statistics.
    if p_sum == 0 and g_sum == 0:
        return {
            "dice": np.nan,
            "iou": np.nan,
            "precision": np.nan,
            "sensitivity": np.nan,
            "hd95": np.nan,
            "gt_pixels": 0,
            "pred_pixels": 0,
        }

    tp = int(np.logical_and(p, g).sum())
    fp = p_sum - tp
    fn = g_sum - tp

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    # No predicted pixels while GT exists -> precision=0.
    precision = 0.0 if p_sum == 0 else (tp + eps) / (tp + fp + eps)

    # If GT absent but model hallucinated the class, sensitivity is undefined;
    # use NaN so it is not mixed into recall statistics.
    sensitivity = np.nan if g_sum == 0 else (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "hd95": hd95_binary(p, g),
        "gt_pixels": g_sum,
        "pred_pixels": p_sum,
    }


def finite_mean_std(values) -> Tuple[float, float, int]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return float("nan"), float("nan"), 0

    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, val_records, device, use_amp: bool):
    model.eval()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows: List[Dict] = []

    record_index = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch_idx, (img, mask) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(img)

        logits = logits.float()
        pred = logits.argmax(dim=1)

        pred_np = pred.detach().cpu().numpy()
        gt_np = mask.detach().cpu().numpy()

        for b in range(pred_np.shape[0]):
            p = pred_np[b]
            g = gt_np[b]

            update_confusion(cm, p, g)

            rec = val_records[record_index]

            for c in range(NUM_CLASSES):
                m = per_class_case_metrics(p, g, c)
                rows.append(
                    {
                        "case_index": record_index,
                        "image_name": rec.image_name,
                        "mask_name": rec.mask_name,
                        "class_id": c,
                        **m,
                    }
                )

            record_index += 1

        print(
            f"\rEvaluating: {record_index}/{len(val_records)}",
            end="",
            flush=True,
        )

    print()

    if record_index != len(val_records):
        raise RuntimeError(
            f"Evaluated {record_index} cases but val_list contains {len(val_records)}."
        )

    return cm, rows


# =============================================================================
# Summary / CSV
# =============================================================================

def summarize(cm, rows, background_index: int):
    dice_g, iou_g, pre_g, sen_g = metrics_from_global_confusion(cm)

    class_summary = []

    for c in range(NUM_CLASSES):
        cr = [r for r in rows if r["class_id"] == c]

        dice_m, dice_s, dice_n = finite_mean_std([r["dice"] for r in cr])
        iou_m, iou_s, iou_n = finite_mean_std([r["iou"] for r in cr])
        pre_m, pre_s, pre_n = finite_mean_std([r["precision"] for r in cr])
        sen_m, sen_s, sen_n = finite_mean_std([r["sensitivity"] for r in cr])
        hd_m, hd_s, hd_n = finite_mean_std([r["hd95"] for r in cr])

        class_summary.append(
            {
                "class_id": c,
                "is_background": int(c == background_index),
                # Dataset/global-confusion metrics: closest to train_cornea.py aggregation.
                "dice_global": float(dice_g[c]),
                "iou_global": float(iou_g[c]),
                "precision_global": float(pre_g[c]),
                "sensitivity_global": float(sen_g[c]),
                # Case-wise descriptive statistics.
                "dice_case_mean": dice_m,
                "dice_case_std": dice_s,
                "dice_case_n": dice_n,
                "iou_case_mean": iou_m,
                "iou_case_std": iou_s,
                "iou_case_n": iou_n,
                "precision_case_mean": pre_m,
                "precision_case_std": pre_s,
                "precision_case_n": pre_n,
                "sensitivity_case_mean": sen_m,
                "sensitivity_case_std": sen_s,
                "sensitivity_case_n": sen_n,
                "hd95_case_mean": hd_m,
                "hd95_case_std": hd_s,
                "hd95_case_n": hd_n,
            }
        )

    fg = [c for c in range(NUM_CLASSES) if c != background_index]

    # Foreground macro of dataset-level per-class metrics.
    fg_global = {
        "dice": float(np.mean(dice_g[fg])),
        "iou": float(np.mean(iou_g[fg])),
        "precision": float(np.mean(pre_g[fg])),
        "sensitivity": float(np.mean(sen_g[fg])),
    }

    # HD95: mean of per-foreground-class case means.
    fg_class_hd = np.asarray(
        [class_summary[c]["hd95_case_mean"] for c in fg],
        dtype=np.float64,
    )
    fg_class_hd = fg_class_hd[np.isfinite(fg_class_hd)]
    fg_global["hd95"] = (
        float(fg_class_hd.mean()) if fg_class_hd.size else float("nan")
    )

    # Image-wise foreground macro statistics.
    case_ids = sorted({int(r["case_index"]) for r in rows})
    per_case_fg = []

    for case_id in case_ids:
        rr = [
            r for r in rows
            if r["case_index"] == case_id and r["class_id"] in fg
        ]

        item = {"case_index": case_id}
        for metric in ("dice", "iou", "precision", "sensitivity", "hd95"):
            vals = np.asarray([r[metric] for r in rr], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            item[metric] = float(vals.mean()) if vals.size else float("nan")
        per_case_fg.append(item)

    fg_case_stats = {}
    for metric in ("dice", "iou", "precision", "sensitivity", "hd95"):
        mean, std, n = finite_mean_std([r[metric] for r in per_case_fg])
        fg_case_stats[metric] = {"mean": mean, "std": std, "n": n}

    return class_summary, fg_global, per_case_fg, fg_case_stats


def write_csvs(output_dir, rows, class_summary, per_case_fg, fg_global, fg_case_stats):
    os.makedirs(output_dir, exist_ok=True)

    per_case_path = os.path.join(output_dir, "cornea_val_per_case.csv")
    class_path = os.path.join(output_dir, "cornea_val_per_class_summary.csv")
    fg_path = os.path.join(output_dir, "cornea_val_foreground_summary.csv")

    with open(per_case_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "case_index", "image_name", "mask_name", "class_id",
            "dice", "iou", "precision", "sensitivity", "hd95",
            "gt_pixels", "pred_pixels",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(class_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = list(class_summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(class_summary)

    with open(fg_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "metric",
            "global_foreground_macro",
            "case_mean",
            "case_std",
            "case_n",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for metric in ("dice", "iou", "precision", "sensitivity", "hd95"):
            w.writerow(
                {
                    "metric": metric,
                    "global_foreground_macro": fg_global[metric],
                    "case_mean": fg_case_stats[metric]["mean"],
                    "case_std": fg_case_stats[metric]["std"],
                    "case_n": fg_case_stats[metric]["n"],
                }
            )

    case_fg_path = os.path.join(output_dir, "cornea_val_foreground_per_case.csv")
    with open(case_fg_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["case_index", "dice", "iou", "precision", "sensitivity", "hd95"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_case_fg)

    return per_case_path, class_path, fg_path, case_fg_path


# =============================================================================
# Main
# =============================================================================

def main():
    cli = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(cli.checkpoint, device)

    rt, saved_args = build_runtime_args(cli, ckpt)

    train_records, val_records, val_loader = make_val_loader(rt)

    background_index = resolve_background_index(
        cli=cli,
        ckpt=ckpt,
        split_dir=rt.split_dir,
        mask_dir=rt.mask_dir,
        train_records=train_records,
    )

    if not (0 <= background_index < NUM_CLASSES):
        raise RuntimeError(f"Invalid background_index={background_index}")

    model = build_model(rt).to(device)
    load_model_weights(model, ckpt)

    use_amp = bool(cli.amp and torch.cuda.is_available())

    print("=" * 100)
    print("Cornea private validation evaluation")
    print("=" * 100)
    print("Device              :", device)
    print("Checkpoint          :", os.path.abspath(cli.checkpoint))
    print("Image dir           :", os.path.abspath(rt.image_dir))
    print("Mask dir            :", os.path.abspath(rt.mask_dir))
    print("Val list            :", os.path.abspath(os.path.join(rt.split_dir, "val_list.txt")))
    print("Validation samples  :", len(val_records))
    print("Image size          :", rt.image_size)
    print("Input channels      :", rt.in_ch)
    print("Classes             :", NUM_CLASSES)
    print("Background class    :", background_index)
    print("Foreground classes  :", [c for c in range(NUM_CLASSES) if c != background_index])
    print("AMP                 :", use_amp)
    print("=" * 100)

    cm, rows = evaluate(
        model=model,
        loader=val_loader,
        val_records=val_records,
        device=device,
        use_amp=use_amp,
    )

    class_summary, fg_global, per_case_fg, fg_case_stats = summarize(
        cm=cm,
        rows=rows,
        background_index=background_index,
    )

    print()
    print("=" * 100)
    print("PER-CLASS DATASET-LEVEL RESULTS")
    print("(Dice/IoU/Pre/Sen from one global confusion matrix; HD95 = case mean ± std)")
    print("=" * 100)

    for r in class_summary:
        tag = " [BACKGROUND]" if r["is_background"] else ""
        print(
            f"Class {r['class_id']}{tag}: "
            f"Dice={r['dice_global']:.6f} | "
            f"IoU={r['iou_global']:.6f} | "
            f"Pre={r['precision_global']:.6f} | "
            f"Sen={r['sensitivity_global']:.6f} | "
            f"HD95={r['hd95_case_mean']:.6f} ± {r['hd95_case_std']:.6f} px"
        )

    print()
    print("=" * 100)
    print("FOREGROUND MACRO RESULTS (background excluded)")
    print("=" * 100)
    print(
        f"Dice  : {fg_global['dice']:.6f} "
        f"| case-wise {fg_case_stats['dice']['mean']:.6f} ± "
        f"{fg_case_stats['dice']['std']:.6f}"
    )
    print(
        f"IoU   : {fg_global['iou']:.6f} "
        f"| case-wise {fg_case_stats['iou']['mean']:.6f} ± "
        f"{fg_case_stats['iou']['std']:.6f}"
    )
    print(
        f"Pre   : {fg_global['precision']:.6f} "
        f"| case-wise {fg_case_stats['precision']['mean']:.6f} ± "
        f"{fg_case_stats['precision']['std']:.6f}"
    )
    print(
        f"Sen   : {fg_global['sensitivity']:.6f} "
        f"| case-wise {fg_case_stats['sensitivity']['mean']:.6f} ± "
        f"{fg_case_stats['sensitivity']['std']:.6f}"
    )
    print(
        f"HD95  : {fg_global['hd95']:.6f} px "
        f"| case-wise {fg_case_stats['hd95']['mean']:.6f} ± "
        f"{fg_case_stats['hd95']['std']:.6f} px"
    )
    print("=" * 100)

    paths = write_csvs(
        output_dir=cli.output_dir,
        rows=rows,
        class_summary=class_summary,
        per_case_fg=per_case_fg,
        fg_global=fg_global,
        fg_case_stats=fg_case_stats,
    )

    print("[SAVE]", paths[0])
    print("[SAVE]", paths[1])
    print("[SAVE]", paths[2])
    print("[SAVE]", paths[3])


if __name__ == "__main__":
    main()
