from __future__ import annotations
import os
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))
sys.path.append(os.path.dirname(os.path.dirname(_THIS_DIR)))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# Same ISIC dataset implementation used in train_adamix_mt.py
from data.dataset_ours_isic import ISICDataset
from wheels.loss_functions import DSCLossH
from wheels.torch_utils import seed_torch

# Your SAR-Mamba + SIA + MGEF U-Net
from UNet2 import SARMambaMGEFUNet, SARMambaMGEFConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Arguments
# =============================================================================

def get_args(known: bool = False):
    parser = argparse.ArgumentParser(
        description="Supervised SARMambaMGEFUNet training on ISIC"
    )

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--project",
        type=str,
        default=os.path.join(_THIS_DIR, "runs", "SAR_MGEF_Supervised_ISIC"),
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=r"",
    )

    # Same resize target used by train_adamix_mt.py
    parser.add_argument("--image_size", type=int, default=128)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_freq", type=int, default=10)

    parser.add_argument(
        "--save_freq",
        type=int,
        default=4,
        help="Save an additional unique snapshot every N epochs; 0 disables.",
    )

    parser.add_argument(
        "--use_amp",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = CUDA AMP, 0 = FP32",
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max gradient norm; <=0 disables clipping.",
    )

    # train_adamix_mt.py receives image, label, imageA1, imageA2.
    # Its main student path uses A1, so supervised training defaults to A1 too.
    parser.add_argument(
        "--train_view",
        type=str,
        default="A1",
        choices=["image", "A1", "A2"],
    )

    # Model
    parser.add_argument(
        "--channels",
        type=int,
        nargs=5,
        default=[32, 64, 128, 256, 512],
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

    args = parser.parse_known_args()[0] if known else parser.parse_args()

    return args


# =============================================================================
# Data
# =============================================================================

def _make_loader_kwargs(args) -> Dict:
    """
    Same DataLoader pattern as train_adamix_mt.py.
    """
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return kwargs


def get_data(args) -> Tuple[DataLoader, DataLoader]:
    """
    PURE FULLY SUPERVISED loading.

    The same ISICDataset performs the resize / augmentation.
    All available labeled training samples are used.
    No unlabeled set, no labeled-subset sampling, no repeated labeled set,
    no reference set, and no auxiliary mask stream.
    """
    # NOTE: this dataset class comes from the former semi-supervised codebase,
    # so its API contains `labeled` / `percentage` fields. Here they are FIXED
    # to labeled=True and percentage=1.0, therefore this is a normal full
    # supervised training set and cannot be changed from the command line.
    train_set = ISICDataset(
        image_path=args.data_path,
        stage="train",
        image_size=args.image_size,
        is_augmentation=True,
        labeled=True,
        percentage=1.0,  # FULL supervised training set; fixed, not configurable
    )

    val_set = ISICDataset(
        image_path=args.data_path,
        stage="val",
        image_size=args.image_size,
        is_augmentation=False,
    )

    loader_kwargs = _make_loader_kwargs(args)

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        dataset=val_set,
        shuffle=False,
        **loader_kwargs,
    )

    print("Train samples :", len(train_set))
    print("Val samples   :", len(val_set))
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))

    return train_loader, val_loader


# =============================================================================
# Model / label / metrics helpers
# =============================================================================

def build_model(args) -> SARMambaMGEFUNet:
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

    # Do not globally reinitialize the network, because SAR/MGEF contains
    # intentional zero-initialized conditioning layers.
    # Only initialize the final segmentation head.
    if isinstance(model.head, nn.Conv2d):
        nn.init.kaiming_normal_(
            model.head.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    return model


def _normalize_label(label: torch.Tensor) -> torch.Tensor:
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)
    return label.long()


def _select_train_view(batch: Sequence[torch.Tensor], train_view: str):
    """
    Expected training output from the uploaded ISIC pipeline:
        image, label, imageA1, imageA2

    Falls back to base image if a future dataset version returns fewer views.
    """
    if len(batch) < 2:
        raise RuntimeError("Training batch must contain at least image and label.")

    image = batch[0]
    label = batch[1]

    if train_view == "A1" and len(batch) >= 3:
        image = batch[2]
    elif train_view == "A2" and len(batch) >= 4:
        image = batch[3]

    return image, label


def foreground_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    foreground_class: int = 1,
    eps: float = 1e-7,
):
    pred = torch.argmax(logits, dim=1)
    target = _normalize_label(target)

    p = (pred == foreground_class).float()
    t = (target == foreground_class).float()

    inter = (p * t).sum(dim=(1, 2))
    p_sum = p.sum(dim=(1, 2))
    t_sum = t.sum(dim=(1, 2))
    union = p_sum + t_sum - inter

    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    iou = (inter + eps) / (union + eps)

    return float(dice.mean().item()), float(iou.mean().item())


# =============================================================================
# Non-overwriting state_dict save
# =============================================================================

def _unique_save_path(save_dir: str, filename: str) -> str:
    """
    Guarantee that an existing .pt file is never overwritten.

    If the desired filename already exists:
        xxx.pt -> xxx_v2.pt -> xxx_v3.pt -> ...
    """
    path = Path(save_dir) / filename

    if not path.exists():
        return str(path)

    stem = path.stem
    suffix = path.suffix
    version = 2

    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return str(candidate)
        version += 1


def save_state_dict_unique(
    model: nn.Module,
    save_dir: str,
    filename: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = _unique_save_path(save_dir, filename)

    # Same content style as train_adamix_mt.py:
    # save model.state_dict(), not a checkpoint dictionary.
    torch.save(model.state_dict(), path)

    return path


# =============================================================================
# CSV logging with Python standard library
# =============================================================================

def write_dict_rows_csv(
    csv_path: str,
    rows,
    fieldnames,
) -> None:
    """
    Write a list of dictionaries to CSV using only Python's standard library.

    This uses only Python's built-in csv module, so the existing
    Mamba/PyTorch environment needs no extra package installation or upgrade.
    """
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_dice",
    "train_iou",
    "val_loss",
    "val_dice",
    "val_iou",
    "best_val_loss",
    "best_epoch",
    "epoch_minutes",
]

SAVED_MODEL_FIELDS = [
    "epoch",
    "reason",
    "val_loss",
    "val_dice",
    "val_iou",
    "path",
]


# =============================================================================
# Train / validate
# =============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    args,
    epoch,
    writer,
    global_step,
):
    """
    AMP-safe training loop.

    Numerical-failure policy:
    1) forward produces/raises NaN/Inf-related error -> skip this batch;
    2) loss is NaN/Inf -> skip this batch;
    3) backward gradients contain NaN/Inf -> let GradScaler skip optimizer.step(),
       reduce the loss scale, and then continue with the next batch;
    4) finite gradients -> normal gradient clipping + optimizer update.

    We intentionally DO NOT use torch.nan_to_num() on model features, because
    replacing broken internal activations and continuing backpropagation can hide
    the real numerical problem and produce meaningless gradients.
    """
    model.train()

    losses = []
    dices = []
    ious = []

    skipped_forward = 0
    skipped_loss = 0
    skipped_grad = 0

    use_amp = bool(args.use_amp) and torch.cuda.is_available()

    def _looks_like_numerical_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        if "nan" in msg:
            return True
        if "non-finite" in msg or "nonfinite" in msg or "not finite" in msg:
            return True
        if "overflow" in msg or "infinite" in msg:
            return True
        # Match the standalone abbreviation "Inf" without accidentally
        # treating unrelated words such as "inference" as numerical errors.
        padded = " " + msg.replace(",", " ").replace(".", " ").replace(":", " ") + " "
        return " inf " in padded

    for idx, batch in enumerate(loader):
        image, label = _select_train_view(batch, args.train_view)

        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        target = _normalize_label(label)

        optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------------------
        # 1. Forward: keep AMP, but skip only numerical failures.
        #    Other RuntimeErrors (e.g. CUDA OOM, shape mismatch) still raise.
        # -------------------------------------------------------------
        try:
            with autocast(enabled=use_amp):
                logits = model(image)
        except (ValueError, RuntimeError) as exc:
            if not _looks_like_numerical_error(exc):
                raise

            skipped_forward += 1
            print(
                "[SKIP][FORWARD] "
                f"epoch={epoch}/{args.num_epochs} "
                f"iter={idx}/{len(loader)} | {exc}"
            )
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        # The loss is evaluated from FP32 logits, as in the original script.
        logits = logits.float()
        loss = criterion(logits, target)

        # -------------------------------------------------------------
        # 2. Loss guard.
        # -------------------------------------------------------------
        if not torch.isfinite(loss).item():
            skipped_loss += 1
            print(
                "[SKIP][LOSS] "
                f"epoch={epoch}/{args.num_epochs} "
                f"iter={idx}/{len(loader)} | loss={loss.detach().item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        # -------------------------------------------------------------
        # 3. AMP backward.
        # -------------------------------------------------------------
        scaler.scale(loss).backward()

        # Always unscale here. Besides making gradient clipping correct,
        # GradScaler records found_inf during this operation and can therefore
        # skip optimizer.step() safely when overflow occurs.
        scaler.unscale_(optimizer)

        # Compute/clip the total gradient norm. With error_if_nonfinite=False,
        # this call never terminates training merely because the norm is Inf/NaN.
        # GradScaler has already recorded found_inf during unscale_().
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
                error_if_nonfinite=False,
            )
        else:
            # When clipping is disabled, build one total norm only for numerical
            # monitoring. This branch is rarely used with the current default.
            grad_sq_sum = torch.zeros((), device=image.device, dtype=torch.float32)
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().float()
                    grad_sq_sum = grad_sq_sum + torch.sum(g * g)
            grad_norm = torch.sqrt(grad_sq_sum)

        grad_is_finite = bool(torch.isfinite(grad_norm).item())
        scale_before = float(scaler.get_scale())

        # GradScaler will skip the real optimizer.step() automatically if
        # unscale_() found Inf/NaN gradients.
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())

        if not grad_is_finite:
            skipped_grad += 1
            print(
                "[SKIP][GRAD] "
                f"epoch={epoch}/{args.num_epochs} "
                f"iter={idx}/{len(loader)} | "
                f"grad_norm={float(grad_norm.item())} | "
                f"AMP scale {scale_before:g} -> {scale_after:g}"
            )

            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        # -------------------------------------------------------------
        # 5. Metrics are accumulated only for batches that reached a normal
        #    finite-gradient update path.
        # -------------------------------------------------------------
        dice, iou = foreground_metrics(
            logits.detach(),
            target,
            foreground_class=1,
        )

        losses.append(float(loss.item()))
        dices.append(dice)
        ious.append(iou)

        writer.add_scalar("train/loss_iter", loss.item(), global_step)
        writer.add_scalar("train/dice_iter", dice, global_step)
        writer.add_scalar("train/iou_iter", iou, global_step)
        writer.add_scalar("train/amp_scale", scale_after, global_step)

        if grad_norm is not None:
            writer.add_scalar(
                "train/grad_norm_before_clip",
                float(grad_norm.item()),
                global_step,
            )

        if idx % args.log_freq == 0:
            grad_text = (
                f"{float(grad_norm.item()):.4e}"
                if grad_norm is not None
                else "disabled"
            )
            print(
                "Train | epoch {}/{} | iter {}/{} | "
                "loss {:.4f} | dice {:.4f} | iou {:.4f} | "
                "grad_norm {} | amp_scale {:.0f}".format(
                    epoch,
                    args.num_epochs,
                    idx,
                    len(loader),
                    loss.item(),
                    dice,
                    iou,
                    grad_text,
                    scale_after,
                )
            )

        global_step += 1

    # -------------------------------------------------------------
    # Epoch-level skip statistics.
    # -------------------------------------------------------------
    skipped_total = skipped_forward + skipped_loss + skipped_grad
    total_batches = len(loader)
    skip_ratio = skipped_total / max(total_batches, 1)

    writer.add_scalar("train/skipped_forward_epoch", skipped_forward, epoch)
    writer.add_scalar("train/skipped_loss_epoch", skipped_loss, epoch)
    writer.add_scalar("train/skipped_grad_epoch", skipped_grad, epoch)
    writer.add_scalar("train/skipped_total_epoch", skipped_total, epoch)
    writer.add_scalar("train/skip_ratio_epoch", skip_ratio, epoch)

    print(
        "[AMP-SAFE SUMMARY] "
        f"epoch={epoch}/{args.num_epochs} | "
        f"valid={len(losses)}/{total_batches} | "
        f"skip_forward={skipped_forward} | "
        f"skip_loss={skipped_loss} | "
        f"skip_grad={skipped_grad} | "
        f"skip_total={skipped_total} ({100.0 * skip_ratio:.2f}%)"
    )

    if len(losses) == 0:
        raise RuntimeError(
            "No valid training batch remained in this epoch. "
            "The model is numerically unstable; do not continue by silently "
            "skipping the entire epoch."
        )

    # Frequent skipping is not stopped automatically, but make it very visible.
    if skip_ratio >= 0.05:
        print(
            "[WARNING] >= 5% of training batches were skipped in this epoch. "
            "Occasional AMP overflow can be tolerated, but this frequency "
            "indicates real numerical instability and should be investigated."
        )

    return (
        float(np.mean(losses)),
        float(np.mean(dices)),
        float(np.mean(ious)),
        global_step,
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    args,
    epoch,
    writer,
):
    model.eval()

    losses = []
    dices = []
    ious = []

    use_amp = bool(args.use_amp) and torch.cuda.is_available()

    for batch in loader:
        if len(batch) < 2:
            raise RuntimeError("Validation batch must contain image and label.")

        image = batch[0].to(device, non_blocking=True)
        label = batch[1].to(device, non_blocking=True)
        target = _normalize_label(label)

        with autocast(enabled=use_amp):
            logits = model(image)

        logits = logits.float()
        loss = criterion(logits, target)

        dice, iou = foreground_metrics(
            logits,
            target,
            foreground_class=1,
        )

        losses.append(loss.item())
        dices.append(dice)
        ious.append(iou)

    mean_loss = float(np.mean(losses))
    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))

    writer.add_scalar("val/loss", mean_loss, epoch)
    writer.add_scalar("val/dice", mean_dice, epoch)
    writer.add_scalar("val/iou", mean_iou, epoch)

    return mean_loss, mean_dice, mean_iou


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    seed_torch(args.seed)

    use_amp = bool(args.use_amp) and torch.cuda.is_available()

    run_name = (
        f"img{args.image_size}"
        f"_bs{args.batch_size}"
        f"_lr{args.learning_rate:g}"
        f"_full_supervised"
    )

    project_path = os.path.join(args.project, run_name)
    weights_path = os.path.join(project_path, "weights")

    os.makedirs(project_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    tb_dir = os.path.join(
        project_path,
        "tensorboard_" + time.strftime("%b%d_%H-%M-%S", time.localtime()),
    )
    writer = SummaryWriter(tb_dir)

    print("=" * 88)
    print("Supervised SARMambaMGEFUNet on ISIC")
    print("=" * 88)
    print("PyTorch              :", torch.__version__)
    print("CUDA available       :", torch.cuda.is_available())
    print("Device               :", device)
    if torch.cuda.is_available():
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Data path            :", args.data_path)
    print("Image size           :", args.image_size)
    print("Preprocess           : ISICDataset resize-first; train augmentation=True")
    print("Training protocol    : FULL supervised (all labeled training samples)")
    print("Train view           :", args.train_view)
    print("Batch size           :", args.batch_size)
    print("Epochs               :", args.num_epochs)
    print("Learning rate        :", args.learning_rate)
    print("AMP                  :", use_amp)
    print("Weights directory    :", weights_path)
    print("Overwrite policy     : NEVER overwrite .pt files")
    print("=" * 88)

    train_loader, val_loader = get_data(args)

    model = build_model(args).to(device)

    criterion = DSCLossH(
        num_classes=args.num_classes,
        device=device,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    best_epoch = 0
    global_step = 0

    history = []
    saved_records = []

    since = time.time()

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice, train_iou, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )

        val_loss, val_dice, val_iou = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            args=args,
            epoch=epoch,
            writer=writer,
        )

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/dice_epoch", train_dice, epoch)
        writer.add_scalar("train/iou_epoch", train_iou, epoch)

        epoch_minutes = (time.time() - epoch_start) / 60.0
        writer.add_scalar("time/epoch_minutes", epoch_minutes, epoch)

        # -------------------------------------------------------------
        # Save condition 1: improved validation loss
        # Every improvement gets a NEW unique .pt file.
        # -------------------------------------------------------------
        if val_loss <= best_loss:
            best_loss = val_loss
            best_epoch = epoch

            filename = (
                f"best_epoch_{epoch:03d}"
                f"_valloss_{val_loss:.8f}"
                f"_dice_{val_dice:.6f}.pt"
            )

            path = save_state_dict_unique(
                model=model,
                save_dir=weights_path,
                filename=filename,
            )

            print("[SAVE][BEST]", path)

            saved_records.append({
                "epoch": epoch,
                "reason": "best_val_loss",
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "path": path,
            })

        # -------------------------------------------------------------
        # Save condition 2: periodic snapshot
        # Also unique, never overwritten.
        # -------------------------------------------------------------
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            filename = (
                f"epoch_{epoch:03d}"
                f"_valloss_{val_loss:.8f}"
                f"_dice_{val_dice:.6f}.pt"
            )

            path = save_state_dict_unique(
                model=model,
                save_dir=weights_path,
                filename=filename,
            )

            print("[SAVE][PERIODIC]", path)

            saved_records.append({
                "epoch": epoch,
                "reason": "periodic",
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "path": path,
            })

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "epoch_minutes": epoch_minutes,
        })

        # Update records every epoch so interruption does not lose the logs.
        # Uses only Python standard-library csv; no extra dependency.
        write_dict_rows_csv(
            os.path.join(project_path, "train_val_metrics.csv"),
            history,
            HISTORY_FIELDS,
        )

        write_dict_rows_csv(
            os.path.join(project_path, "saved_models.csv"),
            saved_records,
            SAVED_MODEL_FIELDS,
        )

        print(
            "Epoch {}/{} | "
            "train loss {:.4f} dice {:.4f} iou {:.4f} | "
            "val loss {:.4f} dice {:.4f} iou {:.4f} | "
            "best {:.4f} @ epoch {} | {:.2f} min".format(
                epoch,
                args.num_epochs,
                train_loss,
                train_dice,
                train_iou,
                val_loss,
                val_dice,
                val_iou,
                best_loss,
                best_epoch,
                epoch_minutes,
            )
        )

    # Final snapshot, also unique.
    final_val_loss = history[-1]["val_loss"]
    final_val_dice = history[-1]["val_dice"]

    final_name = (
        f"final_epoch_{args.num_epochs:03d}"
        f"_valloss_{final_val_loss:.8f}"
        f"_dice_{final_val_dice:.6f}.pt"
    )

    final_path = save_state_dict_unique(
        model=model,
        save_dir=weights_path,
        filename=final_name,
    )

    print("[SAVE][FINAL]", final_path)

    saved_records.append({
        "epoch": args.num_epochs,
        "reason": "final",
        "val_loss": history[-1]["val_loss"],
        "val_dice": history[-1]["val_dice"],
        "val_iou": history[-1]["val_iou"],
        "path": final_path,
    })

    write_dict_rows_csv(
        os.path.join(project_path, "saved_models.csv"),
        saved_records,
        SAVED_MODEL_FIELDS,
    )

    # Plot directly from the in-memory Python list of dictionaries.
    # No external DataFrame library is needed.
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    train_dices = [row["train_dice"] for row in history]
    val_dices = [row["val_dice"] for row in history]

    plt.figure()
    plt.title("Loss During Training and Validation")
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_path, "train_val_loss.png"))
    plt.close()

    plt.figure()
    plt.title("Foreground Dice")
    plt.plot(epochs, train_dices, label="Train")
    plt.plot(epochs, val_dices, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_path, "train_val_dice.png"))
    plt.close()

    writer.close()

    elapsed = time.time() - since

    print("=" * 88)
    print("Training finished")
    print("Best epoch        :", best_epoch)
    print("Best val loss     :", best_loss)
    print(
        "Elapsed           : {:.0f}m {:.0f}s".format(
            elapsed // 60,
            elapsed % 60,
        )
    )
    print("Project path      :", project_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
