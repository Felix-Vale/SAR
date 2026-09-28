from __future__ import annotations

"""
SAR-Mamba + SIA + MGEF Nested U-Net
===================================

This file is the top-level segmentation network that connects the three
standalone modules:

    sar_mamba.py  -> deep structure-adaptive state-space modeling
    sia.py        -> multi-stage structural information aggregation
    mgef.py       -> decoder-side structure-guided skip evidence fusion

The topology follows the current reconstruction specification:

Encoder (five scales)
---------------------
    X0_0 : stride 1   residual convolution
    X1_0 : stride 2   residual convolution
    X2_0 : stride 4   ConvRes -> SAR-Mamba -> ConvRes
    X3_0 : stride 8   residual convolution
    X4_0 : stride 16  ConvRes -> SAR-Mamba -> ConvFFN -> SAR-Mamba

Structural interface
--------------------
    stride-4 SAR  -> c4,  A4
    last stride-16 SAR -> c16, A16
    SIA(c4, A4, c16, A16) -> C, A

Decoder
-------
Two nested reconstruction depths are retained.  Every decoder node performs:

    upsample -> MGEF(D, S, C, A) -> decoder refinement

For the second reconstruction depth, the skip representation is:

    concat(encoder feature, first-depth reconstructed feature)
        -> 1x1 alignment -> MGEF

The old external GTX, old VMamba wrapper, LGFS, and the special stride-8
cross-attention are intentionally not used in this network.

Normal interface
----------------
    model = SARMambaMGEFUNet(cfg)
    logits = model(x)                     # [B, num_classes, H, W]

Optional analysis interface
---------------------------
    logits, aux = model(x, return_aux=True)

The optional ``aux`` dictionary exposes SAR/SIA information required for token
regularization, route analysis, and later ablation experiments without changing
the normal training interface.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sar_mamba import SARMambaBlock
from model.SIA import StructuralInformationAggregation
from model.MGEF import MambaGuidedEvidenceFusion, SkipFeatureAlign


# =============================================================================
# Standalone-module imports
# =============================================================================
# 1) package-relative import: preferred when this file is in Our_model/model/
# 2) absolute project import: compatible with the user's current project style
# 3) local import: convenient for directly running this file as a script
# try:
#     from .sar_mamba import SARMambaBlock
#     from .sia import StructuralInformationAggregation
#     from .mgef import MambaGuidedEvidenceFusion, SkipFeatureAlign
# except ImportError:
#     try:
#         from Our_model.model.sar_mamba import SARMambaBlock
#         from Our_model.model.sia import StructuralInformationAggregation
#         from Our_model.model.mgef import MambaGuidedEvidenceFusion, SkipFeatureAlign
#     except ImportError:
#         from sar_mamba import SARMambaBlock
#         from sia import StructuralInformationAggregation
#         from mgef import MambaGuidedEvidenceFusion, SkipFeatureAlign


# =============================================================================
# Basic CNN blocks
# =============================================================================


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(int(max_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvResBlock(nn.Module):
    """Two-convolution residual block with GroupNorm.

    GroupNorm is retained because the current ISIC training setup uses small
    mini-batches, where BatchNorm can be less stable.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_ch)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ConvFFN2D(nn.Module):
    """Local convolutional FFN used between the two stride-16 SAR blocks."""

    def __init__(self, channels: int, expand: int = 4) -> None:
        super().__init__()
        hidden = int(channels * expand)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm1 = _group_norm(hidden)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.norm2 = _group_norm(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm3 = _group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.pw1(x)))
        h = self.act(self.norm2(self.dw(h)))
        h = self.norm3(self.pw2(h))
        return self.act(x + h)


class DecoderRefineBlock(nn.Module):
    """Post-MGEF decoder refinement.

    MGEF decides which skip evidence is injected.  This block only refines the
    resulting decoder representation, keeping the two responsibilities separate.
    """

    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        self.blocks = nn.Sequential(
            *[ConvResBlock(channels, channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SARMambaMGEFConfig:
    # Input / output
    in_ch: int = 3
    num_classes: int = 2
    channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512)

    # ------------------------------------------------------------------
    # SAR-Mamba
    # ------------------------------------------------------------------
    d_state: int = 16
    expand: float = 1.5
    conv_kernel: int = 3
    dt_rank: int = 4
    dt_min: float = 1e-4
    dt_max: float = 0.1

    token_dim: int = 96
    kmax: int = 8
    token_heads: int = 4
    token_pool_hw: Tuple[int, int] = (8, 8)
    token_mlp_ratio: float = 2.0
    token_tau: float = 0.5
    token_temperature: float = 1.0
    hard_token_inference: bool = False
    token_coverage_rho: float = 0.90

    route_topk: int = 2
    route_mode: Literal["soft", "topk_st", "topk_hard"] = "topk_st"
    sparse_route_train: bool = False
    sparse_route_eval: bool = True
    sar_drop_path: float = 0.0

    # ------------------------------------------------------------------
    # SIA
    # ------------------------------------------------------------------
    sia_hidden_dim: Optional[int] = None  # None -> token_dim
    sia_dropout: float = 0.0

    # ------------------------------------------------------------------
    # MGEF
    # ------------------------------------------------------------------
    mgef_directional_kernel: int = 5
    mgef_branch_ch: Optional[int] = None
    mgef_gate_hidden_dim: Optional[int] = None

    mgef_low_kernel: int = 5
    mgef_low_sigma: float = 1.0
    mgef_dog_kernel1: int = 3
    mgef_dog_sigma1: float = 0.8
    mgef_dog_kernel2: int = 5
    mgef_dog_sigma2: float = 1.6
    mgef_use_out_proj: bool = False

    # Decoder refinement after every MGEF node
    decoder_refine_blocks: int = 2


# =============================================================================
# Top-level segmentation network
# =============================================================================


class SARMambaMGEFUNet(nn.Module):
    """Five-scale SAR-Mamba + SIA + MGEF two-depth nested U-Net."""

    def __init__(self, cfg: Optional[SARMambaMGEFConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else SARMambaMGEFConfig()
        cfg = self.cfg

        if len(cfg.channels) != 5:
            raise ValueError(
                "channels must contain exactly five values for stride "
                "1/2/4/8/16 stages."
            )
        c1, c2, c3, c4, cb = cfg.channels

        # ------------------------------------------------------------------
        # Encoder: stride 1
        # ------------------------------------------------------------------
        self.enc1_a = ConvResBlock(cfg.in_ch, c1)
        self.enc1_b = ConvResBlock(c1, c1)
        self.down1 = Downsample(c1)

        # ------------------------------------------------------------------
        # Encoder: stride 2
        # ------------------------------------------------------------------
        self.enc2_a = ConvResBlock(c1, c2)
        self.enc2_b = ConvResBlock(c2, c2)
        self.down2 = Downsample(c2)

        # ------------------------------------------------------------------
        # Encoder: stride 4 = ConvRes -> SAR-Mamba -> ConvRes
        # ------------------------------------------------------------------
        self.enc4_pre = ConvResBlock(c2, c3)
        self.sar4 = self._make_sar_block(c3)
        self.enc4_post = ConvResBlock(c3, c3)
        self.down3 = Downsample(c3)

        # ------------------------------------------------------------------
        # Encoder: stride 8 = convolutional stage
        # ------------------------------------------------------------------
        self.enc8 = nn.Sequential(
            ConvResBlock(c3, c4),
            ConvResBlock(c4, c4),
        )
        self.down4 = Downsample(c4)

        # ------------------------------------------------------------------
        # Bottleneck: stride 16 = SAR -> local ConvFFN -> SAR
        # ------------------------------------------------------------------
        self.to_bottleneck = ConvResBlock(c4, cb)
        self.sar16_1 = self._make_sar_block(cb)
        self.bottleneck_ffn = ConvFFN2D(cb, expand=4)
        self.sar16_2 = self._make_sar_block(cb)

        # ------------------------------------------------------------------
        # SIA: fixed-dimensional encoder -> decoder structural interface
        # ------------------------------------------------------------------
        self.sia = StructuralInformationAggregation(
            struct_dim=cfg.token_dim,
            num_routes=4,
            hidden_dim=cfg.sia_hidden_dim,
            dropout=cfg.sia_dropout,
        )

        # ------------------------------------------------------------------
        # Scale-wise upsampling.
        # These modules are shared by the two reconstruction depths, preserving
        # the lightweight topology of the original implementation while the
        # MGEF/refinement nodes themselves remain independent.
        # ------------------------------------------------------------------
        self.up_16_to_8 = Upsample(cb, c4)
        self.up_8_to_4 = Upsample(c4, c3)
        self.up_4_to_2 = Upsample(c3, c2)
        self.up_2_to_1 = Upsample(c2, c1)

        # ------------------------------------------------------------------
        # Reconstruction depth 1
        # ------------------------------------------------------------------
        self.mgef_8_1 = self._make_mgef(dec_ch=c4, skip_ch=c4)
        self.mgef_4_1 = self._make_mgef(dec_ch=c3, skip_ch=c3)
        self.mgef_2_1 = self._make_mgef(dec_ch=c2, skip_ch=c2)
        self.mgef_1_1 = self._make_mgef(dec_ch=c1, skip_ch=c1)

        self.dec_8_1 = DecoderRefineBlock(c4, cfg.decoder_refine_blocks)
        self.dec_4_1 = DecoderRefineBlock(c3, cfg.decoder_refine_blocks)
        self.dec_2_1 = DecoderRefineBlock(c2, cfg.decoder_refine_blocks)
        self.dec_1_1 = DecoderRefineBlock(c1, cfg.decoder_refine_blocks)

        # ------------------------------------------------------------------
        # Reconstruction depth 2
        # Raw encoder feature + first-depth reconstructed feature -> 1x1 align
        # ------------------------------------------------------------------
        self.align_8_2 = SkipFeatureAlign(2 * c4, c4)
        self.align_4_2 = SkipFeatureAlign(2 * c3, c3)
        self.align_2_2 = SkipFeatureAlign(2 * c2, c2)
        self.align_1_2 = SkipFeatureAlign(2 * c1, c1)

        self.mgef_8_2 = self._make_mgef(dec_ch=c4, skip_ch=c4)
        self.mgef_4_2 = self._make_mgef(dec_ch=c3, skip_ch=c3)
        self.mgef_2_2 = self._make_mgef(dec_ch=c2, skip_ch=c2)
        self.mgef_1_2 = self._make_mgef(dec_ch=c1, skip_ch=c1)

        self.dec_8_2 = DecoderRefineBlock(c4, cfg.decoder_refine_blocks)
        self.dec_4_2 = DecoderRefineBlock(c3, cfg.decoder_refine_blocks)
        self.dec_2_2 = DecoderRefineBlock(c2, cfg.decoder_refine_blocks)
        self.dec_1_2 = DecoderRefineBlock(c1, cfg.decoder_refine_blocks)

        # Final segmentation head from the highest-resolution second-depth node.
        self.head = nn.Conv2d(c1, cfg.num_classes, kernel_size=1, bias=True)

    # ----------------------------------------------------------------------
    # Module factories
    # ----------------------------------------------------------------------
    def _make_sar_block(self, channels: int) -> SARMambaBlock:
        cfg = self.cfg
        return SARMambaBlock(
            d_model=channels,
            d_state=cfg.d_state,
            expand=cfg.expand,
            conv_kernel=cfg.conv_kernel,
            dt_rank=cfg.dt_rank,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            token_dim=cfg.token_dim,
            kmax=cfg.kmax,
            token_heads=cfg.token_heads,
            token_pool_hw=cfg.token_pool_hw,
            token_mlp_ratio=cfg.token_mlp_ratio,
            token_tau=cfg.token_tau,
            token_temperature=cfg.token_temperature,
            hard_token_inference=cfg.hard_token_inference,
            token_coverage_rho=cfg.token_coverage_rho,
            route_topk=cfg.route_topk,
            route_mode=cfg.route_mode,
            sparse_route_train=cfg.sparse_route_train,
            sparse_route_eval=cfg.sparse_route_eval,
            drop_path=cfg.sar_drop_path,
        )

    def _make_mgef(self, dec_ch: int, skip_ch: int) -> MambaGuidedEvidenceFusion:
        cfg = self.cfg
        return MambaGuidedEvidenceFusion(
            dec_ch=dec_ch,
            skip_ch=skip_ch,
            struct_dim=cfg.token_dim,
            num_routes=4,
            directional_kernel=cfg.mgef_directional_kernel,
            branch_ch=cfg.mgef_branch_ch,
            gate_hidden_dim=cfg.mgef_gate_hidden_dim,
            low_kernel=cfg.mgef_low_kernel,
            low_sigma=cfg.mgef_low_sigma,
            dog_kernel1=cfg.mgef_dog_kernel1,
            dog_sigma1=cfg.mgef_dog_sigma1,
            dog_kernel2=cfg.mgef_dog_kernel2,
            dog_sigma2=cfg.mgef_dog_sigma2,
            use_out_proj=cfg.mgef_use_out_proj,
        )

    # ----------------------------------------------------------------------
    # Training-control helpers
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def set_token_temperature(self, temperature: float) -> None:
        """Apply one token-mask temperature to all three SAR blocks."""
        self.sar4.set_token_temperature(temperature)
        self.sar16_1.set_token_temperature(temperature)
        self.sar16_2.set_token_temperature(temperature)

    def set_route_mode(
        self,
        mode: Literal["soft", "topk_st", "topk_hard"],
    ) -> None:
        """Apply one route policy to all three SAR blocks."""
        self.sar4.set_route_mode(mode)
        self.sar16_1.set_route_mode(mode)
        self.sar16_2.set_route_mode(mode)

    @staticmethod
    def token_budget_loss(aux: Dict[str, Any]) -> torch.Tensor:
        """Average token-budget regularizer over the three SAR blocks.

        This helper is optional.  It expects ``aux`` returned by
        ``forward(..., return_aux=True)`` and can later be used as:

            logits, aux = model(x, return_aux=True)
            loss = seg_loss + lambda_token * model.token_budget_loss(aux)
        """
        sar_aux = aux.get("sar", None)
        if sar_aux is None:
            raise KeyError("aux does not contain SAR information.")
        losses = [
            sar_aux["stride4"]["token_budget_loss"],
            sar_aux["stride16_block1"]["token_budget_loss"],
            sar_aux["stride16_block2"]["token_budget_loss"],
        ]
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------------
    # Shape helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor,
        multiple: int = 16,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0), (
            pad_h,
            pad_w,
        )

    @staticmethod
    def _crop_back(x: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = orig_hw
        return x[:, :, :h, :w].contiguous()

    # ----------------------------------------------------------------------
    # MGEF dispatch helper
    # ----------------------------------------------------------------------
    @staticmethod
    def _run_mgef(
        module: MambaGuidedEvidenceFusion,
        D: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        A: torch.Tensor,
        collect_aux: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if collect_aux:
            out, aux = module(D, S, C, A, return_aux=True)
            return out, aux
        return module(D, S, C, A), None

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
        return_mgef_aux: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Any]],
    ]:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}.")
        if x.shape[1] != self.cfg.in_ch:
            raise ValueError(
                f"Expected in_ch={self.cfg.in_ch}, got input channels={x.shape[1]}."
            )

        want_aux = bool(return_aux or return_mgef_aux)
        orig_hw = (x.shape[-2], x.shape[-1])
        x, _ = self._pad_to_multiple(x, multiple=16)

        # ==================================================================
        # Encoder
        # ==================================================================

        # X0_0: stride 1
        X0_0 = self.enc1_b(self.enc1_a(x))

        # X1_0: stride 2
        X1_0 = self.enc2_b(self.enc2_a(self.down1(X0_0)))

        # X2_0: stride 4, one SAR-Mamba
        X2_pre = self.enc4_pre(self.down2(X1_0))
        X2_sar, c4, A4, aux4 = self.sar4(X2_pre)
        X2_0 = self.enc4_post(X2_sar)


        # X2_pre = self.enc4_pre(self.down2(X1_0))
        # with torch.cuda.amp.autocast(enabled=False):
        #     X2_sar, c4, A4, aux4 = self.sar4(X2_pre.float())
        # X2_0 = self.enc4_post(X2_sar)




        # X3_0: stride 8, convolutional stage
        X3_0 = self.enc8(self.down3(X2_0))

        # X4_0: stride 16, SAR -> local FFN -> SAR
        X4_pre = self.to_bottleneck(self.down4(X3_0))
        X4_a, _c16_1, _A16_1, aux16_1 = self.sar16_1(X4_pre)
        X4_f = self.bottleneck_ffn(X4_a)

        # Numerical-stability guard:
        # If this fires, the non-finite value was already produced upstream
        # (sar16_1 / bottleneck_ffn), rather than by sar16_2 itself.
        if not torch.isfinite(X4_f).all():
            raise ValueError(
                "X4_f contains NaN or Inf before sar16_2."
            )

        # Only the final stride-16 SAR-Mamba block runs in FP32.
        # The outer training autocast remains enabled, so the rest of the
        # network still benefits from AMP.
        with torch.cuda.amp.autocast(enabled=False):
            X4_0, c16, A16, aux16_2 = self.sar16_2(
                X4_f.float()
            )


        # X4_pre = self.to_bottleneck(self.down4(X3_0))
        # with torch.cuda.amp.autocast(enabled=False):
        #     X4_a, _c16_1, _A16_1, aux16_1 = self.sar16_1(
        #         X4_pre.float()
        #     )
        # X4_f = self.bottleneck_ffn(X4_a)
        # with torch.cuda.amp.autocast(enabled=False):
        #     X4_0, c16, A16, aux16_2 = self.sar16_2(
        #         X4_f.float()
        #     )









        # ==================================================================
        # SIA: only stride-4 and the LAST stride-16 SAR outputs are exposed
        # to the decoder.  The first bottleneck block remains internal.
        # ==================================================================
        if want_aux:
            C, A, sia_aux = self.sia(c4, A4, c16, A16, return_aux=True)
        else:
            C, A = self.sia(c4, A4, c16, A16)
            sia_aux = None

        # ==================================================================
        # Reconstruction depth 1
        # ==================================================================
        mgef_aux: Dict[str, Dict[str, torch.Tensor]] = {}

        D3 = self.up_16_to_8(X4_0)
        D3, a = self._run_mgef(self.mgef_8_1, D3, X3_0, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride8_depth1"] = a
        X3_1 = self.dec_8_1(D3)

        D2 = self.up_8_to_4(X3_1)
        D2, a = self._run_mgef(self.mgef_4_1, D2, X2_0, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride4_depth1"] = a
        X2_1 = self.dec_4_1(D2)

        D1 = self.up_4_to_2(X2_1)
        D1, a = self._run_mgef(self.mgef_2_1, D1, X1_0, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride2_depth1"] = a
        X1_1 = self.dec_2_1(D1)

        D0 = self.up_2_to_1(X1_1)
        D0, a = self._run_mgef(self.mgef_1_1, D0, X0_0, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride1_depth1"] = a
        X0_1 = self.dec_1_1(D0)

        # ==================================================================
        # Reconstruction depth 2
        # ==================================================================

        # stride 8: encoder X3_0 + first-depth X3_1
        S3_2 = self.align_8_2(torch.cat([X3_0, X3_1], dim=1))
        D3b = self.up_16_to_8(X4_0)
        D3b, a = self._run_mgef(self.mgef_8_2, D3b, S3_2, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride8_depth2"] = a
        X3_2 = self.dec_8_2(D3b)

        # stride 4: encoder X2_0 + first-depth X2_1
        S2_2 = self.align_4_2(torch.cat([X2_0, X2_1], dim=1))
        D2b = self.up_8_to_4(X3_2)
        D2b, a = self._run_mgef(self.mgef_4_2, D2b, S2_2, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride4_depth2"] = a
        X2_2 = self.dec_4_2(D2b)

        # stride 2: encoder X1_0 + first-depth X1_1
        S1_2 = self.align_2_2(torch.cat([X1_0, X1_1], dim=1))
        D1b = self.up_4_to_2(X2_2)
        D1b, a = self._run_mgef(self.mgef_2_2, D1b, S1_2, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride2_depth2"] = a
        X1_2 = self.dec_2_2(D1b)

        # stride 1: encoder X0_0 + first-depth X0_1
        S0_2 = self.align_1_2(torch.cat([X0_0, X0_1], dim=1))
        D0b = self.up_2_to_1(X1_2)
        D0b, a = self._run_mgef(self.mgef_1_2, D0b, S0_2, C, A, return_mgef_aux)
        if a is not None:
            mgef_aux["stride1_depth2"] = a
        X0_2 = self.dec_1_2(D0b)

        # Final prediction and removal of padding.
        logits = self.head(X0_2)
        logits = self._crop_back(logits, orig_hw)

        if not want_aux:
            return logits

        aux_out: Dict[str, Any] = {
            "sar": {
                "stride4": aux4,
                "stride16_block1": aux16_1,
                "stride16_block2": aux16_2,
            },
            "sia": sia_aux,
            "structural_condition": C,
            "route_profile": A,
        }

        if return_mgef_aux:
            aux_out["mgef"] = mgef_aux

        return logits, aux_out


# =============================================================================
# Convenient aliases / factory
# =============================================================================

# Concise alias used in experiment scripts.
ProposedSegNet = SARMambaMGEFUNet


def build_sar_mgef_unet(**cfg_kwargs: Any) -> SARMambaMGEFUNet:
    """Build the network directly from config keyword arguments."""
    return SARMambaMGEFUNet(SARMambaMGEFConfig(**cfg_kwargs))


__all__ = [
    "SARMambaMGEFConfig",
    "SARMambaMGEFUNet",
    "ProposedSegNet",
    "build_sar_mgef_unet",
]


if __name__ == "__main__":
    # Small standalone smoke test.  It deliberately uses a compact channel
    # setting so the pure-PyTorch selective-scan fallback remains practical on
    # a CPU-only machine.  Real experiments should use the default channels and
    # the mamba_ssm CUDA backend.
    torch.manual_seed(0)

    cfg = SARMambaMGEFConfig(
        in_ch=3,
        num_classes=2,
        channels=(16, 32, 64, 128, 256),
        d_state=8,
        token_dim=32,
        kmax=4,
        token_heads=4,
        token_pool_hw=(4, 4),
        route_topk=2,
        route_mode="topk_st",
        decoder_refine_blocks=1,
    )

    model = SARMambaMGEFUNet(cfg)
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y, aux = model(x, return_aux=True)

    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("C     :", tuple(aux["structural_condition"].shape))
    print("A     :", tuple(aux["route_profile"].shape))
    print("A sum :", aux["route_profile"].sum(dim=-1))
