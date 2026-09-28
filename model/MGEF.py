from __future__ import annotations

"""
Standalone Mamba-Guided Evidence Fusion (MGEF)
================================================

This module is designed to be imported by a U-Net style decoder and used
*together with* the structural interface produced by SAR-Mamba + SIA.

Expected upstream interfaces
----------------------------
SAR-Mamba:
    Y_s, c_s, A_s, aux_s = sar(x_s)

SIA:
    C, A = sia(c4, A4, c16, A16)

where
    C : [B, struct_dim]      unified structural representation (default 96)
    A : [B, 4]               soft route profile in [LR, RL, TB, BT] order

MGEF interface
--------------
For one decoder node:

    D_out = mgef(D, S, C, A)

where
    D : [B, dec_ch,  H, W]   current decoder feature after upsampling
    S : [B, skip_ch, H, W]   encoder/nested skip feature
    C : [B, struct_dim]      SIA structural condition
    A : [B, 4]               SIA route profile [LR, RL, TB, BT]

and
    D_out : [B, dec_ch, H, W]

The decoder refinement block remains outside MGEF:

    D = up(previous_feature)
    D = mgef(D, S, C, A)
    X = decoder_refine(D)

For the second nested reconstruction depth, concatenate the original encoder
feature and the first-depth reconstructed feature, align them with a 1x1
convolution, then pass the aligned feature to MGEF:

    S2 = align(torch.cat([S_encoder, X_depth1], dim=1))
    D2 = up(previous_depth2_feature)
    X2 = decoder_refine(mgef_depth2(D2, S2, C, A))

Core design
-----------
1) C-guided skip recalibration
       S_tilde = S * (1 + gamma(C)) + beta(C)

2) A-guided directional local evidence
       a_H = a_LR + a_RL
       a_V = a_TB + a_BT
       F_dir = a_H * DWConv_1xk(S_tilde)
             + a_V * DWConv_kx1(S_tilde)

3) Fixed frequency decomposition
       F_L  = Gaussian(S_tilde)
       F_HF = Gaussian_sigma1(S_tilde) - Gaussian_sigma2(S_tilde)

4) High-frequency reliability filtering
       q = concat(GAP(D), GAP(S_tilde), C)
       R_HF = sigmoid(MLP_hf(q))
       F_HF_filtered = F_HF * R_HF

5) Lightweight global evidence
       GAP(S_tilde) -> channel gate -> reweight -> 1x1 projection

6) Multi-evidence fusion
       directional/global/HF/LF -> per-branch 1x1 projection -> concat
       -> 1x1 fusion -> GroupNorm -> GELU

7) Overall skip gate + residual injection
       g = sigmoid(MLP_skip(q))
       D_out = D + out_proj(F_fuse * g)

Important constraints
---------------------
- MGEF intentionally contains NO Mamba/SSM block.  Long-range state-space
  modeling is performed by deep SAR-Mamba blocks; MGEF only reuses C and A
  to process high-resolution skip evidence.
- A is used only by the directional branch.  It is intentionally not inserted
  into the HF reliability gate or the overall skip gate.
- C is used for skip recalibration, HF reliability estimation, and overall
  skip gating.
- The four evidence branches are fused by concat + 1x1 convolution.  There is
  no additional branch-wise softmax router.
- No detach() is used.  Decoder loss can backpropagate through MGEF -> SIA ->
  SAR-Mamba.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utility layers
# =============================================================================


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Return a GroupNorm whose group count divides ``channels``."""
    groups = min(int(max_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a normalized 2-D Gaussian kernel on CPU."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)
    return kernel


class FixedGaussianBlurDW(nn.Module):
    """Fixed depthwise Gaussian filtering for NCHW feature maps."""

    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        kernel = _gaussian_kernel_2d(self.kernel_size, self.sigma)
        self.register_buffer(
            "kernel",
            kernel.view(1, 1, self.kernel_size, self.kernel_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}."
            )

        weight = self.kernel.to(device=x.device, dtype=x.dtype)
        weight = weight.expand(self.channels, 1, -1, -1).contiguous()
        return F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )


class DifferenceOfGaussiansDW(nn.Module):
    """Fixed depthwise Difference-of-Gaussians high-frequency extractor."""

    def __init__(
        self,
        channels: int,
        kernel_size1: int = 3,
        sigma1: float = 0.8,
        kernel_size2: int = 5,
        sigma2: float = 1.6,
    ) -> None:
        super().__init__()
        self.g1 = FixedGaussianBlurDW(channels, kernel_size1, sigma1)
        self.g2 = FixedGaussianBlurDW(channels, kernel_size2, sigma2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g1(x) - self.g2(x)


class ZeroInitAffineFromCondition(nn.Module):
    """Generate channel-wise affine parameters from structural condition C.

    The output projection is zero-initialized so MGEF starts with
    ``S_tilde == S`` at initialization.
    """

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.condition_dim)
        self.proj = nn.Linear(self.condition_dim, 2 * self.channels, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        affine = self.proj(self.norm(condition))
        gamma, beta = affine.chunk(2, dim=-1)
        return gamma, beta


class ChannelGateMLP(nn.Module):
    """Small MLP producing a sigmoid channel gate from a pooled control vector."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive.")

        hidden = int(hidden_dim or max(16, in_dim // 4))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, out_dim, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class SkipFeatureAlign(nn.Module):
    """1x1 channel alignment for second-depth nested skip features.

    Example
    -------
    ``concat([X_i_0, X_i_1])`` has 2*C channels.  Before feeding it to an
    MGEF configured with ``skip_ch=C`` use:

        S_i2 = align_i2(torch.cat([X_i_0, X_i_1], dim=1))
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"SkipFeatureAlign expected [B,{self.in_channels},H,W], "
                f"got {tuple(x.shape)}."
            )
        return self.proj(x)


# =============================================================================
# MGEF
# =============================================================================


class MambaGuidedEvidenceFusion(nn.Module):
    """Mamba-Guided Evidence Fusion for one decoder node.

    Parameters
    ----------
    dec_ch:
        Number of channels in the current decoder feature D.

    skip_ch:
        Number of channels in the skip feature S *after* any nested-skip
        alignment convolution.

    struct_dim:
        Dimension of SIA structural representation C.  Default 96.

    num_routes:
        Must be 4, with route order [LR, RL, TB, BT].

    directional_kernel:
        Odd kernel size k for 1xk and kx1 depthwise directional convolutions.
        Default 5.

    branch_ch:
        Projection width for each evidence branch before concatenation.  If
        omitted, uses max(dec_ch // 4, 16), matching the reconstruction spec.

    gate_hidden_dim:
        Hidden dimension shared as a *size choice* (not weights) by the HF and
        overall gate MLPs.  If None, a lightweight automatic value is used.

    low_kernel / low_sigma:
        Fixed Gaussian low-frequency filter parameters.

    dog_kernel1 / dog_sigma1 / dog_kernel2 / dog_sigma2:
        Fixed Difference-of-Gaussians high-frequency filter parameters.

    use_out_proj:
        If True, apply an additional 1x1 projection to gated fused evidence
        before residual addition.  Since F_fuse already has dec_ch channels,
        the minimal/default design uses Identity (False).
    """

    ROUTE_NAMES: Tuple[str, str, str, str] = ("LR", "RL", "TB", "BT")

    def __init__(
        self,
        dec_ch: int,
        skip_ch: int,
        *,
        struct_dim: int = 96,
        num_routes: int = 4,
        directional_kernel: int = 5,
        branch_ch: Optional[int] = None,
        gate_hidden_dim: Optional[int] = None,
        low_kernel: int = 5,
        low_sigma: float = 1.0,
        dog_kernel1: int = 3,
        dog_sigma1: float = 0.8,
        dog_kernel2: int = 5,
        dog_sigma2: float = 1.6,
        use_out_proj: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if dec_ch <= 0 or skip_ch <= 0 or struct_dim <= 0:
            raise ValueError("dec_ch, skip_ch and struct_dim must be positive.")
        if num_routes != 4:
            raise ValueError(
                "MGEF expects four SIA routes ordered as [LR, RL, TB, BT]."
            )
        if directional_kernel <= 0 or directional_kernel % 2 == 0:
            raise ValueError(
                "directional_kernel must be a positive odd integer, "
                f"got {directional_kernel}."
            )

        self.dec_ch = int(dec_ch)
        self.skip_ch = int(skip_ch)
        self.struct_dim = int(struct_dim)
        self.num_routes = int(num_routes)
        self.directional_kernel = int(directional_kernel)
        self.branch_ch = int(branch_ch or max(self.dec_ch // 4, 16))
        self.eps = float(eps)
        self.use_out_proj = bool(use_out_proj)

        # ------------------------------------------------------------------
        # 1) C-guided skip recalibration
        # ------------------------------------------------------------------
        self.skip_condition = ZeroInitAffineFromCondition(
            condition_dim=self.struct_dim,
            channels=self.skip_ch,
        )

        # ------------------------------------------------------------------
        # 2) A-guided directional local evidence
        # ------------------------------------------------------------------
        k = self.directional_kernel
        self.dir_horizontal = nn.Conv2d(
            self.skip_ch,
            self.skip_ch,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=self.skip_ch,
            bias=False,
        )
        self.dir_vertical = nn.Conv2d(
            self.skip_ch,
            self.skip_ch,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            groups=self.skip_ch,
            bias=False,
        )

        # ------------------------------------------------------------------
        # 3) Fixed frequency decomposition
        # ------------------------------------------------------------------
        self.low_filter = FixedGaussianBlurDW(
            self.skip_ch,
            kernel_size=low_kernel,
            sigma=low_sigma,
        )
        self.high_filter = DifferenceOfGaussiansDW(
            self.skip_ch,
            kernel_size1=dog_kernel1,
            sigma1=dog_sigma1,
            kernel_size2=dog_kernel2,
            sigma2=dog_sigma2,
        )

        # ------------------------------------------------------------------
        # 4) HF reliability and 7) overall skip gate
        # Both use q = [GAP(D), GAP(S_tilde), C], but with independent MLPs.
        # ------------------------------------------------------------------
        self.control_dim = self.dec_ch + self.skip_ch + self.struct_dim
        auto_gate_hidden = max(16, self.control_dim // 4)
        gate_hidden = int(gate_hidden_dim or auto_gate_hidden)

        self.hf_reliability = ChannelGateMLP(
            in_dim=self.control_dim,
            out_dim=self.skip_ch,
            hidden_dim=gate_hidden,
        )
        self.overall_skip_gate = ChannelGateMLP(
            in_dim=self.control_dim,
            out_dim=self.dec_ch,
            hidden_dim=gate_hidden,
        )

        # ------------------------------------------------------------------
        # 5) Lightweight global branch
        # ------------------------------------------------------------------
        self.global_gate = nn.Conv2d(
            self.skip_ch,
            self.skip_ch,
            kernel_size=1,
            bias=True,
        )
        self.global_proj = nn.Conv2d(
            self.skip_ch,
            self.skip_ch,
            kernel_size=1,
            bias=False,
        )

        # ------------------------------------------------------------------
        # 6) Per-branch projection + concat + 1x1 fusion
        # ------------------------------------------------------------------
        self.proj_dir = nn.Conv2d(
            self.skip_ch, self.branch_ch, kernel_size=1, bias=False
        )
        self.proj_global = nn.Conv2d(
            self.skip_ch, self.branch_ch, kernel_size=1, bias=False
        )
        self.proj_hf = nn.Conv2d(
            self.skip_ch, self.branch_ch, kernel_size=1, bias=False
        )
        self.proj_lf = nn.Conv2d(
            self.skip_ch, self.branch_ch, kernel_size=1, bias=False
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                4 * self.branch_ch,
                self.dec_ch,
                kernel_size=1,
                bias=False,
            ),
            _group_norm(self.dec_ch),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # 7) Residual evidence injection
        # ------------------------------------------------------------------
        self.out_proj: nn.Module
        if self.use_out_proj:
            self.out_proj = nn.Conv2d(
                self.dec_ch,
                self.dec_ch,
                kernel_size=1,
                bias=False,
            )
        else:
            self.out_proj = nn.Identity()

        self._reset_non_gate_parameters()

    def _reset_non_gate_parameters(self) -> None:
        # Directional DWConv: Kaiming works well for local filtering branches.
        nn.init.kaiming_normal_(
            self.dir_horizontal.weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.kaiming_normal_(
            self.dir_vertical.weight, mode="fan_out", nonlinearity="linear"
        )

        nn.init.trunc_normal_(self.global_gate.weight, std=0.02)
        nn.init.zeros_(self.global_gate.bias)
        nn.init.kaiming_normal_(
            self.global_proj.weight, mode="fan_out", nonlinearity="linear"
        )

        for module in (
            self.proj_dir,
            self.proj_global,
            self.proj_hf,
            self.proj_lf,
        ):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="linear"
            )

        fuse_conv = self.fuse[0]
        assert isinstance(fuse_conv, nn.Conv2d)
        nn.init.kaiming_normal_(
            fuse_conv.weight, mode="fan_out", nonlinearity="linear"
        )

        if isinstance(self.out_proj, nn.Conv2d):
            nn.init.kaiming_normal_(
                self.out_proj.weight, mode="fan_out", nonlinearity="linear"
            )

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    def _validate_inputs(
        self,
        D: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        A: torch.Tensor,
    ) -> None:
        if D.dim() != 4:
            raise ValueError(
                f"D must be NCHW [B,{self.dec_ch},H,W], got {tuple(D.shape)}."
            )
        if S.dim() != 4:
            raise ValueError(
                f"S must be NCHW [B,{self.skip_ch},H,W], got {tuple(S.shape)}."
            )
        if D.shape[1] != self.dec_ch:
            raise ValueError(
                f"D channel mismatch: expected dec_ch={self.dec_ch}, "
                f"got {D.shape[1]}."
            )
        if S.shape[1] != self.skip_ch:
            raise ValueError(
                f"S channel mismatch: expected skip_ch={self.skip_ch}, "
                f"got {S.shape[1]}."
            )
        if D.shape[0] != S.shape[0] or D.shape[-2:] != S.shape[-2:]:
            raise ValueError(
                "D and S must have the same batch and spatial size. "
                f"Got D={tuple(D.shape)}, S={tuple(S.shape)}."
            )

        if C.dim() != 2 or C.shape != (D.shape[0], self.struct_dim):
            raise ValueError(
                f"C must have shape [B,{self.struct_dim}], "
                f"got {tuple(C.shape)}."
            )
        if A.dim() != 2 or A.shape != (D.shape[0], self.num_routes):
            raise ValueError(
                f"A must have shape [B,{self.num_routes}] in "
                f"[LR,RL,TB,BT] order, got {tuple(A.shape)}."
            )

        for name, tensor in (("D", D), ("S", S), ("C", C), ("A", A)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf values.")

    def _build_control_vector(
        self,
        D: torch.Tensor,
        S_tilde: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        d_vec = F.adaptive_avg_pool2d(D, 1).flatten(1)
        s_vec = F.adaptive_avg_pool2d(S_tilde, 1).flatten(1)
        return torch.cat([d_vec, s_vec, C], dim=-1)

    def forward(
        self,
        D: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        A: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Fuse structure-guided skip evidence into current decoder feature."""
        self._validate_inputs(D, S, C, A)
        B = D.shape[0]

        # ------------------------------------------------------------------
        # 1) C-guided skip recalibration
        # ------------------------------------------------------------------
        gamma, beta = self.skip_condition(C)  # [B, Cs], [B, Cs]
        gamma_4d = gamma.view(B, self.skip_ch, 1, 1)
        beta_4d = beta.view(B, self.skip_ch, 1, 1)
        S_tilde = S * (1.0 + gamma_4d) + beta_4d

        # ------------------------------------------------------------------
        # 2) A-guided directional local evidence
        # A order is fixed: [LR, RL, TB, BT].
        # SIA already applies Softmax.  We only collapse directions into
        # horizontal/vertical preferences and renormalize those two axes.
        # ------------------------------------------------------------------
        a_h = A[:, 0] + A[:, 1]
        a_v = A[:, 2] + A[:, 3]
        axis = torch.stack([a_h, a_v], dim=-1)
        axis = axis / axis.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        a_h = axis[:, 0].view(B, 1, 1, 1)
        a_v = axis[:, 1].view(B, 1, 1, 1)

        F_h = self.dir_horizontal(S_tilde)
        F_v = self.dir_vertical(S_tilde)
        F_dir = a_h * F_h + a_v * F_v

        # ------------------------------------------------------------------
        # 3) Fixed low/high-frequency evidence
        # ------------------------------------------------------------------
        F_lf = self.low_filter(S_tilde)
        F_hf_raw = self.high_filter(S_tilde)

        # ------------------------------------------------------------------
        # 4) HF reliability filtering
        # q intentionally excludes A; A has already served the directional
        # branch, keeping responsibilities separated.
        # ------------------------------------------------------------------
        q = self._build_control_vector(D, S_tilde, C)
        R_hf = self.hf_reliability(q)                    # [B, Cs]
        R_hf_4d = R_hf.view(B, self.skip_ch, 1, 1)
        F_hf = F_hf_raw * R_hf_4d

        # ------------------------------------------------------------------
        # 5) Lightweight global evidence
        # ------------------------------------------------------------------
        global_context = F.adaptive_avg_pool2d(S_tilde, 1)
        global_weight = torch.sigmoid(self.global_gate(global_context))
        F_global = self.global_proj(S_tilde * global_weight)

        # ------------------------------------------------------------------
        # 6) Multi-evidence concat + 1x1 fusion
        # ------------------------------------------------------------------
        P_dir = self.proj_dir(F_dir)
        P_global = self.proj_global(F_global)
        P_hf = self.proj_hf(F_hf)
        P_lf = self.proj_lf(F_lf)

        F_cat = torch.cat([P_dir, P_global, P_hf, P_lf], dim=1)
        F_fuse = self.fuse(F_cat)                        # [B, Cd, H, W]

        # ------------------------------------------------------------------
        # 7) Overall skip gate + residual injection
        # ------------------------------------------------------------------
        g = self.overall_skip_gate(q)                    # [B, Cd]
        g_4d = g.view(B, self.dec_ch, 1, 1)
        F_gate = F_fuse * g_4d
        D_out = D + self.out_proj(F_gate)

        if not return_aux:
            return D_out

        aux: Dict[str, torch.Tensor] = {
            "skip_recalibrated": S_tilde,
            "axis_weight": axis,             # [B,2] -> [horizontal, vertical]
            "direction_horizontal": F_h,
            "direction_vertical": F_v,
            "directional_evidence": F_dir,
            "low_frequency": F_lf,
            "high_frequency_raw": F_hf_raw,
            "hf_reliability": R_hf,
            "high_frequency_filtered": F_hf,
            "global_weight": global_weight.flatten(1),
            "global_evidence": F_global,
            "fused_evidence": F_fuse,
            "overall_skip_gate": g,
            "gated_evidence": F_gate,
        }
        return D_out, aux


# Short alias for concise use in UNet.py.
MGEF = MambaGuidedEvidenceFusion


def build_mgef(
    dec_ch: int,
    skip_ch: int,
    *,
    struct_dim: int = 96,
    num_routes: int = 4,
    directional_kernel: int = 5,
    branch_ch: Optional[int] = None,
    gate_hidden_dim: Optional[int] = None,
    low_kernel: int = 5,
    low_sigma: float = 1.0,
    dog_kernel1: int = 3,
    dog_sigma1: float = 0.8,
    dog_kernel2: int = 5,
    dog_sigma2: float = 1.6,
    use_out_proj: bool = False,
) -> MambaGuidedEvidenceFusion:
    """Convenience factory for U-Net integration."""
    return MambaGuidedEvidenceFusion(
        dec_ch=dec_ch,
        skip_ch=skip_ch,
        struct_dim=struct_dim,
        num_routes=num_routes,
        directional_kernel=directional_kernel,
        branch_ch=branch_ch,
        gate_hidden_dim=gate_hidden_dim,
        low_kernel=low_kernel,
        low_sigma=low_sigma,
        dog_kernel1=dog_kernel1,
        dog_sigma1=dog_sigma1,
        dog_kernel2=dog_kernel2,
        dog_sigma2=dog_sigma2,
        use_out_proj=use_out_proj,
    )


__all__ = [
    "FixedGaussianBlurDW",
    "DifferenceOfGaussiansDW",
    "SkipFeatureAlign",
    "MambaGuidedEvidenceFusion",
    "MGEF",
    "build_mgef",
]
