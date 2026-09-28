from __future__ import annotations

"""
Standalone Structure-Adaptive Routing Mamba (SAR-Mamba)
=======================================================

This file is designed to be imported directly by a U-Net style segmentation
network.  It does not depend on the old GTX / VMamba wrapper in modules.py.

Main interface
--------------
    block = SARMambaBlock(d_model=128, ...)
    y, c, A, aux = block(x)

Inputs
------
    x:  [B, C, H, W]

Outputs
-------
    y:   [B, C, H, W]  spatial feature after SAR-Mamba
    c:   [B, token_dim] sample-specific structural representation
    A:   [B, 4]         soft route profile in the order [LR, RL, TB, BT]
    aux: dictionary for training / visualization / ablation

Design implemented here
-----------------------
1) Adaptive structural tokenizer
   - AdaptiveAvgPool2d -> cross-attention with Kmax learnable queries
   - token importance p and differentiable effective-token mask m
   - weighted aggregation to structural representation c

2) Structure-conditioned Mamba dynamics
   - c modulates the input branch X
   - c modulates the gate branch Z
   - c adds a bias to the effective state-update step Delta
   - the condition heads are zero-initialized, so the block starts close to
     the unconditioned Mamba behavior

3) Structure-adaptive routing
   - four regular axis-aligned routes: LR, RL, TB, BT
   - structural tokens predict a sample-specific soft route profile A
   - hard Top-k routing is supported with a straight-through estimator

4) Structural-token-augmented selective scan
   - effective structural tokens are projected to the SSM inner dimension
   - they are prepended to every executed route sequence
   - token outputs are removed after scanning and the spatial part is restored

5) Dense training + sparse inference by default
   - train: all four routes are evaluated but forward fusion is hard Top-k;
            this provides a useful straight-through gradient for the router
   - eval:  only selected Top-k routes are executed (true sparse execution)

Selective scan backend
----------------------
The CUDA implementation from mamba_ssm is used when available.  A pure
PyTorch reference implementation is included as a functional fallback.  The
fallback is intentionally simple and is much slower for long sequences; for
real training, installing mamba_ssm is strongly recommended.
"""

from typing import Dict, Optional, Tuple, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Selective scan backend
# =============================================================================

_HAS_MAMBA_SSM = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn  # type: ignore
    _HAS_MAMBA_SSM = True
except Exception:
    _mamba_selective_scan_fn = None


def _selective_scan_reference(
    u: torch.Tensor,               # [B, D, L]
    delta: torch.Tensor,           # [B, D, L]
    A: torch.Tensor,               # [D, N]
    B: torch.Tensor,               # [B, D, N, L]
    C: torch.Tensor,               # [B, D, N, L]
    D: Optional[torch.Tensor] = None,  # [D]
    z: Optional[torch.Tensor] = None,  # [B, D, L]
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """A slow, explicit selective-scan reference implementation.

    This fallback only needs to support the tensor layout used by this file.
    It is useful for correctness tests and CPU execution, not for speed.
    """
    if u.dim() != 3 or delta.dim() != 3:
        raise ValueError("u and delta must have shape [B, D, L].")
    if B.dim() != 4 or C.dim() != 4:
        raise ValueError("B and C must have shape [B, D, N, L].")

    batch, dim, length = u.shape
    n_state = A.shape[-1]

    work_dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype

    u_w = u.to(work_dtype)
    delta_w = delta.to(work_dtype)
    A_w = A.to(work_dtype)
    B_w = B.to(work_dtype)
    C_w = C.to(work_dtype)

    if delta_bias is not None:
        db = delta_bias.to(work_dtype)
        if db.dim() == 1:
            db = db.view(1, -1, 1)
        elif db.dim() == 2:
            db = db.unsqueeze(-1)
        delta_w = delta_w + db

    if delta_softplus:
        delta_w = F.softplus(delta_w)

    state = torch.zeros(batch, dim, n_state, device=u.device, dtype=work_dtype)
    ys = []

    # A is negative in this implementation.
    for t in range(length):
        dt_t = delta_w[:, :, t]                     # [B, D]
        u_t = u_w[:, :, t]                          # [B, D]
        B_t = B_w[:, :, :, t]                       # [B, D, N]
        C_t = C_w[:, :, :, t]                       # [B, D, N]

        dA = torch.exp(dt_t.unsqueeze(-1) * A_w.unsqueeze(0))
        dBu = dt_t.unsqueeze(-1) * B_t * u_t.unsqueeze(-1)
        state = dA * state + dBu

        y_t = (state * C_t).sum(dim=-1)             # [B, D]
        if D is not None:
            y_t = y_t + D.to(work_dtype).view(1, -1) * u_t
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)                     # [B, D, L]
    if z is not None:
        y = y * F.silu(z.to(work_dtype))

    y = y.to(u.dtype)
    if return_last_state:
        return y, state.to(u.dtype)
    return y


def selective_scan_fn(
    *,
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    if _HAS_MAMBA_SSM:
        return _mamba_selective_scan_fn(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )

    return _selective_scan_reference(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


# =============================================================================
# Utility blocks
# =============================================================================


class DropPath(nn.Module):
    """Per-sample stochastic depth without a timm dependency."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(rnd)
        return x * mask / keep_prob


class ZeroInitConditionMLP(nn.Module):
    """Condition MLP whose last projection is zero-initialized.

    At initialization the output is exactly zero, so affine modulation starts
    from identity and Delta conditioning starts from no change.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 1. Adaptive structural tokenizer
# =============================================================================


class AdaptiveStructuralTokenizer(nn.Module):
    """Extract candidate structural tokens and input-dependent effective tokens."""

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 96,
        kmax: int = 8,
        num_heads: int = 4,
        pool_hw: Tuple[int, int] = (8, 8),
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        tau: float = 0.5,
        temperature: float = 1.0,
        hard_token_inference: bool = False,
        coverage_rho: float = 0.90,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by num_heads={num_heads}.")
        if kmax <= 0:
            raise ValueError("kmax must be positive.")

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.kmax = int(kmax)
        self.pool_hw = tuple(pool_hw)
        self.tau = float(tau)
        self.hard_token_inference = bool(hard_token_inference)
        self.coverage_rho = float(coverage_rho)

        self.register_buffer(
            "temperature",
            torch.tensor(float(temperature), dtype=torch.float32),
            persistent=True,
        )

        self.kv_proj = nn.Linear(in_channels, token_dim, bias=False)
        self.q_tokens = nn.Parameter(torch.zeros(kmax, token_dim))
        nn.init.trunc_normal_(self.q_tokens, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden = max(token_dim, int(token_dim * mlp_ratio))
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
        )

        self.score = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )

    @torch.no_grad()
    def set_temperature(self, temperature: float) -> None:
        """Update soft-mask temperature, e.g. for epoch-wise annealing."""
        value = max(float(temperature), 1e-4)
        self.temperature.fill_(value)

    def _coverage_mask(self, p: torch.Tensor) -> torch.Tensor:
        """Keep the minimum number of tokens reaching cumulative score coverage.

        p: [B, K] in [0, 1]
        returns hard mask [B, K] in {0, 1}
        """
        sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)
        denom = sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        cumulative = torch.cumsum(sorted_p, dim=-1) / denom

        # Number strictly below rho + the first token crossing rho.
        num_keep = (cumulative < self.coverage_rho).sum(dim=-1) + 1
        num_keep = num_keep.clamp(min=1, max=self.kmax)

        rank = torch.arange(self.kmax, device=p.device).view(1, -1)
        sorted_keep = rank < num_keep.unsqueeze(-1)

        hard = torch.zeros_like(p)
        hard.scatter_(1, sorted_idx, sorted_keep.to(p.dtype))
        return hard

    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, _, _ = x.shape

        x_pool = F.adaptive_avg_pool2d(x, self.pool_hw)
        grid = x_pool.flatten(2).transpose(1, 2).contiguous()   # [B, S, C]
        kv = self.kv_proj(grid)                                # [B, S, D]

        q = self.q_tokens.unsqueeze(0).expand(b, -1, -1)       # [B, K, D]
        tokens, _ = self.attn(q, kv, kv, need_weights=False)
        tokens = tokens + self.token_ffn(tokens)                # [B, K, D]

        p = torch.sigmoid(self.score(tokens)).squeeze(-1)       # [B, K]

        temp = self.temperature.to(device=x.device, dtype=p.dtype).clamp_min(1e-4)
        soft_m = torch.sigmoid((p - self.tau) / temp)           # [B, K]

        if (not self.training) and self.hard_token_inference:
            hard_m = self._coverage_mask(p)
            m = hard_m
        else:
            hard_m = self._coverage_mask(p).detach()
            m = soft_m

        t_eff = tokens * m.unsqueeze(-1)                        # [B, K, D]

        # c uses importance * activation, while the inserted tokens use m only.
        token_weight_raw = p * m
        token_weight = token_weight_raw / token_weight_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        c = (tokens * token_weight.unsqueeze(-1)).sum(dim=1)    # [B, D]

        aux = {
            "token_score": p,
            "token_soft_mask": soft_m,
            "token_mask": m,
            "token_hard_mask": hard_m,
            "token_weight": token_weight,
            "active_token_count_soft": soft_m.sum(dim=-1),
            "active_token_count_hard": hard_m.sum(dim=-1),
            "token_budget_loss": soft_m.mean(),
        }
        return tokens, t_eff, p, m, c, aux


# =============================================================================
# 2. Structural router
# =============================================================================


ROUTE_NAMES = ("LR", "RL", "TB", "BT")


class StructuralRouter(nn.Module):
    """Predict the soft route profile A and execution weights."""

    def __init__(
        self,
        token_dim: int = 96,
        num_routes: int = 4,
        topk: int = 2,
        route_mode: Literal["soft", "topk_st", "topk_hard"] = "topk_st",
    ):
        super().__init__()
        if num_routes != 4:
            raise ValueError("This implementation intentionally uses exactly four routes: LR/RL/TB/BT.")
        if not (1 <= topk <= num_routes):
            raise ValueError(f"topk must be in [1, {num_routes}].")

        self.token_dim = int(token_dim)
        self.num_routes = int(num_routes)
        self.topk = int(topk)
        self.route_mode = route_mode

        self.route_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim, bias=False),
        )
        self.route_embedding = nn.Parameter(torch.zeros(num_routes, token_dim))
        nn.init.trunc_normal_(self.route_embedding, std=0.02)

    def set_route_mode(self, mode: Literal["soft", "topk_st", "topk_hard"]) -> None:
        if mode not in ("soft", "topk_st", "topk_hard"):
            raise ValueError(f"Unknown route mode: {mode}")
        self.route_mode = mode

    def forward(
        self,
        t_eff: torch.Tensor,   # [B, K, D]
        p: torch.Tensor,       # [B, K]
        m: torch.Tensor,       # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_w_raw = p * m
        token_w = token_w_raw / token_w_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        t_route = self.route_proj(t_eff)                        # [B, K, D]
        scale = 1.0 / math.sqrt(self.token_dim)
        score = torch.einsum("bkd,rd->bkr", t_route, self.route_embedding) * scale
        token_route_prob = torch.softmax(score, dim=-1)         # [B, K, 4]

        A_soft = (token_route_prob * token_w.unsqueeze(-1)).sum(dim=1)
        A_soft = A_soft / A_soft.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        selected_idx = torch.topk(A_soft, k=self.topk, dim=-1, largest=True, sorted=True).indices
        hard_mask = torch.zeros_like(A_soft)
        hard_mask.scatter_(1, selected_idx, 1.0)

        selected_soft = A_soft * hard_mask
        hard_weight = selected_soft / selected_soft.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.route_mode == "soft":
            exec_weight = A_soft
        elif self.route_mode == "topk_hard":
            exec_weight = hard_weight
        elif self.route_mode == "topk_st":
            # Forward == hard Top-k normalized weight.
            # Backward approximates the gradient of the soft profile A_soft.
            exec_weight = hard_weight.detach() - A_soft.detach() + A_soft
        else:
            raise ValueError(f"Unknown route_mode: {self.route_mode}")

        return A_soft, exec_weight, selected_idx, hard_mask, token_route_prob


# =============================================================================
# 3. Four regular axis-aligned sequence builders
# =============================================================================


def _route_to_sequence(x: torch.Tensor, route: int) -> torch.Tensor:
    """[B, D, H, W] -> [B, D, L] for LR/RL/TB/BT."""
    if route == 0:      # LR: row-major, each row left -> right
        return x.flatten(2)
    if route == 1:      # RL: each row right -> left
        return torch.flip(x, dims=[-1]).flatten(2)
    if route == 2:      # TB: column-major, each column top -> bottom
        return x.permute(0, 1, 3, 2).contiguous().flatten(2)
    if route == 3:      # BT: each column bottom -> top
        return torch.flip(x, dims=[-2]).permute(0, 1, 3, 2).contiguous().flatten(2)
    raise ValueError(f"Invalid route index: {route}")


def _sequence_to_route_map(seq: torch.Tensor, route: int, h: int, w: int) -> torch.Tensor:
    """Inverse of _route_to_sequence.  [B, D, L] -> [B, D, H, W]."""
    b, d, l = seq.shape
    if l != h * w:
        raise ValueError(f"Sequence length {l} does not match H*W={h*w}.")

    if route == 0:      # LR
        return seq.view(b, d, h, w)
    if route == 1:      # RL
        return torch.flip(seq.view(b, d, h, w), dims=[-1])
    if route == 2:      # TB
        return seq.view(b, d, w, h).permute(0, 1, 3, 2).contiguous()
    if route == 3:      # BT
        x = seq.view(b, d, w, h).permute(0, 1, 3, 2).contiguous()
        return torch.flip(x, dims=[-2])
    raise ValueError(f"Invalid route index: {route}")


# =============================================================================
# 4. SAR-Mamba block
# =============================================================================


class SARMambaBlock(nn.Module):
    """Structure-Adaptive Routing Mamba.

    Parameters are intentionally close to the old VMamba/SS2D code so the block
    can replace it with minimal changes in U-Net.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 1.5,
        conv_kernel: int = 3,
        dt_rank: int = 4,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        token_dim: int = 96,
        kmax: int = 8,
        token_heads: int = 4,
        token_pool_hw: Tuple[int, int] = (8, 8),
        token_mlp_ratio: float = 2.0,
        token_tau: float = 0.5,
        token_temperature: float = 1.0,
        hard_token_inference: bool = False,
        token_coverage_rho: float = 0.90,
        route_topk: int = 2,
        route_mode: Literal["soft", "topk_st", "topk_hard"] = "topk_st",
        sparse_route_train: bool = False,
        sparse_route_eval: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if d_state <= 0:
            raise ValueError("d_state must be positive.")
        if dt_rank <= 0:
            raise ValueError("dt_rank must be positive.")
        if dt_min <= 0 or dt_max <= dt_min:
            raise ValueError("Require 0 < dt_min < dt_max.")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = float(expand)
        self.d_inner = int(round(d_model * expand))
        self.dt_rank = int(dt_rank)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.token_dim = int(token_dim)
        self.route_topk = int(route_topk)
        self.sparse_route_train = bool(sparse_route_train)
        self.sparse_route_eval = bool(sparse_route_eval)

        # Pre-norm keeps the external API channel-first while using channel-last LN.
        self.in_norm = nn.LayerNorm(d_model)

        # Tokenizer and router live inside each SAR block; no external MultiScaleGTX is required.
        self.tokenizer = AdaptiveStructuralTokenizer(
            in_channels=d_model,
            token_dim=token_dim,
            kmax=kmax,
            num_heads=token_heads,
            pool_hw=token_pool_hw,
            mlp_ratio=token_mlp_ratio,
            tau=token_tau,
            temperature=token_temperature,
            hard_token_inference=hard_token_inference,
            coverage_rho=token_coverage_rho,
        )
        self.router = StructuralRouter(
            token_dim=token_dim,
            num_routes=4,
            topk=route_topk,
            route_mode=route_mode,
        )

        # Standard Mamba input projection -> X branch + Z gate branch.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.dwconv = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=self.d_inner,
            bias=False,
        )

        # Structure-conditioned modulation.
        self.cond_x = ZeroInitConditionMLP(token_dim, 2 * self.d_inner)
        self.cond_z = ZeroInitConditionMLP(token_dim, 2 * self.d_inner)
        self.cond_delta = ZeroInitConditionMLP(token_dim, self.d_inner)

        # Structural tokens inserted into X and Z sequences.
        self.token_to_x = nn.Linear(token_dim, self.d_inner, bias=False)
        self.token_to_z = nn.Linear(token_dim, self.d_inner, bias=False)

        # Input-dependent SSM parameters per channel.
        self.param_proj = nn.Conv1d(
            self.d_inner,
            self.d_inner * (dt_rank + 2 * d_state),
            kernel_size=1,
            groups=self.d_inner,
            bias=False,
        )
        self.dt_w = nn.Parameter(torch.empty(self.d_inner, dt_rank))
        self.dt_b = nn.Parameter(torch.empty(self.d_inner))

        # Continuous-time state matrix and skip coefficient.
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.dt_w, std=0.02)

        # Initialize softplus(dt_b) into [dt_min, dt_max].
        with torch.no_grad():
            log_min = math.log(self.dt_min)
            log_max = math.log(self.dt_max)
            dt_init = torch.exp(
                torch.rand(self.d_inner, device=self.dt_b.device) * (log_max - log_min) + log_min
            )
            # inverse softplus: x = y + log(-expm1(-y))
            inv_sp = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_b.copy_(inv_sp)

        # Keep the old code's conservative A scale rather than changing the base dynamics too much.
        nn.init.normal_(self.A_log, mean=-3.0, std=0.02)
        nn.init.ones_(self.D_skip)

    @torch.no_grad()
    def set_token_temperature(self, temperature: float) -> None:
        self.tokenizer.set_temperature(temperature)

    def set_route_mode(self, mode: Literal["soft", "topk_st", "topk_hard"]) -> None:
        self.router.set_route_mode(mode)

    def _compute_delta_and_state_params(
        self,
        x_seq: torch.Tensor,          # [B, D, L]
        delta_bias_c: torch.Tensor,   # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, d, l = x_seq.shape
        p = self.param_proj(x_seq)
        p = p.view(b, d, self.dt_rank + 2 * self.d_state, l)

        dt_raw = p[:, :, : self.dt_rank, :]                         # [B,D,R,L]
        Bp = p[:, :, self.dt_rank : self.dt_rank + self.d_state, :] # [B,D,N,L]
        Cp = p[:, :, self.dt_rank + self.d_state :, :]              # [B,D,N,L]

        dt_pre = (
            dt_raw * self.dt_w.view(1, d, self.dt_rank, 1)
        ).sum(dim=2) + self.dt_b.view(1, d, 1)

        # Structural condition acts directly on the pre-softplus Delta parameter.
        dt_pre = dt_pre + delta_bias_c.unsqueeze(-1)
        dt = F.softplus(dt_pre)
        dt = dt.clamp(min=self.dt_min, max=self.dt_max)
        dt = dt.to(dtype=x_seq.dtype)

        return dt, Bp, Cp

    def _ssm(
        self,
        x_seq: torch.Tensor,          # [B,D,L]
        z_seq: torch.Tensor,          # [B,D,L]
        delta_bias_c: torch.Tensor,   # [B,D]
    ) -> torch.Tensor:
        dt, Bp, Cp = self._compute_delta_and_state_params(x_seq, delta_bias_c)
        A = -torch.exp(self.A_log)     # [D,N], negative

        y = selective_scan_fn(
            u=x_seq,
            delta=dt,
            A=A,
            B=Bp,
            C=Cp,
            D=self.D_skip,
            z=z_seq,
            delta_bias=None,
            delta_softplus=False,      # dt was already softplus'ed exactly once
            return_last_state=False,
        )
        return y

    def _scan_one_route(
        self,
        x_map: torch.Tensor,          # [B,D,H,W]
        z_map: torch.Tensor,          # [B,D,H,W]
        token_x: torch.Tensor,        # [B,D,K]
        token_z: torch.Tensor,        # [B,D,K]
        delta_bias_c: torch.Tensor,   # [B,D]
        route: int,
    ) -> torch.Tensor:
        b, d, h, w = x_map.shape

        x_seq = _route_to_sequence(x_map, route)
        z_seq = _route_to_sequence(z_map, route)

        # Structural tokens are real sequence elements at the prefix.
        x_ext = torch.cat([token_x, x_seq], dim=-1)
        z_ext = torch.cat([token_z, z_seq], dim=-1)

        y_ext = self._ssm(x_ext, z_ext, delta_bias_c)
        k = token_x.shape[-1]
        y_spatial = y_ext[:, :, k:]

        return _sequence_to_route_map(y_spatial, route, h, w)

    def _execute_routes_dense(
            self,
            x_map: torch.Tensor,
            z_map: torch.Tensor,
            token_x: torch.Tensor,
            token_z: torch.Tensor,
            delta_bias_c: torch.Tensor,
            exec_weight: torch.Tensor,
    ) -> torch.Tensor:

        y = torch.zeros_like(x_map)

        for route in range(4):
            yr = self._scan_one_route(
                x_map=x_map,
                z_map=z_map,
                token_x=token_x,
                token_z=token_z,
                delta_bias_c=delta_bias_c,
                route=route,
            )

            # AMP dtype safety
            yr = yr.to(dtype=y.dtype)

            wr = exec_weight[:, route].view(
                -1, 1, 1, 1
            ).to(
                device=yr.device,
                dtype=yr.dtype,
            )

            y = y + yr * wr

        return y


    def _execute_routes_sparse(
            self,
            x_map: torch.Tensor,
            z_map: torch.Tensor,
            token_x: torch.Tensor,
            token_z: torch.Tensor,
            delta_bias_c: torch.Tensor,
            exec_weight: torch.Tensor,  # [B,4]
            hard_mask: torch.Tensor,  # [B,4]
    ) -> torch.Tensor:
        """Only selected routes are scanned; samples are grouped per route."""

        y = torch.zeros_like(x_map)

        for route in range(4):
            idx = torch.nonzero(
                hard_mask[:, route] > 0,
                as_tuple=False,
            ).squeeze(-1)

            if idx.numel() == 0:
                continue

            xr = x_map.index_select(0, idx)
            zr = z_map.index_select(0, idx)
            txr = token_x.index_select(0, idx)
            tzr = token_z.index_select(0, idx)
            dbr = delta_bias_c.index_select(0, idx)

            yr = self._scan_one_route(
                x_map=xr,
                z_map=zr,
                token_x=txr,
                token_z=tzr,
                delta_bias_c=dbr,
                route=route,
            )

            # ----------------------------------------------------------
            # AMP dtype safety:
            # sparse index_copy requires source.dtype == y.dtype
            # ----------------------------------------------------------
            yr = yr.to(dtype=y.dtype)

            wr = exec_weight.index_select(
                0, idx
            )[:, route].view(-1, 1, 1, 1)

            wr = wr.to(
                device=yr.device,
                dtype=yr.dtype,
            )

            yr = yr * wr

            update = y.index_select(0, idx) + yr

            # Defensive cast: index_copy requires exact dtype match.
            update = update.to(dtype=y.dtype)

            # Functional index_copy keeps autograd connectivity from yr.
            y = y.index_copy(
                0,
                idx,
                update,
            )
        return y



    def forward(
        self,
        x: torch.Tensor,  # [B,C,H,W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"Expected x as [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != self.d_model:
            raise ValueError(
                f"Expected {self.d_model} input channels, got {x.shape[1]}."
            )

        residual = x
        b, c, h, w = x.shape

        # ---------------------------------------------------------------------
        # 1) Adaptive structural tokens -> c
        # ---------------------------------------------------------------------
        tokens, t_eff, p, m, c_struct, token_aux = self.tokenizer(x)

        # ---------------------------------------------------------------------
        # 2) Soft route profile A and Top-k execution weights
        # ---------------------------------------------------------------------
        A_soft, exec_weight, selected_idx, hard_mask, token_route_prob = self.router(
            t_eff=t_eff,
            p=p,
            m=m,
        )

        # ---------------------------------------------------------------------
        # 3) Standard Mamba input projection and local mixing
        # ---------------------------------------------------------------------
        x_cl = x.permute(0, 2, 3, 1).contiguous()
        x_cl = self.in_norm(x_cl)
        xz = self.in_proj(x_cl)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.permute(0, 3, 1, 2).contiguous()
        x_branch = F.silu(self.dwconv(x_branch))                    # [B,D,H,W]
        z_branch = z_branch.permute(0, 3, 1, 2).contiguous()        # [B,D,H,W]

        # ---------------------------------------------------------------------
        # 4) c conditions X / Z / Delta
        # ---------------------------------------------------------------------
        gx, bx = self.cond_x(c_struct).chunk(2, dim=-1)
        gz, bz = self.cond_z(c_struct).chunk(2, dim=-1)
        delta_bias_c = self.cond_delta(c_struct)                    # [B,D]

        x_branch = x_branch * (1.0 + gx.unsqueeze(-1).unsqueeze(-1)) + bx.unsqueeze(-1).unsqueeze(-1)
        z_branch = z_branch * (1.0 + gz.unsqueeze(-1).unsqueeze(-1)) + bz.unsqueeze(-1).unsqueeze(-1)

        # ---------------------------------------------------------------------
        # 5) Effective structural tokens are prepended to every executed route
        # ---------------------------------------------------------------------
        token_x = self.token_to_x(t_eff).transpose(1, 2).contiguous()  # [B,D,K]
        token_z = self.token_to_z(t_eff).transpose(1, 2).contiguous()  # [B,D,K]

        # ---------------------------------------------------------------------
        # 6) Route-specific selective scan
        # ---------------------------------------------------------------------
        if self.router.route_mode == "soft":
            sparse_now = False
        else:
            sparse_now = self.sparse_route_train if self.training else self.sparse_route_eval

        if sparse_now:
            y_inner = self._execute_routes_sparse(
                x_map=x_branch,
                z_map=z_branch,
                token_x=token_x,
                token_z=token_z,
                delta_bias_c=delta_bias_c,
                exec_weight=exec_weight,
                hard_mask=hard_mask,
            )
        else:
            y_inner = self._execute_routes_dense(
                x_map=x_branch,
                z_map=z_branch,
                token_x=token_x,
                token_z=token_z,
                delta_bias_c=delta_bias_c,
                exec_weight=exec_weight,
            )

        # ---------------------------------------------------------------------
        # 7) Output projection + residual
        # ---------------------------------------------------------------------
        y = y_inner.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = self.out_proj(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = residual + self.drop_path(y)

        aux: Dict[str, torch.Tensor] = dict(token_aux)
        aux.update(
            {
                "route_profile": A_soft,
                "route_exec_weight": exec_weight,
                "route_selected_idx": selected_idx,
                "route_hard_mask": hard_mask,
                "token_route_prob": token_route_prob,
                "structural_representation": c_struct,
            }
        )

        return y, c_struct, A_soft, aux


# =============================================================================
# Convenience factory + smoke test
# =============================================================================


def build_sar_mamba(d_model: int, **kwargs) -> SARMambaBlock:
    """Convenience factory for use from UNet.py."""
    return SARMambaBlock(d_model=d_model, **kwargs)


if __name__ == "__main__":
    # Lightweight CPU smoke test.  The pure PyTorch fallback is slow for large
    # maps, so keep this example small.
    torch.manual_seed(0)

    block = SARMambaBlock(
        d_model=32,
        d_state=8,
        expand=1.5,
        dt_rank=4,
        token_dim=32,
        kmax=4,
        token_heads=4,
        token_pool_hw=(4, 4),
        route_topk=2,
        route_mode="topk_st",
        sparse_route_train=False,
        sparse_route_eval=True,
    )

    x = torch.randn(2, 32, 8, 8)
    y, c, A, aux = block(x)

    print("mamba_ssm CUDA backend:", _HAS_MAMBA_SSM)
    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    print("c:", tuple(c.shape))
    print("A:", tuple(A.shape), "sum=", A.sum(dim=-1))
    print("selected routes:", aux["route_selected_idx"])
