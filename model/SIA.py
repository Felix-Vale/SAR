from __future__ import annotations

"""
Standalone Structural Information Aggregation (SIA)
===================================================

This module is the encoder-to-decoder structural interface used together with
SAR-Mamba.

Expected SAR-Mamba interface
----------------------------
    y4,  c4,  A4,  aux4  = sar4(x4)
    y16, c16, A16, aux16 = sar16(x16)

where
    c4, c16 : [B, Dc]     stage-level structural representations
    A4, A16 : [B, 4]      soft route profiles in [LR, RL, TB, BT] order

SIA interface
-------------
    C, A = sia(c4, A4, c16, A16)

Outputs
-------
    C : [B, Dc]            unified structural representation for decoder MGEF
    A : [B, 4]             unified route profile for decoder MGEF
                            (Softmax normalized, sum(A, dim=-1) == 1)

Design
------
The implementation follows the current SAR-Mamba/MGEF architecture:

Structural representation fusion:
    c4  -> Linear(Dc -> Dc)  --\
                                 concat -> Linear(2Dc -> H) -> GELU
    c16 -> Linear(Dc -> Dc)  --/        -> Linear(H -> Dc) -> C

Route-profile fusion:
    concat(A4, A16) -> Linear(8 -> 4) -> Softmax -> A

The full structural-token sequences are intentionally NOT passed to the
high-resolution decoder.  SIA only exposes fixed-dimensional C and A.

Important integration detail
----------------------------
Use the *soft route profile* returned as the third output of SARMambaBlock:

    y, c, A, aux = sar(x)

Do NOT pass aux["route_exec_weight"] or aux["route_hard_mask"] to SIA.
The decoder should receive the full learned route preference, while Top-k is
only the execution policy inside SAR-Mamba.

No detach() is used: gradients from decoder-side MGEF can flow through SIA
back into the structural representations and route profiles produced by
SAR-Mamba.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralInformationAggregation(nn.Module):
    """Fuse stride-4 and stride-16 SAR-Mamba structural information.

    Parameters
    ----------
    struct_dim:
        Dimension of c4/c16 and output C.  The current SAR-Mamba design uses
        token_dim=96, therefore struct_dim=96 is the recommended default.

    num_routes:
        Number of regular SAR routes.  The current framework is intentionally
        fixed to four axis-aligned routes in the order [LR, RL, TB, BT].

    hidden_dim:
        Hidden width of structural fusion MLP.  Defaults to struct_dim, which
        gives the intended 192 -> 96 -> 96 mapping when struct_dim=96.

    dropout:
        Optional dropout inside the structural fusion MLP.  Default 0.0 keeps
        the module identical to the minimal design in the reconstruction spec.
    """

    ROUTE_NAMES: Tuple[str, str, str, str] = ("LR", "RL", "TB", "BT")

    def __init__(
        self,
        struct_dim: int = 96,
        num_routes: int = 4,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if struct_dim <= 0:
            raise ValueError(f"struct_dim must be positive, got {struct_dim}.")
        if num_routes != 4:
            raise ValueError(
                "This SIA implementation is paired with SAR-Mamba's four "
                "routes [LR, RL, TB, BT], therefore num_routes must be 4."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.struct_dim = int(struct_dim)
        self.num_routes = int(num_routes)
        self.hidden_dim = int(hidden_dim or struct_dim)
        self.dropout_p = float(dropout)

        # ------------------------------------------------------------------
        # Structural representation fusion
        # ------------------------------------------------------------------
        # The two SAR stages have different semantic granularity, therefore
        # they first receive independent projections before concatenation.
        self.c4_proj = nn.Linear(self.struct_dim, self.struct_dim, bias=True)
        self.c16_proj = nn.Linear(self.struct_dim, self.struct_dim, bias=True)

        self.struct_fuse = nn.Sequential(
            nn.Linear(2 * self.struct_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.struct_dim, bias=True),
        )

        # ------------------------------------------------------------------
        # Route-profile fusion
        # ------------------------------------------------------------------
        # A4 and A16 are both [B,4] soft route distributions.  Their
        # concatenation is mapped to four logits and normalized with Softmax.
        self.route_fuse = nn.Linear(2 * self.num_routes, self.num_routes, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Stable small-weight initialization for the lightweight interface."""
        for module in (self.c4_proj, self.c16_proj):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

        for module in self.struct_fuse.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        nn.init.trunc_normal_(self.route_fuse.weight, std=0.02)
        nn.init.zeros_(self.route_fuse.bias)

    # ----------------------------------------------------------------------
    # Validation helpers
    # ----------------------------------------------------------------------
    def _validate_struct(self, x: torch.Tensor, name: str) -> None:
        if x.dim() != 2:
            raise ValueError(
                f"{name} must have shape [B,{self.struct_dim}], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.struct_dim:
            raise ValueError(
                f"{name} last dimension must be struct_dim={self.struct_dim}, "
                f"got {x.shape[-1]}."
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values.")

    def _validate_route(self, x: torch.Tensor, name: str) -> None:
        if x.dim() != 2:
            raise ValueError(
                f"{name} must have shape [B,{self.num_routes}], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.num_routes:
            raise ValueError(
                f"{name} last dimension must be num_routes={self.num_routes}, "
                f"got {x.shape[-1]}."
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values.")

    def forward(
        self,
        c4: torch.Tensor,       # [B, Dc]
        A4: torch.Tensor,       # [B, 4], SAR soft profile
        c16: torch.Tensor,      # [B, Dc]
        A16: torch.Tensor,      # [B, 4], SAR soft profile
        *,
        return_aux: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Aggregate multi-stage SAR conditions into decoder controls.

        The normal U-Net call is simply:

            C, A = self.sia(c4, A4, c16, A16)

        Set ``return_aux=True`` only for analysis/visualization.
        """
        self._validate_struct(c4, "c4")
        self._validate_struct(c16, "c16")
        self._validate_route(A4, "A4")
        self._validate_route(A16, "A16")

        batch = c4.shape[0]
        if c16.shape[0] != batch or A4.shape[0] != batch or A16.shape[0] != batch:
            raise ValueError(
                "c4, A4, c16 and A16 must have the same batch size, got "
                f"{c4.shape[0]}, {A4.shape[0]}, {c16.shape[0]}, {A16.shape[0]}."
            )

        # ------------------------------------------------------------------
        # 1) Unified structural representation C
        # ------------------------------------------------------------------
        c4_p = self.c4_proj(c4)                  # [B, Dc]
        c16_p = self.c16_proj(c16)               # [B, Dc]
        c_cat = torch.cat([c4_p, c16_p], dim=-1) # [B, 2Dc]
        C = self.struct_fuse(c_cat)               # [B, Dc]

        # ------------------------------------------------------------------
        # 2) Unified route profile A
        # ------------------------------------------------------------------
        # SARMambaBlock already returns normalized A4/A16.  SIA does not
        # hard-mask or detach them; the learned linear fusion produces decoder
        # route logits, and Softmax guarantees a valid four-route distribution.
        route_cat = torch.cat([A4, A16], dim=-1) # [B, 8]
        route_logits = self.route_fuse(route_cat) # [B, 4]
        A = F.softmax(route_logits, dim=-1)       # [B, 4]

        if not return_aux:
            return C, A

        aux: Dict[str, torch.Tensor] = {
            "c4_projected": c4_p,
            "c16_projected": c16_p,
            "structural_condition": C,
            "route_logits": route_logits,
            "route_profile": A,
            "route_profile_stride4": A4,
            "route_profile_stride16": A16,
        }
        return C, A, aux


# Short alias for concise use in UNet.py.
SIA = StructuralInformationAggregation


def build_sia(
    struct_dim: int = 96,
    num_routes: int = 4,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
) -> StructuralInformationAggregation:
    """Convenience factory for U-Net integration."""
    return StructuralInformationAggregation(
        struct_dim=struct_dim,
        num_routes=num_routes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Standalone smoke test.
    torch.manual_seed(0)

    B = 2
    Dc = 96

    sia = StructuralInformationAggregation(struct_dim=Dc, num_routes=4)

    c4 = torch.randn(B, Dc, requires_grad=True)
    c16 = torch.randn(B, Dc, requires_grad=True)

    # Simulate the *soft* A returned by SARMambaBlock.
    A4 = torch.softmax(torch.randn(B, 4, requires_grad=True), dim=-1)
    A16 = torch.softmax(torch.randn(B, 4, requires_grad=True), dim=-1)

    C, A, aux = sia(c4, A4, c16, A16, return_aux=True)

    print("C shape:", tuple(C.shape))
    print("A shape:", tuple(A.shape))
    print("A sum:", A.sum(dim=-1))
    print("route order:", sia.ROUTE_NAMES)

    # Verify decoder-side gradients can flow through SIA toward SAR conditions.
    loss = C.square().mean() + A.square().mean()
    loss.backward()

    print("c4 grad exists:", c4.grad is not None)
    print("c16 grad exists:", c16.grad is not None)
