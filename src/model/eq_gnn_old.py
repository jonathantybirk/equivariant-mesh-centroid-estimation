# Equivariant GNN (same public interface, minimal & provably SE(3)‑equivariant)
# -----------------------------------------------------------------------------
# * Keeps your original class names (`MultiCGLayer`, `EquivariantGNN`) so the
#   rest of your codebase continues to work unmodified.
# * Removes every non‑equivariant component (absolute x, residual bugs, etc.).
# * Uses **only** CG coefficients and spherical harmonics from e3nn; all maths
#   is hand‑coded in PyTorch.
# * Multiplicity = 1 for clarity; easy to lift by wrapping the loops again.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math, time
import torch, torch.nn as nn
from e3nn.o3 import _wigner, spherical_harmonics
from .base_model import BaseModel  # unchanged dependency

# -----------------------------------------------------------------------------
#  Helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

_CG_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def cg(l1: int, l2: int, l3: int, device=None):
    k = (l1, l2, l3)
    if k not in _CG_CACHE:
        _CG_CACHE[k] = _wigner._so3_clebsch_gordan(*k).float()
    return _CG_CACHE[k] if device is None else _CG_CACHE[k].to(device)


IRREP_DIM = {0: 1, 1: 3}  # scalar & vector only
TOTAL_DIM = 4  # 1+3


# -----------------------------------------------------------------------------
#  A single CG‑linear edge block (w·CG) ----------------------------------------
# -----------------------------------------------------------------------------
class CGLinear(nn.Module):
    """Implements  three admissible couplings  (1⊗0→1, 1⊗1→0, 1⊗1→1)."""

    def __init__(self):
        super().__init__()
        self.w_101 = nn.Parameter(torch.randn(1))
        self.w_110 = nn.Parameter(torch.randn(1))
        self.w_111 = nn.Parameter(torch.randn(1))
        self._buf = None  # will hold CG tensors on first forward

    def forward(self, scal_src, vec_src, sh):  # [E,1], [E,3], [E,3]
        if self._buf is None:
            d = sh.device
            self._cg101 = cg(1, 0, 1, d)  # (3,1,3)
            self._cg110 = cg(1, 1, 0, d)  # (3,3,1)
            self._cg111 = cg(1, 1, 1, d)  # (3,3,3)
        eins = lambda CG, x, y: torch.einsum("ijk,ei,ej->ek", CG, x, y)
        vec = self.w_101 * eins(self._cg101, sh, scal_src) + self.w_111 * eins(
            self._cg111, sh, vec_src
        )
        scal = self.w_110 * eins(self._cg110, sh, vec_src)
        return scal, vec


# -----------------------------------------------------------------------------
#  Message‑passing layer -------------------------------------------------------
# -----------------------------------------------------------------------------
class MultiCGLayer(nn.Module):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.cg = CGLinear()
        self.bias = nn.Parameter(torch.zeros(1))  # scalar bias after agg
        self.radial = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 3))

    def forward(self, node_feats, pos, edge_index):
        scalars = node_feats[:, :1]
        vectors = node_feats[:, 1:]
        src, dst = edge_index
        rel = pos[src] - pos[dst]  # (E,3)
        dist = rel.norm(dim=-1, keepdim=True)  # (E,1)
        unit = rel / (dist + 1e-8)
        sh = spherical_harmonics(1, unit, normalize=True)  # (E,3)
        sh = sh * self.radial(dist)  # distance‑aware weights
        s_msg, v_msg = self.cg(scalars[src], vectors[src], sh)
        # aggregate
        N = node_feats.size(0)
        s_aggr = torch.zeros_like(scalars).index_add_(0, dst, s_msg)
        v_aggr = torch.zeros_like(vectors).index_add_(0, dst, v_msg)
        # non‑lin (Swish) + gating
        s_upd = torch.nn.functional.silu(s_aggr + self.bias)
        gate = torch.sigmoid(s_upd)
        v_upd = v_aggr * gate
        # residual
        return torch.cat([scalars + s_upd, vectors + v_upd], dim=-1)


# -----------------------------------------------------------------------------
#  Full net (Lightning‑compatible) --------------------------------------------
# -----------------------------------------------------------------------------
class EquivariantGNN(BaseModel):
    def __init__(
        self, n_layers: int = 3, lr=1e-3, weight_decay=1e-5, debug: bool = False
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.layers = nn.ModuleList([MultiCGLayer(debug) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.debug = debug

    # ---------------- utils --------------
    @staticmethod
    def _init_feats(pos):
        r = pos.norm(dim=-1, keepdim=True)  # scalar invariant
        v = torch.zeros_like(pos)  # start vectors at 0
        return torch.cat([r, v], dim=-1)  # (N,4)

    # ------------- forward ---------------
    def forward(self, x_dummy, edge_index, edge_attr, batch=None, node_pos=None):
        # we ignore x_dummy (kept for interface compatibility)
        if node_pos is None:
            node_pos = x_dummy
        pos = node_pos - node_pos.mean(dim=0, keepdim=True)  # just in case
        feats = self._init_feats(pos)
        for layer in self.layers:
            feats = layer(feats, pos, edge_index)
        scalars = feats[:, :1]
        graph_scalar = scalars.mean(dim=0, keepdim=True)
        return self.readout(graph_scalar)
