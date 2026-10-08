"""Small construction compatibility layer for the frozen GIFT models.

Extracted from the project's local experiment snapshot
``ccfa-workfiles/experiments/gift_ce_www_full_20261006``:
``reference/vendor/old_models.py``, ``reference/models.py``,
``legacy/screening_candidates.py``, and ``legacy/candidates.py``.
No external baseline implementation is imported or distributed here.

The discarded calendar and routing layers below deliberately consume the same
initialization RNG stream as the frozen implementation. Constructing only the
retained layers would change the initialized values of later parameters.
"""
from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicLatentRouter(nn.Module):
    """Construction skeleton; GIFT's forward is in ``WWWForecaster``.

    ``assign``, ``role_key``, ``role_value``, ``query``, and ``role_mixer``
    are deleted after their RNG draws. They are absent from released models.
    """

    def __init__(self, dim: int, roles: int, ff_dim: int, dropout: float):
        super().__init__()
        self.roles = roles
        self.center_routing = False
        self.residual_gate = "scalar"
        self.assignment_topk = None
        self.norm1 = nn.LayerNorm(dim)
        self.assign = nn.Linear(dim, roles)
        self.value = nn.Linear(dim, dim)
        self.role_key = nn.Linear(dim, dim)
        self.role_value = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.role_mixer = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, dim),
        )
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.spatial_scale = nn.Parameter(torch.tensor(-2.0))


class StrongTransferForecaster(nn.Module):
    """Dynamic constructor retained for exact initialization compatibility."""

    def __init__(self, variant: str, history: int = 12, horizon: int = 12,
                 dim: int = 128, heads: int = 4, ff_dim: int = 256,
                 layers: int = 3, roles: int = 16, dropout: float = 0.1,
                 use_profile: bool = False):
        super().__init__()
        if variant != "dynamic" or use_profile:
            raise ValueError("The compatibility constructor supports GIFT only")
        self.variant = variant
        self.history = history
        self.horizon = horizon
        self.use_profile = use_profile
        self.register_buffer("graph", torch.empty(0), persistent=False)
        self.history_embed = nn.Sequential(
            nn.Linear(history, dim), nn.GELU(), nn.LayerNorm(dim),
        )
        # Discarded later, but the original constructor consumed these draws.
        self.calendar_embed = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            DynamicLatentRouter(dim, roles, ff_dim, dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, horizon)


class ZeroCalendar(nn.Module):
    def forward(self, value):
        return value.new_zeros(value.shape[0], 128)


def _history_mask(mask, data):
    if mask is None:
        return torch.ones_like(data[..., 0].transpose(1, 2), dtype=torch.bool)
    if mask.ndim == 4:
        mask = mask[..., 0]
    if tuple(mask.shape) == tuple(data.shape[:3]):
        mask = mask.transpose(1, 2)
    expected = (data.shape[0], data.shape[2], 12)
    if tuple(mask.shape) != expected:
        raise ValueError(f"History mask must represent {expected}, got {tuple(mask.shape)}")
    return mask.to(device=data.device, dtype=torch.bool)


def _masked_mae(prediction, target, mask):
    mask = mask & torch.isfinite(target)
    clean = torch.where(mask, target, prediction.detach())
    absolute = (prediction - clean).abs()
    return absolute.sum() / mask.sum().clamp_min(1)


def _read(query, key, value, sender_weight=None):
    weighted_key = key if sender_weight is None else key * sender_weight.unsqueeze(-1)
    moment = torch.einsum("bnr,bnd->brd", weighted_key, value)
    mass = weighted_key.sum(dim=1)
    numerator = torch.einsum("bnr,brd->bnd", query, moment)
    denominator = torch.einsum("bnr,br->bn", query, mass)
    context = numerator / denominator.unsqueeze(-1).clamp_min(1e-6)
    return context, moment, mass


def _paired_content(content, permutation, speed):
    if permutation is None:
        return content
    permutation = permutation.to(device=content.device, dtype=torch.long)
    if tuple(permutation.shape) != tuple(speed.shape[:2]):
        raise ValueError("Node permutation must be [batch,nodes]")
    return torch.gather(content, 1, permutation[..., None].expand(-1, -1, content.shape[-1]))


def build(candidate, horizon=12, seed=0):
    """Baseline/C/E/CE construction with the frozen independent seed streams."""
    if candidate not in ("baseline", "C", "E", "CE") or horizon < 1:
        raise ValueError("Unknown GIFT construction variant")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = StrongTransferForecaster(
            variant="dynamic", history=12, horizon=horizon, dim=128,
            heads=4, ff_dim=256, layers=3, roles=16, dropout=.1,
            use_profile=False,
        )
        model.calendar_embed = ZeroCalendar()
        for index, block in enumerate(model.blocks):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 3000 + index)
                block.kernel_query = nn.Linear(128, 16)
                block.kernel_key = nn.Linear(128, 16)
            for name in ("assign", "query", "role_key", "role_value", "role_mixer"):
                delattr(block, name)
    if candidate in ("C", "CE"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 702_000)
            model.statistics_encoder = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 128),
            )
            nn.init.zeros_(model.statistics_encoder[-1].weight)
            nn.init.zeros_(model.statistics_encoder[-1].bias)
    if candidate in ("E", "CE"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 704_000)
            model.history_predictor = nn.Linear(3, 1)
            model.deviation_encoder = nn.Sequential(
                nn.Linear(9, 128), nn.GELU(), nn.LayerNorm(128),
            )
            model.deviation_projection = nn.Linear(128, 128)
            nn.init.zeros_(model.deviation_projection.weight)
            nn.init.zeros_(model.deviation_projection.bias)
    return model
