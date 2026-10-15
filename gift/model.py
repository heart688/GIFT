"""GIFT: temporal deviation encoding and graph-free information exchange."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_width, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionBlock(nn.Module):
    def __init__(self, width: int, rank: int, hidden_width: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        # Keep the original parameter initialization stream for the published seeds.
        # These temporary layers have no role in the GIFT computation.
        nn.Linear(width, rank)
        self.value = nn.Linear(width, width)
        nn.Linear(width, width)
        nn.Linear(width, width)
        nn.Linear(width, width)
        nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_width, width),
        )
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = FeedForward(width, hidden_width, dropout)
        self.drop = nn.Dropout(dropout)
        self.spatial_scale = nn.Parameter(torch.tensor(-2.0))


class GIFT(nn.Module):
    """Shared node-wise encoder with state-informed kernel communication."""

    def __init__(self, horizon: int, seed: int = 0, width: int = 128,
                 rank: int = 16, blocks: int = 3, ffn_width: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        if horizon < 1 or width < 1 or rank < 1 or blocks < 1:
            raise ValueError("Model dimensions must be positive")
        self.history, self.horizon, self.width, self.rank = 12, horizon, width, rank
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.history_embed = nn.Sequential(
                nn.Linear(12, width), nn.GELU(), nn.LayerNorm(width),
            )
            # Preserve the released model's subsequent initialization values.
            nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            self.blocks = nn.ModuleList(
                InteractionBlock(width, rank, ffn_width, dropout) for _ in range(blocks)
            )
            self.final_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, horizon)
            for index, block in enumerate(self.blocks):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 3000 + index)
                    block.kernel_query = nn.Linear(width, rank)
                    block.kernel_key = nn.Linear(width, rank)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 702_000)
            self.statistics_encoder = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(), nn.Linear(16, width),
            )
            nn.init.zeros_(self.statistics_encoder[-1].weight)
            nn.init.zeros_(self.statistics_encoder[-1].bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 704_000)
            self.history_predictor = nn.Linear(3, 1)
            self.deviation_encoder = nn.Sequential(
                nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width),
            )
            self.deviation_projection = nn.Linear(width, width)
            nn.init.zeros_(self.deviation_projection.weight)
            nn.init.zeros_(self.deviation_projection.bias)
        self.history_valid_mask = None
        self.last_aux = {}

    def _valid_history(self, data: torch.Tensor) -> torch.Tensor:
        mask = self.history_valid_mask
        if mask is None:
            return torch.ones_like(data[..., 0].transpose(1, 2), dtype=torch.bool)
        if mask.ndim == 4:
            mask = mask[..., 0]
        if tuple(mask.shape) == tuple(data.shape[:3]):
            mask = mask.transpose(1, 2)
        expected = (data.shape[0], data.shape[2], 12)
        if tuple(mask.shape) != expected:
            raise ValueError(f"History mask must have shape {expected}")
        return mask.to(device=data.device, dtype=torch.bool)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim != 4 or data.shape[1] != 12:
            raise ValueError("Expected [batch, 12, nodes, channels] history")
        speed = data[..., 0].transpose(1, 2)
        mean = speed.mean(-1, keepdim=True)
        std = speed.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        x = self.history_embed((speed - mean) / std)
        state = self.statistics_encoder(torch.cat((mean, std.log()), dim=-1))

        predicted = self.history_predictor(speed.unfold(-1, 3, 1)[..., :9, :]).squeeze(-1)
        valid = self._valid_history(data).unfold(-1, 4, 1).all(-1)
        deviation = (speed[..., 3:] - predicted) / std
        deviation = torch.where(valid, deviation, torch.zeros_like(deviation))
        content = self.deviation_projection(self.deviation_encoder(deviation))
        self.last_aux = {"history_prediction": predicted, "history_target": speed[..., 3:]}

        for index, block in enumerate(self.blocks):
            z = block.norm1(x)
            matched = z + state if index == 0 else z
            query = F.elu(block.kernel_query(matched)) + 1
            key = F.elu(block.kernel_key(matched)) + 1
            value = block.value(z) + content if index == 0 else block.value(z)
            moment = torch.einsum("bnr,bnd->brd", key, value)
            mass = key.sum(dim=1)
            numerator = torch.einsum("bnr,brd->bnd", query, moment)
            denominator = torch.einsum("bnr,br->bn", query, mass)
            context = numerator / denominator.unsqueeze(-1).clamp_min(1e-6)
            x = x + torch.sigmoid(block.spatial_scale) * block.drop(block.out(context))
            x = x + block.ffn(block.norm2(x))
        return self.head(self.final_norm(x)) * std + mean

    def auxiliary_loss(self, valid_history: torch.Tensor) -> torch.Tensor:
        predicted = self.last_aux["history_prediction"]
        target = self.last_aux["history_target"]
        if valid_history.ndim != 3 or valid_history.shape != (predicted.shape[0], predicted.shape[1], 12):
            raise ValueError("Expected valid history with shape [batch, nodes, 12]")
        mask = valid_history.unfold(-1, 4, 1).all(-1) & torch.isfinite(target)
        clean = torch.where(mask, target, predicted.detach())
        return (predicted - clean).abs().sum() / mask.sum().clamp_min(1)

    def freeze_communication(self) -> dict:
        """Adapt encoders and forecast layers while retaining source exchange."""
        frozen = set()
        for block in self.blocks:
            for layer in (block.kernel_query, block.kernel_key, block.value, block.out):
                frozen.update(id(parameter) for parameter in layer.parameters())
            frozen.add(id(block.spatial_scale))
        frozen.update(id(parameter) for parameter in self.deviation_projection.parameters())
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) not in frozen)
        return self.parameter_audit()

    def parameter_audit(self) -> dict:
        parameters = list(self.named_parameters())
        return {
            "total": sum(p.numel() for _, p in parameters),
            "trainable": sum(p.numel() for _, p in parameters if p.requires_grad),
            "frozen_names": [name for name, p in parameters if not p.requires_grad],
        }
