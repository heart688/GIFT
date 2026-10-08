"""Versioned GIFT models for the fixed WWW experiment registry.

Extracted from the project's local frozen experiment file
``ccfa-workfiles/experiments/gift_ce_www_full_20261006/models.py``
(SHA256 16629612a981d53daeda3ab7be4d61ddd21f08b635b085b2e52a953ba4b03f5c).
The forward formulas, parameter keys, and adaptation policies are retained.
The local compatibility layer preserves discarded construction-time RNG draws.
Road-constrained communication consumes sparse row lists.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import _compat
from ._compat import DynamicLatentRouter, FeedForward, StrongTransferForecaster

ALIASES = {"BASE": "GIFT_BASE", "baseline": "GIFT_BASE", "C": "C_ONLY", "E": "E_ONLY",
           "CE": "GIFT_CE", "CE_GIK": "GIFT_CE", "CE_DENSE_1H": "DENSE_1H",
           "CE_GRAPH_RING8": "GRAPH_CONSTRAINED"}
VARIANTS = ("GIFT_BASE", "C_ONLY", "E_ONLY", "GIFT_CE", "LOCAL_INFO_MATCHED",
            "GRAPH_CONSTRAINED", "DENSE_1H", "UNIFORM_E")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_from_dense(graph, metadata=None):
    array = graph.detach().cpu().numpy() if torch.is_tensor(graph) else np.asarray(graph)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not np.isfinite(array).all():
        raise ValueError("Road adjacency must be a finite square matrix")
    allowed = array > 0
    np.fill_diagonal(allowed, True)
    rows, columns = np.nonzero(allowed)
    counts = np.bincount(rows, minlength=len(array))
    row_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return {"nodes": len(array), "row_ptr": torch.from_numpy(row_ptr),
            "col_index": torch.from_numpy(columns.astype(np.int64)),
            "metadata": {"binary_positive_edges": True, "self_added": True,
                         "orientation": "receiver_row_sender_column", **(metadata or {})}}


def load_road_graph(manifest, city):
    if isinstance(manifest, (str, Path)):
        import json
        manifest = json.loads(Path(manifest).read_text())
    spec = manifest["cities"][city]
    path = Path(manifest["root"]) / spec["graph_file"]
    digest = _file_sha(path)
    if digest != spec["graph_sha256"]:
        raise ValueError("Road graph SHA mismatch: " + city)
    graph = np.load(path)
    if len(graph) != spec["expected_shape"][1]:
        raise ValueError("Road graph node count/order metadata differs: " + city)
    return graph_from_dense(graph, {"city": city, "graph_file": str(path), "graph_sha256": digest,
                                    "node_order": "unchanged audited dataset node order",
                                    "affinity_weights_used": False})


def ring_graph(nodes, neighbours=8):
    if nodes < 1 or neighbours != 8:
        raise ValueError("The prescribed synthetic graph uses eight neighbours and self")
    columns, row_ptr = [], [0]
    for receiver in range(nodes):
        sender = sorted({receiver, *[(receiver + offset) % nodes for offset in (-4, -3, -2, -1, 1, 2, 3, 4)]})
        columns.extend(sender)
        row_ptr.append(len(columns))
    return {"nodes": nodes, "row_ptr": torch.tensor(row_ptr, dtype=torch.long),
            "col_index": torch.tensor(columns, dtype=torch.long),
            "metadata": {"synthetic": True, "definition": "degree-eight symmetric ring plus self"}}


class ZeroCalendar(nn.Module):
    def forward(self, value):
        raise RuntimeError("Calendar is not part of any WWW model input")


def _new_block(width, rank, ffn, seed, index):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 810_000 + index)
        block = DynamicLatentRouter(width, rank, ffn, .1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 3000 + index)
        block.kernel_query = nn.Linear(width, rank)
        block.kernel_key = nn.Linear(width, rank)
    for name in ("assign", "query", "role_key", "role_value", "role_mixer"):
        delattr(block, name)
    return block


def _variable_reference(horizon, seed, width, rank, blocks, ffn):
    if width == 128 and ffn == 256:
        model = _compat.build("baseline", horizon=horizon, seed=seed)
        if blocks < 3:
            model.blocks = nn.ModuleList(list(model.blocks)[:blocks])
        elif blocks > 3:
            for index in range(3, blocks):
                model.blocks.append(_new_block(width, rank, ffn, seed, index))
        if rank != 16:
            for index, block in enumerate(model.blocks):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 3000 + index)
                    block.kernel_query = nn.Linear(width, rank)
                    block.kernel_key = nn.Linear(width, rank)
        return model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = StrongTransferForecaster(variant="dynamic", history=12, horizon=horizon, dim=width,
                                        heads=4, ff_dim=ffn, layers=blocks, roles=rank, dropout=.1,
                                        use_profile=False)
    model.calendar_embed = ZeroCalendar()
    for index, block in enumerate(model.blocks):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 3000 + index)
            block.kernel_query = nn.Linear(width, rank)
            block.kernel_key = nn.Linear(width, rank)
        for name in ("assign", "query", "role_key", "role_value", "role_mixer"):
            delattr(block, name)
    return model


def _add_state(model, width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 702_000)
        model.statistics_encoder = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, width))
        nn.init.zeros_(model.statistics_encoder[-1].weight)
        nn.init.zeros_(model.statistics_encoder[-1].bias)


def _add_deviation(model, width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 704_000)
        model.history_predictor = nn.Linear(3, 1)
        model.deviation_encoder = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width))
        model.deviation_projection = nn.Linear(width, width)
        nn.init.zeros_(model.deviation_projection.weight)
        nn.init.zeros_(model.deviation_projection.bias)


class WWWForecaster(nn.Module):
    def _remember(self, key, value):
        if self.capture_diagnostics:
            self.last_diagnostics[key] = value.detach()

    def set_graph(self, graph):
        if not isinstance(graph, dict):
            graph = graph_from_dense(graph)
        count = int(graph["nodes"])
        row_ptr = torch.as_tensor(graph.get("row_ptr", graph.get("rowptr")), dtype=torch.long).cpu()
        columns = torch.as_tensor(graph.get("col_index", graph.get("columns")), dtype=torch.long).cpu()
        if row_ptr.shape != (count + 1,) or row_ptr[0] != 0 or row_ptr[-1] != len(columns):
            raise ValueError("Invalid sparse graph row pointers")
        degrees = row_ptr.diff()
        if bool((degrees < 1).any()) or bool((columns < 0).any()) or bool((columns >= count).any()):
            raise ValueError("Invalid sparse graph neighbours")
        for row in range(count):
            neighbours = columns[row_ptr[row]:row_ptr[row + 1]]
            if row not in neighbours.tolist() or len(torch.unique(neighbours)) != len(neighbours):
                raise ValueError("Graph requires unique neighbours and self for each receiver")
        for name in getattr(self, "_graph_buffer_names", []):
            delattr(self, name)
        device = next(self.parameters()).device
        self._graph_buffer_names, self._graph_bucket_names = [], []
        all_rows = []
        for bucket, degree in enumerate(torch.unique(degrees, sorted=True).tolist()):
            rows = torch.nonzero(degrees == degree, as_tuple=False).flatten()
            neighbours = torch.stack([columns[row_ptr[row]:row_ptr[row + 1]] for row in rows])
            row_name, col_name = f"_road_rows_{bucket}", f"_road_cols_{bucket}"
            self.register_buffer(row_name, rows.to(device), persistent=False)
            self.register_buffer(col_name, neighbours.to(device), persistent=False)
            self._graph_buffer_names.extend((row_name, col_name))
            self._graph_bucket_names.append((row_name, col_name))
            all_rows.append(rows)
        inverse = torch.argsort(torch.cat(all_rows))
        self.register_buffer("_road_inverse_order", inverse.to(device), persistent=False)
        self._graph_buffer_names.append("_road_inverse_order")
        self.graph_nodes, self.graph_edges = count, len(columns)
        self.graph_metadata = graph.get("metadata", {})
        return self

    def _road_read(self, query, key, value):
        if query.shape[1] != getattr(self, "graph_nodes", None):
            raise ValueError("The source/target correct road graph must be set before forward")
        pieces = []
        for row_name, col_name in self._graph_bucket_names:
            rows, columns = getattr(self, row_name), getattr(self, col_name)
            receive = query.index_select(1, rows)
            send_key = key[:, columns]
            send_value = value[:, columns]
            score = (receive.unsqueeze(2) * send_key).sum(-1)
            pieces.append((score.unsqueeze(-1) * send_value).sum(2) /
                          score.sum(2, keepdim=True).clamp_min(1e-6))
        return torch.cat(pieces, dim=1).index_select(1, self._road_inverse_order)

    def forward(self, data, profile=None):
        if data.ndim != 4 or data.shape[1] != 12:
            raise ValueError("Expected [batch,12,nodes,channels] speed history")
        self.last_aux, self.last_diagnostics, self.last_interaction = {}, {}, []
        speed = data[..., 0].transpose(1, 2)
        mean = speed.mean(-1, keepdim=True)
        std = speed.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        history = (speed - mean) / std
        x = self.history_embed(history)
        self.last_aux.update(window_mean=mean, window_std=std)
        self._remember("history", history)
        self._remember("window_mean", mean)
        self._remember("window_std", std)
        state, content = None, None
        if hasattr(self, "statistics_encoder"):
            statistics = torch.cat((mean, std.log()), dim=-1)
            state = self.statistics_encoder(statistics)
            self._remember("statistics", statistics)
            self._remember("statistics_embedding", state)
        if hasattr(self, "history_predictor"):
            predicted_history = self.history_predictor(speed.unfold(-1, 3, 1)[..., :9, :]).squeeze(-1)
            residual = (speed[..., 3:] - predicted_history) / std
            hmask = _compat._history_mask(self.history_valid_mask, data)
            valid = hmask.unfold(-1, 4, 1).all(-1)
            residual = torch.where(valid, residual, torch.zeros_like(residual))
            content = self.deviation_projection(self.deviation_encoder(residual))
            self.last_aux.update(history_prediction=predicted_history, history_target=speed[..., 3:],
                                 history_prediction_valid=valid)
            self._remember("history_prediction", predicted_history)
            self._remember("actual_deviation", residual)
            self._remember("deviation", residual)
            self._remember("chosen_content", residual)
            self._remember("content_valid", valid)
            self._remember("deviation_value", content)
            difference = (speed[..., 3:] - speed[..., 2:-1]) / std
            self._remember("difference", torch.where(valid, difference, torch.zeros_like(difference)))
        for index, block in enumerate(self.blocks):
            z = block.norm1(x)
            extra_state = state if index == 0 and self.enable_statistics else None
            extra_content = content if index == 0 and self.enable_deviation_content else None
            if index == 0 and content is not None and extra_content is None:
                self._remember("used_deviation_value", torch.zeros_like(content))
            if self.formal_variant == "LOCAL_INFO_MATCHED":
                context = block.value(z if extra_state is None else z + extra_state)
                if extra_content is not None:
                    used_content = _compat._paired_content(extra_content, self.content_permutation, speed)
                    context = context + used_content
                    self._remember("used_deviation_value", used_content)
                query = key = query_logits = key_logits = None
                value = context
            else:
                value = block.value(z)
                if extra_content is not None:
                    used_content = _compat._paired_content(extra_content, self.content_permutation, speed)
                    value = value + used_content
                    self._remember("used_deviation_value", used_content)
                if self.formal_variant == "UNIFORM_E":
                    context = value.mean(1, keepdim=True).expand_as(value)
                    query = key = query_logits = key_logits = None
                else:
                    query_logits = block.kernel_query(z)
                    key_logits = block.kernel_key(z)
                    if extra_state is not None:
                        new_query = block.kernel_query(z + extra_state)
                        key_state = _compat._paired_content(extra_state, self.stats_permutation, speed)
                        new_key = block.kernel_key(z + key_state)
                        self._remember("query_delta", new_query - query_logits)
                        self._remember("key_delta", new_key - key_logits)
                        self._remember("used_statistics_embedding", key_state)
                        query_logits, key_logits = new_query, new_key
                    elif index == 0 and state is not None:
                        self._remember("query_delta", torch.zeros_like(query_logits))
                        self._remember("key_delta", torch.zeros_like(key_logits))
                        self._remember("used_statistics_embedding", torch.zeros_like(state))
                    query, key = F.elu(query_logits) + 1, F.elu(key_logits) + 1
                    if self.formal_variant == "DENSE_1H":
                        # Standard SDPA, one head, rank-width Q/K and width-d V.
                        # Math backend provides deterministic gradients and
                        # supports the differing QK and V feature dimensions.
                        manager = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                                 enable_mem_efficient=False)
                                   if data.device.type == "cuda" else nullcontext())
                        with manager:
                            context = F.scaled_dot_product_attention(query_logits.unsqueeze(1),
                                                                     key_logits.unsqueeze(1), value.unsqueeze(1),
                                                                     dropout_p=0.0).squeeze(1)
                        self.last_sdpa_backend = "torch_SDPA_math_one_head"
                    elif self.formal_variant == "GRAPH_CONSTRAINED":
                        context = self._road_read(query, key, value)
                    else:
                        context, moment, mass = _compat._read(query, key, value)
                        if index == 0:
                            self._remember("first_moment", moment)
                            self._remember("first_mass", mass)
            if self.disable_context:
                context = torch.zeros_like(context)
            if self.capture_diagnostics:
                for name, tensor in (("query", query), ("key", key), ("value", value), ("context", context)):
                    if tensor is not None:
                        self._remember(f"block{index}_{name}", tensor)
                        if index == 0:
                            self._remember("first_" + name, tensor)
                self.last_interaction.append({"query": None if query is None else query.detach(),
                                              "key": None if key is None else key.detach(), "value": value.detach(),
                                              "context": context.detach(),
                                              "query_logits": None if query_logits is None else query_logits.detach(),
                                              "key_logits": None if key_logits is None else key_logits.detach()})
            x = x + torch.sigmoid(block.spatial_scale) * block.drop(block.out(context))
            x = x + block.ffn(block.norm2(x))
        return self.head(self.final_norm(x)) * std + mean

    def auxiliary_losses(self, data, target=None, valid_history=None, valid_target=None):
        if not hasattr(self, "history_predictor"):
            return {}
        speed = data[..., 0].transpose(1, 2)
        mask = _compat._history_mask(valid_history, data).unfold(-1, 4, 1).all(-1)
        return {"history": _compat._masked_mae(self.last_aux["history_prediction"], speed[..., 3:], mask)}


def build(variant, horizon=12, seed=0, width=128, rank=16, blocks=3, ffn_width=256):
    requested = variant
    variant = ALIASES.get(variant, variant)
    if variant not in VARIANTS or width < 1 or rank < 1 or blocks < 1 or horizon < 1:
        raise ValueError("Invalid fixed model definition")
    is_default = (width, rank, blocks, ffn_width) == (128, 16, 3, 256)
    legacy_variant = {"GIFT_BASE": "baseline", "C_ONLY": "C", "E_ONLY": "E",
                      "GIFT_CE": "CE", "LOCAL_INFO_MATCHED": "CE",
                      "GRAPH_CONSTRAINED": "CE", "DENSE_1H": "CE", "UNIFORM_E": "E"}[variant]
    if is_default:
        model = _compat.build(legacy_variant, horizon=horizon, seed=seed)
    else:
        model = _variable_reference(horizon, seed, width, rank, blocks, ffn_width)
        if legacy_variant in ("C", "CE"):
            _add_state(model, width, seed)
        if legacy_variant in ("E", "CE"):
            _add_deviation(model, width, seed)
    if variant == "LOCAL_INFO_MATCHED":
        for index, block in enumerate(model.blocks):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 830_000 + index)
                block.ffn = FeedForward(width, 272, .1)
    if variant in ("LOCAL_INFO_MATCHED", "UNIFORM_E"):
        for block in model.blocks:
            del block.kernel_query
            del block.kernel_key
    model.__class__ = WWWForecaster
    model.formal_variant = variant
    model.variant = variant
    model.requested_variant = requested
    model.screening_candidate = legacy_variant
    model.width, model.rank, model.block_count = width, rank, blocks
    model.actual_ffn_width = 272 if variant == "LOCAL_INFO_MATCHED" else ffn_width
    model.enable_statistics = True
    model.enable_deviation_content = True
    model.disable_context = False
    model.stats_permutation = None
    model.content_permutation = None
    model.history_valid_mask = None
    model.capture_diagnostics = False
    model.last_aux, model.last_diagnostics, model.last_interaction = {}, {}, []
    return model


def communication_roles(model):
    roles = {}
    for index, block in enumerate(model.blocks):
        for component in ("kernel_query", "kernel_key", "value", "out"):
            if hasattr(block, component):
                for parameter in getattr(block, component).parameters():
                    roles[id(parameter)] = f"block{index}.{component}"
        roles[id(block.spatial_scale)] = f"block{index}.scalar_gate"
    if hasattr(model, "deviation_projection"):
        for parameter in model.deviation_projection.parameters():
            roles[id(parameter)] = "deviation_value_projection"
    return roles


def configure_policy(model, policy="DEFAULT_CE", seed=0):
    policy = {"DEFAULT": "DEFAULT_CE", "FROZEN": "FROZEN_TWO_DAY", "SOURCE": "FULL"}.get(policy, policy)
    if policy not in ("FULL", "HEAD_ONLY", "INTERACTION_ONLY", "RESET_QK", "DEFAULT_CE", "FROZEN_TWO_DAY", "ZERO_DAY"):
        raise ValueError("Unknown adaptation policy")
    if policy == "RESET_QK":
        devices = sorted({p.device.index if p.device.index is not None else torch.cuda.current_device()
                          for p in model.parameters() if p.device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 920_000)
            for block in model.blocks:
                block.kernel_query.reset_parameters()
                block.kernel_key.reset_parameters()
        model.reset_qk_seed = seed + 920_000
    roles = communication_roles(model)
    head = {id(parameter) for parameter in model.head.parameters()}
    for parameter in model.parameters():
        if policy == "FULL":
            train = True
        elif policy == "HEAD_ONLY":
            train = id(parameter) in head
        elif policy == "INTERACTION_ONLY":
            train = id(parameter) in roles
        elif policy in ("FROZEN_TWO_DAY", "ZERO_DAY"):
            train = False
        else:
            train = id(parameter) not in roles
        parameter.requires_grad_(train)
    model.adaptation_policy_name = policy
    return parameter_audit(model)


def parameter_audit(model):
    roles, records = communication_roles(model), OrderedDict()
    total = trainable = 0
    trainable_names, frozen_names = [], []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        trainable += count if parameter.requires_grad else 0
        (trainable_names if parameter.requires_grad else frozen_names).append(name)
        records[name] = {"shape": list(parameter.shape), "count": count, "trainable": parameter.requires_grad,
                         "role": roles.get(id(parameter), "representation_or_prediction"), "effective": True}
    return {"variant": model.formal_variant, "policy": getattr(model, "adaptation_policy_name", "FULL"),
            "horizon": model.horizon, "width": model.width, "rank": model.rank, "blocks": len(model.blocks),
            "ffn_width": model.actual_ffn_width, "total_parameters": total, "effective_parameters": total,
            "trainable_parameters": trainable, "frozen_parameters": total - trainable,
            "trainable_names": trainable_names, "frozen_names": frozen_names, "parameters": records,
            "communication_backend": "degree_bucket_sparse_kernel" if model.formal_variant == "GRAPH_CONSTRAINED" else
                                     "torch_SDPA_math_one_head" if model.formal_variant == "DENSE_1H" else
                                     "local_information" if model.formal_variant == "LOCAL_INFO_MATCHED" else
                                     "uniform_value_mean" if model.formal_variant == "UNIFORM_E" else "compact_positive_kernel"}


adaptation_policy = configure_policy
adapt_policy = configure_policy
