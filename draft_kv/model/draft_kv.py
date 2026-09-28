"""Core Draft-KV model for latent communication.

The frozen Sharer first produces a draft solution.  This module consumes the
Sharer's exact ``prompt + generated draft`` token sequence, captures KV at
selected Sharer layers, projects every draft-token KV into selected Receiver
layer spaces, and exposes it through a parallel cross-attention branch.

Only the K/V projectors and signed KV-head gates are trainable; both language
models and the copied Receiver Q/K-norm/O path are frozen.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .qwen_split import make_qwen_depth_split


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("model must expose model.layers")
    return layers


def _kv_shape(model: nn.Module, layer_index: int) -> Tuple[int, int]:
    attention = _decoder_layers(model)[int(layer_index)].self_attn
    heads = int(getattr(model.config, "num_key_value_heads"))
    width = int(attention.k_proj.out_features)
    if width % heads:
        raise ValueError("k_proj width is not divisible by num_key_value_heads")
    return heads, width // heads


@dataclass
class LayerKV:
    """One layer of unrotated or cache-space K/V tensors."""

    key: Tensor  # [B, H_kv, T, D]
    value: Tensor  # [B, H_kv, T, D]


def _reshape_projection(output: Tensor, heads: int) -> Tensor:
    if output.ndim != 3 or output.shape[-1] % int(heads):
        raise ValueError("projection output must have shape [B,T,H*D]")
    batch, length, width = output.shape
    return output.view(batch, length, int(heads), width // int(heads))


def capture_unrotated_kv(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    layer_indices: Iterable[int],
) -> Dict[int, LayerKV]:
    """Capture selected prompt K/V tensors without retaining backbone graphs."""

    requested = tuple(sorted({int(index) for index in layer_indices}))
    layers = _decoder_layers(model)
    if not requested:
        raise ValueError("at least one layer must be requested")
    if requested[0] < 0 or requested[-1] >= len(layers):
        raise ValueError("requested KV layer is outside the model")

    captured_k: Dict[int, Tensor] = {}
    captured_v: Dict[int, Tensor] = {}
    handles = []

    def save_into(store: Dict[int, Tensor], index: int):
        def hook(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: Tensor):
            store[index] = output.detach()

        return hook

    for index in requested:
        attention = layers[index].self_attn
        handles.append(attention.k_proj.register_forward_hook(save_into(captured_k, index)))
        handles.append(attention.v_proj.register_forward_hook(save_into(captured_v, index)))

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    result: Dict[int, LayerKV] = {}
    for index in requested:
        if index not in captured_k or index not in captured_v:
            raise RuntimeError(f"failed to capture layer {index} K/V")
        heads, _ = _kv_shape(model, index)
        key = _reshape_projection(captured_k[index], heads)
        value = _reshape_projection(captured_v[index], heads)
        key_norm = getattr(layers[index].self_attn, "k_norm", None)
        if key_norm is not None:
            with torch.no_grad():
                key = key_norm(key)
        result[index] = LayerKV(
            key=key.transpose(1, 2).contiguous().detach(),
            value=value.transpose(1, 2).contiguous().detach(),
        )
    return result


@dataclass
class DraftKVPacket:
    """Projected Sharer draft memory.

    ``attention_mask`` is one only at generated-draft positions.  Prompt and
    padding positions may remain in the tensors for efficient batching but
    are never visible to the Receiver cross-attention.
    """

    layers: Dict[int, LayerKV]
    attention_mask: Tensor


class DraftKVProjector(nn.Module):
    """Bias-free, per-token projection across the complete KV-head width."""

    def __init__(
        self,
        source_heads: int,
        source_head_dim: int,
        target_heads: int,
        target_head_dim: int,
    ) -> None:
        super().__init__()
        self.source_heads = int(source_heads)
        self.source_head_dim = int(source_head_dim)
        self.target_heads = int(target_heads)
        self.target_head_dim = int(target_head_dim)
        source_width = self.source_heads * self.source_head_dim
        target_width = self.target_heads * self.target_head_dim
        self.key = nn.Linear(source_width, target_width, bias=False)
        self.value = nn.Linear(source_width, target_width, bias=False)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def _project(self, tensor: Tensor, layer: nn.Linear) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError("KV tensor must have shape [B,H,T,D]")
        if (
            tensor.shape[1] != self.source_heads
            or tensor.shape[-1] != self.source_head_dim
        ):
            raise ValueError("source KV shape does not match projector")
        batch, _, length, _ = tensor.shape
        flattened = tensor.transpose(1, 2).reshape(batch, length, -1)
        projected = layer(flattened)
        return (
            projected.view(
                batch,
                length,
                self.target_heads,
                self.target_head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, source: LayerKV) -> LayerKV:
        return LayerKV(
            key=self._project(source.key, self.key),
            value=self._project(source.value, self.value),
        )


class DraftKVProjectionStack(nn.Module):
    """One independent task-trained K/V projector per Receiver target layer."""

    def __init__(
        self,
        receiver: nn.Module,
        sharer: nn.Module,
        layer_mapping: Mapping[int, int],
    ) -> None:
        super().__init__()
        self.layer_mapping = {
            int(target): int(source) for target, source in layer_mapping.items()
        }
        if not self.layer_mapping:
            raise ValueError("layer_mapping cannot be empty")
        receiver_layers = _decoder_layers(receiver)
        sharer_layers = _decoder_layers(sharer)
        if min(self.layer_mapping) < 0 or max(self.layer_mapping) >= len(receiver_layers):
            raise ValueError("Receiver layer mapping is outside the model")
        sources = tuple(self.layer_mapping.values())
        if min(sources) < 0 or max(sources) >= len(sharer_layers):
            raise ValueError("Sharer layer mapping is outside the model")

        projectors: Dict[str, DraftKVProjector] = {}
        for target, source in self.layer_mapping.items():
            source_heads, source_dim = _kv_shape(sharer, source)
            target_heads, target_dim = _kv_shape(receiver, target)
            projectors[str(target)] = DraftKVProjector(
                source_heads,
                source_dim,
                target_heads,
                target_dim,
            )
        self.projectors = nn.ModuleDict(projectors)

    @property
    def target_layers(self) -> Tuple[int, ...]:
        return tuple(sorted(self.layer_mapping))

    @property
    def source_layers(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.layer_mapping.values())))

    def forward(self, source_layers: Mapping[int, LayerKV]) -> Dict[int, LayerKV]:
        return {
            target: self.projectors[str(target)](source_layers[source_index])
            for target, source_index in self.layer_mapping.items()
        }


class DraftKVCrossAttention(nn.Module):
    """Frozen Receiver Q/K-norm/O path plus trainable signed KV-head gates.

    Q and K intentionally remain unrotated.  The Receiver decoding sequence
    and Sharer draft are separate sequences, so a self-attention relative RoPE
    offset would not have a well-defined meaning here.
    """

    def __init__(
        self,
        receiver_layer: nn.Module,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        attention = receiver_layer.self_attn
        self.q_proj = copy.deepcopy(attention.q_proj)
        self.q_norm = copy.deepcopy(getattr(attention, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(attention, "k_norm", nn.Identity()))
        self.o_proj = copy.deepcopy(attention.o_proj)
        for module in (self.q_proj, self.q_norm, self.k_norm, self.o_proj):
            module.requires_grad_(False)
            module.eval()

        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.q_proj.out_features % self.num_query_heads:
            raise ValueError("q_proj width must divide over query heads")
        self.head_dim = self.q_proj.out_features // self.num_query_heads
        self.groups = self.num_query_heads // self.num_kv_heads
        self.gate_logits = nn.Parameter(torch.zeros(self.num_kv_heads))

    def train(self, mode: bool = True) -> "DraftKVCrossAttention":
        super().train(mode)
        for module in (self.q_proj, self.q_norm, self.k_norm, self.o_proj):
            module.eval()
        return self

    def forward(
        self,
        normalized_hidden_states: Tensor,
        packet: LayerKV,
        packet_mask: Tensor,
        *,
        communication_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch, length, _ = normalized_hidden_states.shape
        query = self.q_proj(normalized_hidden_states)
        query = query.view(batch, length, self.num_query_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        key = packet.key
        value = packet.value
        if key.shape != value.shape:
            raise ValueError("projected communication K/V shapes differ")
        if key.shape[0] != batch:
            raise ValueError("packet and Receiver batch sizes differ")
        if key.shape[1] != self.num_kv_heads or key.shape[-1] != self.head_dim:
            raise ValueError("communication key shape does not match Receiver GQA")
        if packet_mask.shape != (batch, key.shape[2]):
            raise ValueError("communication mask shape does not match packet")
        if not bool(packet_mask.to(dtype=torch.bool).any(dim=1).all()):
            raise ValueError("each communication packet must contain a draft token")

        # Receiver k_norm is defined on the head dimension.  Applying it after
        # projection gives the task-trained K a Receiver-native scale without
        # imposing tokenwise representation alignment.
        key = self.k_norm(key.transpose(1, 2)).transpose(1, 2)
        key = key.repeat_interleave(self.groups, dim=1)
        value = value.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores * (self.head_dim**-0.5)
        visible = packet_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(
            ~visible[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attended = torch.matmul(weights, value)
        scale = float(communication_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("communication_scale must be finite and non-negative")
        gates = (
            torch.tanh(self.gate_logits).repeat_interleave(self.groups) * scale
        )
        attended = attended * gates.to(attended.dtype)[None, :, None, None]
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        update = self.o_proj(attended)

        diagnostics: Dict[str, Tensor] = {}
        if return_diagnostics:
            update_token_rms = torch.sqrt(
                update.float().square().mean(dim=2)
            ).detach()
            input_token_rms = torch.sqrt(
                normalized_hidden_states.float().square().mean(dim=2)
            ).detach()
            diagnostics = {
                "gate": torch.tanh(self.gate_logits).detach(),
                "attention": weights.mean(dim=1).detach(),
                "update_rms": torch.sqrt(
                    update.float().square().mean(dim=(1, 2))
                ).detach(),
                # Tokenwise values let evaluators remove Receiver padding and
                # compare communication scale across domains without changing
                # the forward path.  They are diagnostics only.
                "update_token_rms": update_token_rms,
                "input_token_rms": input_token_rms,
                "relative_update_token_rms": (
                    update_token_rms / input_token_rms.clamp_min(1e-12)
                ),
            }
        return update, diagnostics


class DraftKVConsumerStack(nn.Module):
    """Dispatch one Draft-KV cross-attention branch per selected layer."""

    def __init__(self, receiver: nn.Module, target_layers: Sequence[int]) -> None:
        super().__init__()
        query_heads = int(receiver.config.num_attention_heads)
        kv_heads = int(receiver.config.num_key_value_heads)
        layers = _decoder_layers(receiver)
        self.external = nn.ModuleDict(
            {
                str(index): DraftKVCrossAttention(
                    layers[index], query_heads, kv_heads
                )
                for index in sorted({int(value) for value in target_layers})
            }
        )
        self.active_layer: Optional[int] = None

    def set_active_layer(self, layer: int) -> None:
        self.active_layer = int(layer)

    @property
    def consumer_layers(self) -> Tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.external))

    @property
    def gate_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.external.values():
            yield module.gate_logits

    def parallel_forward(
        self,
        normalized_hidden_states: Tensor,
        *,
        packet: Optional[DraftKVPacket],
        position_embeddings: Tuple[Tensor, Tensor],
        disable_communication: bool = False,
        communication_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del position_embeddings
        if self.active_layer is None:
            raise RuntimeError("active Receiver layer was not set")
        key = str(self.active_layer)
        if disable_communication or packet is None or key not in self.external:
            return torch.zeros_like(normalized_hidden_states), {}
        update, diagnostics = self.external[key](
            normalized_hidden_states,
            packet.layers[self.active_layer],
            packet.attention_mask,
            communication_scale=communication_scale,
            return_diagnostics=return_diagnostics,
        )
        return update, {f"draft_kv_{name}": value for name, value in diagnostics.items()}

    def post_attention_forward(
        self,
        hidden_states: Tensor,
        *,
        packet: Optional[DraftKVPacket],
        position_embeddings: Tuple[Tensor, Tensor],
        disable_communication: bool = False,
        communication_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del (
            packet,
            position_embeddings,
            disable_communication,
            communication_scale,
            return_diagnostics,
        )
        return hidden_states, {}


class DraftKVModel(nn.Module):
    """Frozen two-model wrapper with task-trained Sharer-draft KV memory."""

    def __init__(
        self,
        receiver: nn.Module,
        sharer: nn.Module,
        layer_mapping: Mapping[int, int],
    ) -> None:
        super().__init__()
        self.receiver = receiver
        self.sharer = sharer
        self.receiver_split = make_qwen_depth_split(receiver)
        self.projection = DraftKVProjectionStack(receiver, sharer, layer_mapping)
        self.consumer = DraftKVConsumerStack(
            receiver, self.projection.target_layers
        )
        self.receiver.requires_grad_(False)
        self.sharer.requires_grad_(False)
        self.receiver.eval()
        self.sharer.eval()

    @property
    def device(self) -> torch.device:
        return next(self.receiver.parameters()).device

    def train(self, mode: bool = True) -> "DraftKVModel":
        super().train(mode)
        self.receiver.eval()
        self.sharer.eval()
        return self

    def set_stage(self, stage: str) -> None:
        if stage not in {"communication", "eval"}:
            raise ValueError("stage must be communication or eval")
        for parameter in self.projection.parameters():
            parameter.requires_grad_(stage == "communication")
        for parameter in self.consumer.gate_parameters:
            parameter.requires_grad_(stage == "communication")
        if stage == "eval":
            self.eval()
        else:
            self.train()

    @property
    def communication_parameters(self) -> Iterable[nn.Parameter]:
        for parameter in self.projection.parameters():
            if parameter.requires_grad:
                yield parameter
        for parameter in self.consumer.gate_parameters:
            if parameter.requires_grad:
                yield parameter

    def make_packet(
        self,
        sharer_input_ids: Tensor,
        sharer_attention_mask: Tensor,
        sharer_draft_mask: Tensor,
        *,
        detach: bool = False,
    ) -> DraftKVPacket:
        sharer_input_ids = sharer_input_ids.to(self.device)
        sharer_attention_mask = sharer_attention_mask.to(self.device)
        sharer_draft_mask = sharer_draft_mask.to(self.device)
        if sharer_input_ids.shape != sharer_attention_mask.shape:
            raise ValueError("Sharer IDs and attention mask shapes differ")
        if sharer_input_ids.shape != sharer_draft_mask.shape:
            raise ValueError("Sharer IDs and draft mask shapes differ")
        invalid = sharer_draft_mask.to(dtype=torch.bool) & ~sharer_attention_mask.to(
            dtype=torch.bool
        )
        if bool(invalid.any()):
            raise ValueError("draft mask includes padding positions")
        if not bool(sharer_draft_mask.to(dtype=torch.bool).any(dim=1).all()):
            raise ValueError("each Sharer sequence must contain a draft token")

        source = capture_unrotated_kv(
            self.sharer,
            sharer_input_ids,
            sharer_attention_mask,
            self.projection.source_layers,
        )
        projected = self.projection(source)
        packet = DraftKVPacket(
            layers=projected,
            attention_mask=sharer_draft_mask,
        )
        if not detach:
            return packet
        return DraftKVPacket(
            layers={
                layer: LayerKV(kv.key.detach(), kv.value.detach())
                for layer, kv in packet.layers.items()
            },
            attention_mask=packet.attention_mask.detach(),
        )

    def forward_receiver(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        packet: Optional[DraftKVPacket],
        disable_communication: bool = False,
        communication_scale: float = 1.0,
        return_diagnostics: bool = False,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Any], Dict[str, Any]]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        consumer_layers = self.consumer.consumer_layers
        first_consumer = min(consumer_layers)
        with torch.no_grad():
            front = self.receiver_split.forward_front(
                input_ids=input_ids,
                attention_mask=attention_mask,
                request_layer=first_consumer - 1,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        front.hidden_states = front.hidden_states.detach()
        back = self.receiver_split.forward_back(
            front,
            consumer=self.consumer,
            consumer_layers=consumer_layers,
            consumer_kwargs={
                "packet": packet,
                "position_embeddings": front.position_embeddings,
                "disable_communication": disable_communication,
                "communication_scale": communication_scale,
                "return_diagnostics": return_diagnostics,
            },
            use_cache=use_cache,
        )
        return back.logits, back.cache, back.diagnostics

    def trainable_parameter_names(self) -> Tuple[str, ...]:
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )


__all__ = [
    "DraftKVCrossAttention",
    "DraftKVConsumerStack",
    "DraftKVPacket",
    "DraftKVProjectionStack",
    "DraftKVProjector",
    "DraftKVModel",
]
