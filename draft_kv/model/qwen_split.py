"""Explicit depth-split execution for Qwen2 and Qwen3.

The supported Transformers implementations currently share the same model and
decoder-layer call contracts.  This executor mirrors those native loops while
exposing a pause point between decoder layers.  Dispatch is based on the
explicit ``config.model_type`` value rather than class-name guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache, DynamicCache


@dataclass
class SplitFrontOutput:
    hidden_states: Tensor
    cache: Optional[Cache]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Tensor]
    position_ids: Tensor
    cache_position: Tensor
    position_embeddings: Tuple[Tensor, Tensor]
    request_layer: int
    input_shape: Tuple[int, int]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitBackOutput:
    last_hidden_state: Tensor
    logits: Tensor
    cache: Optional[Cache]
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class QwenDepthSplit:
    """Run a supported Qwen causal LM in front/back depth segments.

    ``request_layer`` is the last layer executed by ``forward_front``.  Thus
    ``request_layer=0`` means layer 0 is in the front segment and
    ``request_layer=-1`` means that only token embeddings are returned.
    """

    def __init__(self, receiver: nn.Module) -> None:
        if not hasattr(receiver, "model") or not hasattr(receiver, "lm_head"):
            raise TypeError("receiver must expose model and lm_head")
        model_type = str(getattr(getattr(receiver, "config", None), "model_type", ""))
        if model_type not in {"qwen2", "qwen3"}:
            raise TypeError(
                "depth split supports only config.model_type qwen2 or qwen3; "
                f"got {model_type!r}"
            )
        self.receiver = receiver
        self.backbone = receiver.model
        self.lm_head = receiver.lm_head
        self.model_type = model_type

    @property
    def config(self):
        return self.receiver.config

    @property
    def num_layers(self) -> int:
        return len(self.backbone.layers)

    def _prepare_inputs(
        self,
        *,
        input_ids: Optional[Tensor],
        inputs_embeds: Optional[Tensor],
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        cache_position: Optional[Tensor],
        past_key_values: Optional[Cache],
        use_cache: bool,
        output_attentions: bool = False,
    ) -> Tuple[
        Tensor,
        Optional[Tensor],
        Tensor,
        Tensor,
        Tuple[Tensor, Tensor],
        Optional[Cache],
    ]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.backbone.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            if inputs_embeds.shape[0] != 1:
                position_ids = position_ids.expand(inputs_embeds.shape[0], -1)

        causal_mask = self.backbone._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        position_embeddings = self.backbone.rotary_emb(inputs_embeds, position_ids)
        return (
            inputs_embeds,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
            past_key_values,
        )

    def forward_front(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        request_layer: int = 0,
        *,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        cache_position: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> SplitFrontOutput:
        if request_layer < -1 or request_layer >= self.num_layers:
            raise ValueError(
                f"request_layer must be in [-1, {self.num_layers - 1}], "
                f"got {request_layer}"
            )
        (
            hidden_states,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
            cache,
        ) = self._prepare_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        for layer in self.backbone.layers[: request_layer + 1]:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
        return SplitFrontOutput(
            hidden_states=hidden_states,
            cache=cache,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            request_layer=request_layer,
            input_shape=(hidden_states.shape[0], hidden_states.shape[1]),
        )

    @staticmethod
    def _run_layer_with_consumer(
        layer: nn.Module,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor],
        position_ids: Tensor,
        cache: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        consumer: Optional[nn.Module],
        consumer_kwargs: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
        if consumer is None:
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return outputs[0], outputs[1] if output_attentions else None, {}

        # Draft-KV consumers can request a true parallel-attention branch.
        # The branch receives the exact normalized states used by native
        # self-attention, so its frozen Receiver Q projection lives in the
        # same coordinate system as the captured Receiver K targets.
        parallel_forward = getattr(consumer, "parallel_forward", None)
        post_attention_forward = getattr(consumer, "post_attention_forward", None)
        if callable(parallel_forward) and callable(post_attention_forward):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            self_output, self_attn_weights = layer.self_attn(
                hidden_states=normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            parallel_update, parallel_diagnostics = parallel_forward(
                normalized, **(consumer_kwargs or {})
            )
            hidden_states = residual + self_output + parallel_update
            hidden_states, post_diagnostics = post_attention_forward(
                hidden_states, **(consumer_kwargs or {})
            )
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            consumer_diagnostics = dict(parallel_diagnostics)
            consumer_diagnostics.update(post_diagnostics)
            return hidden_states, self_attn_weights, consumer_diagnostics

        # This is Qwen3DecoderLayer.forward with the consumer inserted after
        # self-attention residual and before the MLP, as specified by Draft-KV.
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        updated, consumer_diagnostics = consumer(
            hidden_states, **(consumer_kwargs or {})
        )
        hidden_states = updated
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights, consumer_diagnostics

    def forward_back(
        self,
        front_output: Optional[SplitFrontOutput] = None,
        *,
        hidden_states: Optional[Tensor] = None,
        cache: Optional[Cache] = None,
        attention_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        request_layer: Optional[int] = None,
        consumer: Optional[nn.Module] = None,
        consumer_layers: Optional[Tuple[int, ...]] = None,
        consumer_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> SplitBackOutput:
        if front_output is not None:
            hidden_states = front_output.hidden_states
            cache = front_output.cache
            attention_mask = front_output.attention_mask
            causal_mask = front_output.causal_mask
            position_ids = front_output.position_ids
            cache_position = front_output.cache_position
            position_embeddings = front_output.position_embeddings
            request_layer = front_output.request_layer
        if hidden_states is None or request_layer is None:
            raise ValueError("front_output or hidden_states/request_layer is required")
        if position_ids is None or cache_position is None or position_embeddings is None:
            raise ValueError("position_ids, cache_position and RoPE are required")
        # ``causal_mask`` is the mask constructed for this exact call.
        attn_mask = causal_mask if causal_mask is not None else attention_mask
        if (
            attn_mask is not None
            and attn_mask.ndim == 2
            and attn_mask.dtype not in (torch.bool, hidden_states.dtype)
        ):
            attn_mask = attn_mask.to(dtype=hidden_states.dtype)
        consumer_layers = (
            tuple(consumer_layers)
            if consumer_layers is not None
            else tuple(range(request_layer + 1, self.num_layers))
        )
        consumer_layer_set = set(consumer_layers)
        diagnostics: Dict[str, Any] = {}
        for layer_index, layer in enumerate(self.backbone.layers[request_layer + 1 :], start=request_layer + 1):
            layer_consumer = consumer if layer_index in consumer_layer_set else None
            # Newer Draft-KV consumers may own a separate parameter set per
            # receiver layer. Keep the consumer API unchanged while
            # offering an optional, deliberately narrow dispatch hook.
            if layer_consumer is not None:
                set_layer = getattr(layer_consumer, "set_active_layer", None)
                if set_layer is not None:
                    set_layer(layer_index)
            hidden_states, _, layer_diag = self._run_layer_with_consumer(
                layer,
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                cache=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                consumer=layer_consumer,
                consumer_kwargs=consumer_kwargs,
                **kwargs,
            )
            if layer_diag:
                diagnostics[f"layer_{layer_index}"] = layer_diag

        hidden_states = self.backbone.norm(hidden_states)
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return SplitBackOutput(
            last_hidden_state=hidden_states,
            logits=logits,
            cache=cache if use_cache else None,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        request_layer: int = 0,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
        consumer: Optional[nn.Module] = None,
        consumer_layers: Optional[Tuple[int, ...]] = None,
        consumer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SplitBackOutput:
        front = self.forward_front(
            input_ids=input_ids,
            attention_mask=attention_mask,
            request_layer=request_layer,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return self.forward_back(
            front,
            consumer=consumer,
            consumer_layers=consumer_layers,
            consumer_kwargs=consumer_kwargs,
            use_cache=use_cache,
            **kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> SplitBackOutput:
        """Callable wrapper for ordinary ``nn.Module`` callers.

        The wrapper intentionally is not an ``nn.Module`` itself: registering
        the frozen receiver twice would duplicate its state-dict namespace.
        """

        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def native_forward(self, *args: Any, **kwargs: Any):
        """Reference native receiver forward used by depth-split tests."""

        return self.receiver(*args, **kwargs)


class Qwen2DepthSplit(QwenDepthSplit):
    """Qwen2-only checked depth-split wrapper."""

    def __init__(self, receiver: nn.Module) -> None:
        super().__init__(receiver)
        if self.model_type != "qwen2":
            raise TypeError(
                f"Qwen2DepthSplit requires config.model_type 'qwen2', got {self.model_type!r}"
            )


class Qwen3DepthSplit(QwenDepthSplit):
    """Qwen3 depth-split wrapper."""

    def __init__(self, receiver: nn.Module) -> None:
        super().__init__(receiver)
        if self.model_type != "qwen3":
            raise TypeError(
                f"Qwen3DepthSplit requires config.model_type 'qwen3', got {self.model_type!r}"
            )


def make_qwen_depth_split(receiver: nn.Module) -> QwenDepthSplit:
    """Construct the receiver executor using explicit model-type dispatch."""

    model_type = str(getattr(getattr(receiver, "config", None), "model_type", ""))
    if model_type == "qwen2":
        return Qwen2DepthSplit(receiver)
    if model_type == "qwen3":
        return Qwen3DepthSplit(receiver)
    raise TypeError(
        "Draft-KV receiver must have config.model_type qwen2 or qwen3; "
        f"got {model_type!r}"
    )


__all__ = [
    "Qwen2DepthSplit",
    "Qwen3DepthSplit",
    "QwenDepthSplit",
    "SplitFrontOutput",
    "SplitBackOutput",
    "make_qwen_depth_split",
]
