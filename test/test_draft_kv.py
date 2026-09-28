import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from draft_kv.model.draft_kv import (
    DraftKVProjector,
    DraftKVModel,
)
from draft_kv.model.draft_kv import LayerKV
from script.draft_kv.draft_kv_common import (
    answer_nll,
    load_trainable_state,
    trainable_state,
)


def _tiny_models():
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_cache=True,
        _attn_implementation="eager",
    )
    receiver = Qwen3ForCausalLM(Qwen3Config(**common)).eval()
    sharer_config = dict(common)
    sharer_config["num_key_value_heads"] = 1
    sharer = Qwen2ForCausalLM(Qwen2Config(**sharer_config)).eval()
    return receiver, sharer


def _tiny_reverse_models():
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_cache=True,
        _attn_implementation="eager",
    )
    receiver = Qwen2ForCausalLM(Qwen2Config(**common)).eval()
    sharer_config = dict(common)
    sharer_config["num_key_value_heads"] = 1
    sharer = Qwen3ForCausalLM(Qwen3Config(**sharer_config)).eval()
    return receiver, sharer


def test_draft_projector_preserves_all_token_positions():
    projector = DraftKVProjector(1, 4, 2, 3)
    source = LayerKV(
        key=torch.randn(2, 1, 5, 4),
        value=torch.randn(2, 1, 5, 4),
    )
    output = projector(source)
    assert output.key.shape == (2, 2, 5, 3)
    assert output.value.shape == (2, 2, 5, 3)
    assert projector.key.bias is None and projector.value.bias is None


def test_draft_kv_zero_gate_base_parity_and_task_gradients():
    torch.manual_seed(13)
    receiver, sharer = _tiny_models()
    model = DraftKVModel(receiver, sharer, {2: 1})
    model.set_stage("communication")
    sharer_ids = torch.tensor([[2, 3, 4, 20, 21], [5, 6, 7, 22, 23]])
    sharer_attention = torch.ones_like(sharer_ids)
    draft_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    packet = model.make_packet(sharer_ids, sharer_attention, draft_mask)

    receiver_ids = torch.tensor([[2, 3, 4, 8, 9], [5, 6, 7, 10, 11]])
    receiver_attention = torch.ones_like(receiver_ids)
    native = receiver(
        input_ids=receiver_ids,
        attention_mask=receiver_attention,
        use_cache=False,
    ).logits
    base, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=None,
        disable_communication=True,
    )
    zero_gate, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
    )
    assert torch.equal(native, base)
    assert torch.equal(base, zero_gate)

    with torch.no_grad():
        model.consumer.external["2"].gate_logits.fill_(0.1)
    communicated, _, diagnostics = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
        return_diagnostics=True,
    )
    assert not torch.equal(base, communicated)
    half_strength, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
        communication_scale=0.5,
    )
    assert not torch.equal(base, half_strength)
    assert not torch.equal(communicated, half_strength)
    zero_strength, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
        communication_scale=0.0,
    )
    assert torch.equal(base, zero_strength)
    attention = diagnostics["layer_2"]["draft_kv_attention"]
    assert torch.equal(attention[..., :3], torch.zeros_like(attention[..., :3]))
    layer_diagnostics = diagnostics["layer_2"]
    assert layer_diagnostics["draft_kv_update_token_rms"].shape == receiver_ids.shape
    assert layer_diagnostics["draft_kv_input_token_rms"].shape == receiver_ids.shape
    assert layer_diagnostics["draft_kv_relative_update_token_rms"].shape == receiver_ids.shape
    assert torch.isfinite(layer_diagnostics["draft_kv_relative_update_token_rms"]).all()

    labels = receiver_ids.clone()
    labels[:, :3] = -100
    answer_nll(communicated, labels)["mean"].mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.projection.parameters()
    )
    assert model.consumer.external["2"].gate_logits.grad is not None
    assert not any(parameter.grad is not None for parameter in receiver.parameters())
    assert not any(parameter.grad is not None for parameter in sharer.parameters())

    receiver_copy, sharer_copy = _tiny_models()
    reloaded = DraftKVModel(receiver_copy, sharer_copy, {2: 1})
    load_trainable_state(reloaded, trainable_state(model))
    for expected, observed in zip(
        model.projection.parameters(), reloaded.projection.parameters()
    ):
        assert torch.equal(expected, observed)
    assert torch.equal(
        model.consumer.external["2"].gate_logits,
        reloaded.consumer.external["2"].gate_logits,
    )


def _greedy_tokens(model, input_ids, attention_mask, *, steps):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    cache = output.past_key_values
    generated = []
    next_token = output.logits[:, -1].argmax(dim=-1)
    for index in range(steps):
        generated.append(next_token)
        if index + 1 == steps:
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
        )
        output = model(
            input_ids=next_token[:, None],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def _split_greedy_tokens(model, input_ids, attention_mask, *, steps):
    logits, cache, _ = model.forward_receiver(
        input_ids,
        attention_mask,
        packet=None,
        disable_communication=True,
        use_cache=True,
    )
    generated = []
    next_token = logits[:, -1].argmax(dim=-1)
    for index in range(steps):
        generated.append(next_token)
        if index + 1 == steps:
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
        )
        logits, cache, _ = model.forward_receiver(
            next_token[:, None],
            attention_mask,
            packet=None,
            disable_communication=True,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = logits[:, -1].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def test_reverse_draft_kv_qwen2_receiver_depth_split_and_gradients():
    torch.manual_seed(29)
    receiver, sharer = _tiny_reverse_models()
    mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    model = DraftKVModel(receiver, sharer, mapping)
    assert model.receiver_split.model_type == "qwen2"
    model.set_stage("communication")

    sharer_ids = torch.tensor([[2, 3, 4, 20, 21], [5, 6, 7, 22, 23]])
    sharer_attention = torch.ones_like(sharer_ids)
    draft_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    packet = model.make_packet(sharer_ids, sharer_attention, draft_mask)
    receiver_ids = torch.tensor([[2, 3, 4, 8, 9], [5, 6, 7, 10, 11]])
    receiver_attention = torch.ones_like(receiver_ids)

    native = receiver(
        input_ids=receiver_ids,
        attention_mask=receiver_attention,
        use_cache=False,
    ).logits
    base, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=None,
        disable_communication=True,
        use_cache=False,
    )
    assert float((native - base).abs().max()) <= 1e-5

    zero_gate, _, _ = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
        use_cache=False,
    )
    assert torch.equal(base, zero_gate)
    assert sum(module.gate_logits.numel() for module in model.consumer.external.values()) == 8

    with torch.no_grad():
        for module in model.consumer.external.values():
            module.gate_logits.fill_(0.1)
    communicated, _, diagnostics = model.forward_receiver(
        receiver_ids,
        receiver_attention,
        packet=packet,
        disable_communication=False,
        return_diagnostics=True,
        use_cache=False,
    )
    assert not torch.equal(base, communicated)
    for layer_index in mapping:
        attention = diagnostics[f"layer_{layer_index}"]["draft_kv_attention"]
        assert torch.equal(attention[..., :3], torch.zeros_like(attention[..., :3]))

    labels = receiver_ids.clone()
    labels[:, :3] = -100
    answer_nll(communicated, labels)["mean"].mean().backward()
    projector_gradients = [
        parameter.grad for parameter in model.projection.parameters()
    ]
    gate_gradients = [
        module.gate_logits.grad for module in model.consumer.external.values()
    ]
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0
        for gradient in projector_gradients
    )
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0
        for gradient in gate_gradients
    )
    assert not any(parameter.grad is not None for parameter in receiver.parameters())
    assert not any(parameter.grad is not None for parameter in sharer.parameters())

    receiver_copy, sharer_copy = _tiny_reverse_models()
    reloaded = DraftKVModel(receiver_copy, sharer_copy, mapping)
    load_trainable_state(reloaded, trainable_state(model))
    expected = trainable_state(model)
    observed = trainable_state(reloaded)
    for name, tensor in expected["projection"].items():
        assert torch.equal(tensor, observed["projection"][name])
    for layer, tensor in expected["gate_logits"].items():
        assert torch.equal(tensor, observed["gate_logits"][layer])


def test_reverse_qwen2_cache_incremental_greedy_matches_native():
    torch.manual_seed(31)
    receiver, sharer = _tiny_reverse_models()
    model = DraftKVModel(receiver, sharer, {2: 2})
    input_ids = torch.tensor([[2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    native = _greedy_tokens(
        receiver, input_ids, attention_mask.clone(), steps=5
    )
    split = _split_greedy_tokens(
        model, input_ids, attention_mask.clone(), steps=5
    )
    assert torch.equal(native, split)
