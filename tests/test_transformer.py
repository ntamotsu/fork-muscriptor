"""Tests for muscriptor/modules/transformer.py — CPU only, small tensors."""

import einops
import torch

from muscriptor.modules import transformer as transformer_module
from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.modules.transformer import (
    create_sin_embedding,
    StreamingMultiheadAttention,
    StreamingTransformer,
)


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------


def test_create_sin_embedding_shape():
    positions = torch.arange(10).float().view(1, 10, 1)  # [B, T, 1]
    emb = create_sin_embedding(positions, dim=16)
    assert emb.shape == (1, 10, 16)


def test_create_sin_embedding_dim_even():
    positions = torch.arange(5).float().view(1, 5, 1)
    emb = create_sin_embedding(positions, dim=8)
    assert emb.shape == (1, 5, 8)


def test_create_sin_embedding_different_positions():
    pos1 = torch.tensor([[[0.0]]])  # [1, 1, 1]
    pos2 = torch.tensor([[[1.0]]])
    e1 = create_sin_embedding(pos1, dim=8)
    e2 = create_sin_embedding(pos2, dim=8)
    assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# Streaming attention state
# ---------------------------------------------------------------------------


class _CountingCache:
    def __init__(self, tensor):
        self.tensor = tensor
        self.writes = []

    def __getitem__(self, key):
        return self.tensor[key]

    def __setitem__(self, key, value):
        self.writes.append(key)
        self.tensor[key] = value


def _run_attention_with_counted_cache(token_count):
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    state = init_states(attention, batch_size=1, sequence_length=3)
    cache = _CountingCache(state[""]["cache"])
    state[""]["cache"] = cache

    with torch.no_grad():
        output = attention(torch.randn(1, token_count, 8), model_state=state)

    return output, cache


def test_attention_state_leaves_unread_cache_uninitialized(monkeypatch):
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2)
    expected_cache = torch.empty(2, 3, 7, 2, 4)
    empty_calls = []

    def fake_empty(*shape, **kwargs):
        empty_calls.append((shape, kwargs))
        return expected_cache

    monkeypatch.setattr(torch, "empty", fake_empty)

    state = attention.init_state(batch_size=3, sequence_length=7)

    assert state["cache"] is expected_cache
    assert state["offset"] == 0
    assert empty_calls == [
        (
            ((2, 3, 7, 2, 4),),
            {"device": attention.in_proj_weight.device, "dtype": torch.float32},
        )
    ]


def test_attention_never_reads_the_unwritten_cache_tail():
    torch.manual_seed(0)
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    prefill = torch.randn(1, 2, 8)
    next_token = torch.randn(1, 1, 8)
    states = [init_states(attention, batch_size=1, sequence_length=7) for _ in range(2)]
    states[0][""]["cache"].fill_(123)
    states[1][""]["cache"].fill_(-456)

    outputs = []
    with torch.no_grad():
        for state in states:
            prefill_output = attention(prefill, model_state=state)
            increment_steps(attention, state, increment=prefill.shape[1])
            decode_output = attention(next_token, model_state=state)
            outputs.append((prefill_output, decode_output))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert torch.equal(
        states[0][""]["cache"][:, :, :3], states[1][""]["cache"][:, :, :3]
    )
    assert torch.all(states[0][""]["cache"][:, :, 3:] == 123)
    assert torch.all(states[1][""]["cache"][:, :, 3:] == -456)


def test_attention_combines_the_single_token_kv_cache_write():
    output, cache = _run_attention_with_counted_cache(token_count=1)

    assert output.shape == (1, 1, 8)
    assert len(cache.writes) == 1


def test_attention_single_token_kv_write_preserves_axes_and_cache_regions():
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    packed = torch.arange(2 * 1 * 3 * 2 * 4).reshape(2, 1, 3, 2, 4)
    kv = packed[:, :, 1:].permute(2, 0, 1, 3, 4)
    cache_tensor = torch.full((2, 2, 5, 2, 4), -1)
    cache_tensor[:, :, :2] = 99
    cache = _CountingCache(cache_tensor)
    state = {"cache": cache, "offset": 2}

    k, v = attention._complete_kv(kv, state)

    assert len(cache.writes) == 1
    assert torch.equal(cache.tensor[:, :, 2:3], kv)
    assert torch.all(cache.tensor[:, :, :2] == 99)
    assert torch.all(cache.tensor[:, :, 3:] == -1)
    assert torch.equal(k, cache.tensor[0, :, :3])
    assert torch.equal(v, cache.tensor[1, :, :3])


def test_attention_keeps_prefill_kv_cache_writes_separate():
    output, cache = _run_attention_with_counted_cache(token_count=2)

    assert output.shape == (1, 2, 8)
    assert len(cache.writes) == 2


def test_attention_forward_avoids_einops_in_the_decode_loop(monkeypatch):
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    state = init_states(attention, batch_size=1, sequence_length=3)

    def fail_rearrange(*_args, **_kwargs):
        raise AssertionError("decode loopからeinopsを呼ばない")

    monkeypatch.setattr(einops, "rearrange", fail_rearrange)
    monkeypatch.setattr(transformer_module, "rearrange", fail_rearrange, raising=False)

    with torch.no_grad():
        output = attention(torch.randn(1, 1, 8), model_state=state)

    assert output.shape == (1, 1, 8)


# ---------------------------------------------------------------------------
# StreamingTransformer forward
# ---------------------------------------------------------------------------


def _make_transformer(**kwargs):
    defaults = dict(d_model=32, num_heads=2, num_layers=2, dim_feedforward=64)
    defaults.update(kwargs)
    return StreamingTransformer(**defaults)


def test_streaming_transformer_output_shape():
    model = _make_transformer()
    model.eval()
    x = torch.randn(2, 10, 32)  # [B, T, D]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10, 32)


def test_streaming_transformer_streaming_mode():
    """Feed tokens one at a time with explicit state; check output shape."""
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 6, 32)

    model_state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    streaming_outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            out_t = model(x[:, t : t + 1, :], model_state=model_state)
            streaming_outs.append(out_t)
            increment_steps(model, model_state, increment=1)
    streaming_out = torch.cat(streaming_outs, dim=1)

    assert streaming_out.shape == (1, 6, 32)


def test_streaming_transformer_fresh_state():
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 3, 32)
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        model(x, model_state=state)
    # Allocating a new state starts from scratch.
    state = init_states(model, batch_size=1, sequence_length=x.shape[1])
    with torch.no_grad():
        out = model(x, model_state=state)
    assert out.shape == (1, 3, 32)
