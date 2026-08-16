from types import SimpleNamespace
import types

import pytest
import torch
from torch.nn.modules.module import register_module_forward_hook

import muscriptor.models.lm as lm_module
import muscriptor.transcription_model as transcription_module
from muscriptor.models.lm import LMModel
from muscriptor.models.speculative import HistoryNgramDraft
from muscriptor.modules.conditioners import ConditioningProvider
from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.modules.transformer import (
    StreamingMultiheadAttention,
    StreamingTransformer,
)
from muscriptor.transcription_model import TranscriptionModel


def test_history_draft_uses_only_completed_chunks_and_observed_prefix():
    draft = HistoryNgramDraft(block_size=4, min_context=2, max_context=16)

    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12, 13, 14, 15]:
        draft.observe(token)
    draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    assert draft.propose() is None

    draft.observe(10)
    assert draft.propose() == (11, 12, 13, 14)


def test_history_draft_prefers_longest_then_most_recent_match():
    draft = HistoryNgramDraft(block_size=2, min_context=2, max_context=8)

    draft.start_chunk(initial_token=99)
    for token in [1, 2, 7, 8, 99, 1, 2, 9, 10]:
        draft.observe(token)
    draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(1)
    draft.observe(2)

    assert draft.propose() == (9, 10)


def test_history_draft_does_not_cross_chunk_end():
    draft = HistoryNgramDraft(block_size=2, min_context=2, max_context=8)

    draft.start_chunk(initial_token=99)
    draft.observe(10)
    draft.observe(1)  # EOS
    draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() is None


def test_history_draft_discards_an_unfinished_chunk():
    draft = HistoryNgramDraft(block_size=2, min_context=2, max_context=8)

    draft.start_chunk(initial_token=99)
    draft.observe(10)
    draft.discard_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() is None


def test_history_draft_drops_matches_older_than_eight_completed_chunks():
    draft = HistoryNgramDraft(
        block_size=2,
        min_context=2,
        max_context=2,
    )

    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12]:
        draft.observe(token)
    draft.finish_chunk()
    for initial_token in range(200, 208):
        draft.start_chunk(initial_token=initial_token)
        for token in [20, 21, 22]:
            draft.observe(token)
        draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() is None


def test_history_draft_keeps_the_eighth_most_recent_completed_chunk():
    draft = HistoryNgramDraft(
        block_size=2,
        min_context=2,
        max_context=2,
    )

    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12]:
        draft.observe(token)
    draft.finish_chunk()
    for initial_token in range(200, 207):
        draft.start_chunk(initial_token=initial_token)
        for token in [20, 21, 22]:
            draft.observe(token)
        draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() == (11, 12)


def test_history_draft_still_uses_the_active_chunk_when_history_is_full():
    draft = HistoryNgramDraft(
        block_size=2,
        min_context=2,
        max_context=2,
        max_completed_chunks=1,
    )
    draft.start_chunk(initial_token=200)
    for token in [20, 21, 22]:
        draft.observe(token)
    draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    for token in [10, 11, 99, 10]:
        draft.observe(token)

    assert draft.propose() == (11, 99)


def test_discarded_chunk_does_not_consume_completed_history_capacity():
    draft = HistoryNgramDraft(
        block_size=2,
        min_context=2,
        max_context=2,
        max_completed_chunks=1,
    )
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12]:
        draft.observe(token)
    draft.finish_chunk()

    draft.start_chunk(initial_token=200)
    draft.observe(20)
    draft.discard_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() == (11, 12)


def test_history_draft_rejects_non_positive_completed_chunk_limit():
    with pytest.raises(ValueError, match="max_completed_chunks"):
        HistoryNgramDraft(max_completed_chunks=0)


def test_history_draft_stops_after_eight_unproductive_verifications():
    draft = HistoryNgramDraft()
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12, 13, 14]:
        draft.observe(token)
    draft.finish_chunk()
    draft.start_chunk(initial_token=99)
    draft.observe(10)

    for _ in range(7):
        draft.record_block(proposed=4, committed=1)
    assert draft.propose() == (11, 12, 13, 14)

    draft.record_block(proposed=4, committed=4)

    assert draft.propose() is None


def test_history_draft_keeps_drafting_at_twelve_committed_tokens():
    draft = HistoryNgramDraft()
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12, 13, 14]:
        draft.observe(token)
    draft.finish_chunk()
    draft.start_chunk(initial_token=99)
    draft.observe(10)

    for committed in [2, 1, 1, 1, 1, 1, 1, 4]:
        draft.record_block(proposed=4, committed=committed)

    assert draft.propose() == (11, 12, 13, 14)


def test_history_draft_stops_when_the_rolling_window_falls_below_twelve():
    draft = HistoryNgramDraft()
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12, 13, 14]:
        draft.observe(token)
    draft.finish_chunk()
    draft.start_chunk(initial_token=99)
    draft.observe(10)
    for committed in [2, 1, 1, 1, 1, 1, 1, 4]:
        draft.record_block(proposed=4, committed=committed)
    assert draft.propose() == (11, 12, 13, 14)

    draft.record_block(proposed=4, committed=1)

    assert draft.propose() is None


def test_history_draft_reenables_drafting_for_the_next_chunk():
    draft = HistoryNgramDraft()
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12, 13, 14]:
        draft.observe(token)
    draft.finish_chunk()
    draft.start_chunk(initial_token=99)
    draft.observe(10)
    for _ in range(8):
        draft.record_block(proposed=4, committed=1)
    assert draft.propose() is None

    draft.finish_chunk()
    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() == (11, 12, 13, 14)


def _record_track_evidence(draft, committed_tokens):
    committed = iter(committed_tokens)
    for _ in range(4):
        draft.start_chunk(initial_token=99)
        draft.observe(10)
        for _ in range(8):
            for _ in range(8):
                draft.propose()
            draft.record_block(proposed=4, committed=next(committed))
        draft.finish_chunk()


def test_history_draft_disables_the_track_when_cumulative_savings_are_low():
    draft = HistoryNgramDraft()

    _record_track_evidence(draft, [2] * 31 + [1])

    assert draft.enabled is False


def test_history_draft_keeps_the_track_at_the_cumulative_savings_boundary():
    draft = HistoryNgramDraft()

    _record_track_evidence(draft, [2] * 32)

    assert draft.enabled is True


def test_short_completed_chunk_does_not_consume_history_capacity():
    draft = HistoryNgramDraft(
        block_size=2,
        min_context=2,
        max_context=2,
        max_completed_chunks=1,
    )
    draft.start_chunk(initial_token=99)
    for token in [10, 11, 12]:
        draft.observe(token)
    draft.finish_chunk()

    draft.start_chunk(initial_token=200)
    draft.observe(20)
    draft.finish_chunk()

    draft.start_chunk(initial_token=99)
    draft.observe(10)

    assert draft.propose() == (11, 12)


@pytest.mark.parametrize(
    ("block_size", "min_context", "max_context"),
    [(0, 2, 8), (2, 0, 8), (2, 3, 2)],
)
def test_history_draft_rejects_invalid_configuration(
    block_size, min_context, max_context
):
    with pytest.raises(ValueError):
        HistoryNgramDraft(
            block_size=block_size,
            min_context=min_context,
            max_context=max_context,
        )


def test_rectangular_attention_matches_scalar_decode():
    torch.manual_seed(11)
    attention = StreamingMultiheadAttention(embed_dim=8, num_heads=2).eval()
    prefix = torch.randn(1, 3, 8)
    block = torch.randn(1, 2, 8)

    scalar_state = init_states(attention, batch_size=1, sequence_length=8)
    block_state = init_states(attention, batch_size=1, sequence_length=8)
    attention(prefix, model_state=scalar_state)
    attention(prefix, model_state=block_state)
    increment_steps(attention, scalar_state, 3)
    increment_steps(attention, block_state, 3)

    scalar_outputs = []
    for index in range(block.shape[1]):
        scalar_outputs.append(
            attention(block[:, index : index + 1], model_state=scalar_state)
        )
        increment_steps(attention, scalar_state)

    mask = torch.ones(2, 5, dtype=torch.bool).tril(diagonal=3)
    block_output = attention(
        block,
        model_state=block_state,
        _attention_mask=mask,
    )
    increment_steps(attention, block_state, 2)

    torch.testing.assert_close(block_output, torch.cat(scalar_outputs, dim=1))
    assert scalar_state[""]["offset"] == block_state[""]["offset"] == 5
    torch.testing.assert_close(
        scalar_state[""]["cache"][:, :, :5],
        block_state[""]["cache"][:, :, :5],
    )


def test_transformer_shares_one_rectangular_mask_across_layers(monkeypatch):
    torch.manual_seed(12)
    transformer = StreamingTransformer(
        d_model=8,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
    ).eval()
    state = init_states(transformer, batch_size=1, sequence_length=8)
    transformer(torch.randn(1, 3, 8), model_state=state)
    increment_steps(transformer, state, 3)

    observed_masks = []
    for layer in transformer.layers:
        original = layer.self_attn.forward

        def record_mask(
            query,
            model_state=None,
            *,
            _attention_mask=None,
            _original=original,
        ):
            observed_masks.append(_attention_mask)
            if _attention_mask is None:
                return _original(query, model_state=model_state)
            return _original(
                query,
                model_state=model_state,
                _attention_mask=_attention_mask,
            )

        monkeypatch.setattr(layer.self_attn, "forward", record_mask)

    transformer(torch.randn(1, 2, 8), model_state=state)

    assert len(observed_masks) == 2
    assert observed_masks[0] is observed_masks[1]
    assert observed_masks[0].shape == (2, 5)


def _tiny_lm() -> LMModel:
    torch.manual_seed(21)
    device = torch.device("cpu")
    model = LMModel(
        condition_provider=ConditioningProvider(conditioners={}, device=device),
        card=16,
        dim=16,
        num_heads=2,
        hidden_scale=2,
        num_layers=1,
        max_period=10_000,
        device=device,
    )
    return model.eval()


def _token_ids(steps) -> list[int]:
    return [int(step[0]) for step in steps]


class _TraceDraft:
    def __init__(self, expected: list[int], mismatch_index: int | None = None):
        self.expected = expected
        self.mismatch_index = mismatch_index
        self.observed: list[int] = []
        self.calls = 0

    def propose(self) -> tuple[int, ...] | None:
        start = len(self.observed)
        if start + 4 > len(self.expected):
            return None
        proposal = self.expected[start : start + 4].copy()
        if self.mismatch_index is not None:
            index = self.mismatch_index
            proposal[index] = (proposal[index] + 1) % 16
        self.calls += 1
        return tuple(proposal)

    def observe(self, token: int) -> None:
        self.observed.append(token)


def _speculative_tokens(
    model: LMModel,
    expected: list[int],
    *,
    mismatch_index: int | None = None,
    stop_token: int | None = None,
) -> tuple[list[int], _TraceDraft]:
    draft = _TraceDraft(expected, mismatch_index=mismatch_index)
    tokens = []
    for step in model.generate(
        max_gen_len=len(expected),
        num_samples=1,
        use_sampling=False,
        _draft_provider=draft.propose,
        _speculative_stop_token=stop_token,
    ):
        token = int(step[0])
        tokens.append(token)
        draft.observe(token)
    return tokens, draft


@pytest.mark.parametrize("mismatch_index", [None, 0, 1, 3])
def test_speculative_greedy_matches_scalar_after_accept_or_rollback(mismatch_index):
    model = _tiny_lm()
    expected = _token_ids(
        model.generate(max_gen_len=12, num_samples=1, use_sampling=False)
    )

    actual, draft = _speculative_tokens(
        model,
        expected,
        mismatch_index=mismatch_index,
    )

    assert actual == expected
    assert draft.calls > 0


@pytest.mark.parametrize(
    ("mismatch_index", "committed"),
    [(None, 4), (0, 1), (1, 2), (3, 4)],
)
def test_speculative_greedy_reports_only_verified_and_committed_tokens(
    mismatch_index,
    committed,
):
    model = _tiny_lm()
    expected = _token_ids(
        model.generate(max_gen_len=5, num_samples=1, use_sampling=False)
    )
    draft = _TraceDraft(expected, mismatch_index=mismatch_index)
    feedback = []

    actual = []
    for step in model.generate(
        max_gen_len=5,
        num_samples=1,
        use_sampling=False,
        _draft_provider=draft.propose,
        _draft_feedback=lambda proposed, accepted: feedback.append(
            (proposed, accepted)
        ),
    ):
        token = int(step[0])
        actual.append(token)
        draft.observe(token)

    assert actual == expected
    assert feedback == [(4, committed)]


def test_speculative_greedy_stops_at_eos_inside_an_accepted_block():
    model = _tiny_lm()
    expected = _token_ids(
        model.generate(max_gen_len=16, num_samples=1, use_sampling=False)
    )
    stop_index = next(
        index
        for index in range(1, len(expected))
        if expected[index] not in expected[:index]
    )
    stop_token = expected[stop_index]

    actual, _ = _speculative_tokens(model, expected, stop_token=stop_token)

    assert actual == expected[: stop_index + 1]


def test_speculative_feedback_uses_the_eos_clamped_commit_count():
    model = _tiny_lm()
    expected = _token_ids(
        model.generate(max_gen_len=5, num_samples=1, use_sampling=False)
    )
    stop_token = expected[1]
    draft = _TraceDraft(expected)
    feedback = []
    actual = []

    for step in model.generate(
        max_gen_len=5,
        num_samples=1,
        use_sampling=False,
        _draft_provider=draft.propose,
        _speculative_stop_token=stop_token,
        _draft_feedback=lambda proposed, committed: feedback.append(
            (proposed, committed)
        ),
    ):
        token = int(step[0])
        actual.append(token)
        draft.observe(token)

    assert actual == expected[:2]
    assert feedback == [(4, 1)]


def test_speculative_greedy_honours_the_existing_early_stop_token():
    model = _tiny_lm()
    full = _token_ids(model.generate(max_gen_len=16, num_samples=1, use_sampling=False))
    stop_index = next(
        index for index in range(1, len(full)) if full[index] not in full[:index]
    )
    stop_token = full[stop_index]
    expected = _token_ids(
        model.generate(
            max_gen_len=16,
            num_samples=1,
            use_sampling=False,
            early_stop_on_token=stop_token,
        )
    )
    draft = _TraceDraft(full)
    actual = []
    for step in model.generate(
        max_gen_len=16,
        num_samples=1,
        use_sampling=False,
        early_stop_on_token=stop_token,
        _draft_provider=draft.propose,
    ):
        token = int(step[0])
        actual.append(token)
        draft.observe(token)

    assert actual == expected


def test_speculative_greedy_falls_back_for_an_overridden_sampling_seam():
    model = _tiny_lm()
    calls = 0

    def fixed_token(self, *args, **kwargs):
        del args, kwargs
        return torch.tensor([3], device=self.emb.weight.device)

    def propose():
        nonlocal calls
        calls += 1
        return (4, 4, 4, 4)

    model._sample_next_token = types.MethodType(fixed_token, model)
    tokens = _token_ids(
        model.generate(
            max_gen_len=8,
            num_samples=1,
            use_sampling=False,
            _draft_provider=propose,
        )
    )

    assert tokens == [3] * 8
    assert calls == 0


def test_speculative_greedy_detects_a_class_level_sampling_override(monkeypatch):
    model = _tiny_lm()
    calls = 0

    def fixed_token(self, *args, **kwargs):
        del args, kwargs
        return torch.tensor([3], device=self.emb.weight.device)

    def propose():
        nonlocal calls
        calls += 1
        return (4, 4, 4, 4)

    monkeypatch.setattr(LMModel, "_sample_next_token", fixed_token)
    tokens = _token_ids(
        model.generate(
            max_gen_len=8,
            num_samples=1,
            use_sampling=False,
            _draft_provider=propose,
        )
    )

    assert tokens == [3] * 8
    assert calls == 0


def test_speculative_guard_detects_an_overridden_block_verifier():
    model = _tiny_lm()
    original = model._compute_speculative_logits

    def wrapped(self, *args, **kwargs):
        return original(*args, **kwargs)

    model._compute_speculative_logits = types.MethodType(wrapped, model)

    assert model._can_use_speculative_greedy() is False


def test_speculative_guard_method_cannot_be_monkeypatched_to_bypass_checks():
    model = _tiny_lm()
    model._can_use_speculative_greedy = lambda: True

    assert lm_module._supports_speculative_greedy(model) is False


def test_speculative_guard_detects_an_overridden_generate_method():
    model = _tiny_lm()
    original = model.generate

    def wrapped(self, *args, **kwargs):
        return original(*args, **kwargs)

    model.generate = types.MethodType(wrapped, model)

    assert lm_module._supports_speculative_greedy(model) is False


def test_speculative_greedy_falls_back_for_a_patched_attention(monkeypatch):
    model = _tiny_lm()
    calls = 0
    attention = model.transformer.layers[0].self_attn
    original = attention.forward

    def wrapped(self, *args, **kwargs):
        return original(*args, **kwargs)

    def propose():
        nonlocal calls
        calls += 1
        return (4, 4, 4, 4)

    monkeypatch.setattr(attention, "forward", types.MethodType(wrapped, attention))
    _token_ids(
        model.generate(
            max_gen_len=8,
            num_samples=1,
            use_sampling=False,
            _draft_provider=propose,
        )
    )

    assert calls == 0


def test_speculative_greedy_falls_back_while_a_global_hook_is_registered():
    model = _tiny_lm()
    calls = 0

    def propose():
        nonlocal calls
        calls += 1
        return (4, 4, 4, 4)

    handle = register_module_forward_hook(lambda _module, _args, _output: None)
    try:
        _token_ids(
            model.generate(
                max_gen_len=8,
                num_samples=1,
                use_sampling=False,
                _draft_provider=propose,
            )
        )
    finally:
        handle.remove()

    assert calls == 0


def test_speculative_greedy_preserves_prompt_and_generation_limit():
    model = _tiny_lm()
    prompt = torch.tensor([[5, 3, 9]])
    expected = _token_ids(
        model.generate(
            prompt=prompt,
            max_gen_len=11,
            num_samples=1,
            use_sampling=False,
        )
    )
    draft = _TraceDraft(expected)
    actual = []
    for step in model.generate(
        prompt=prompt,
        max_gen_len=11,
        num_samples=1,
        use_sampling=False,
        _draft_provider=draft.propose,
    ):
        token = int(step[0])
        actual.append(token)
        draft.observe(token)

    assert actual == expected
    assert actual[:3] == [5, 3, 9]
    assert len(actual) == 11


def test_speculative_greedy_applies_forbidden_mask_to_every_block_row():
    model = _tiny_lm()
    allowed = 3
    forbidden = [token for token in range(16) if token != allowed]
    expected = _token_ids(
        model.generate(
            max_gen_len=10,
            num_samples=1,
            use_sampling=False,
            forbidden_tokens=forbidden,
        )
    )
    draft = _TraceDraft(expected)
    actual = []
    for step in model.generate(
        max_gen_len=10,
        num_samples=1,
        use_sampling=False,
        forbidden_tokens=forbidden,
        _draft_provider=draft.propose,
    ):
        token = int(step[0])
        actual.append(token)
        draft.observe(token)

    assert actual == expected == [allowed] * 10


def test_fixed_workload_guard_rejects_unverified_devices():
    fake = SimpleNamespace(
        _device=torch.device("cpu"),
        _model=_tiny_lm(),
    )

    with pytest.raises(ValueError, match="MPS"):
        TranscriptionModel._validate_speculative_ngram(fake, profile=False)


def test_fixed_workload_guard_checks_the_loaded_weight_device(monkeypatch):
    model = transcription_module._build_model(
        torch.device("meta"), transcription_module._CONFIGS["large"]
    ).half()
    model.eval()
    monkeypatch.setattr(
        "muscriptor.transcription_model._supports_speculative_greedy",
        lambda _model: True,
    )
    fake = SimpleNamespace(
        _device=torch.device("mps"),
        _model=model,
    )

    with pytest.raises(ValueError, match="MPS"):
        TranscriptionModel._validate_speculative_ngram(fake, profile=False)


def test_verified_speculative_architecture_rejects_a_non_large_head_layout():
    model = transcription_module._build_model(
        torch.device("meta"), transcription_module._CONFIGS["large"]
    )

    assert TranscriptionModel._has_verified_speculative_architecture(model)

    model.transformer.layers[0].self_attn.num_heads = 16

    assert not TranscriptionModel._has_verified_speculative_architecture(model)


def test_token_stream_drafts_from_a_completed_chunk_history():
    scripts = [[10, 11, 12, 13, 14, 1], [10, 11, 1]]
    proposals: list[tuple[int, ...] | None] = []
    calls = 0

    def generate(
        prompt=None,
        *,
        _draft_provider=None,
        _speculative_stop_token=None,
        **_kwargs,
    ):
        nonlocal calls
        assert prompt is None
        assert _draft_provider is not None
        assert _speculative_stop_token == 1
        script = scripts[calls]
        calls += 1
        for token in script:
            yield torch.tensor([token])
            proposals.append(_draft_provider())

    fake = SimpleNamespace(
        _model=SimpleNamespace(initial_token_id=99, generate=generate),
        _tokenizer=SimpleNamespace(eos_id=1),
        _device=torch.device("cpu"),
    )

    list(
        TranscriptionModel._generate_token_stream(
            fake,
            [object(), object()],
            [0.0, 5.0],
            batch_size=1,
            max_gen_len=16,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=False,
            _speculative_ngram=True,
        )
    )

    # 2曲目の最初のtokenを観測した時点で、1曲目の続きがdraftになる。
    assert (11, 12, 13, 14) in proposals


def test_token_stream_passes_verified_block_feedback_to_the_history(monkeypatch):
    feedback = []

    class RecordingHistory:
        enabled = True

        def start_chunk(self, initial_token):
            del initial_token

        def propose(self):
            return None

        def record_block(self, proposed, committed):
            feedback.append((proposed, committed))

        def observe(self, token):
            del token

        def finish_chunk(self):
            pass

        def discard_chunk(self):
            pass

    monkeypatch.setattr(transcription_module, "HistoryNgramDraft", RecordingHistory)

    def generate(*, _draft_feedback=None, **_kwargs):
        assert _draft_feedback is not None
        _draft_feedback(4, 2)
        yield torch.tensor([1])

    fake = SimpleNamespace(
        _model=SimpleNamespace(initial_token_id=99, generate=generate),
        _tokenizer=SimpleNamespace(eos_id=1),
        _device=torch.device("cpu"),
    )

    list(
        TranscriptionModel._generate_token_stream(
            fake,
            [object()],
            [0.0],
            batch_size=1,
            max_gen_len=16,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=False,
            _speculative_ngram=True,
        )
    )

    assert feedback == [(4, 2)]


def test_token_stream_does_not_keep_a_chunk_that_never_emits_eos(monkeypatch):
    lifecycle = []

    class RecordingHistory:
        enabled = True

        def start_chunk(self, initial_token):
            del initial_token

        def propose(self):
            return None

        def record_block(self, proposed, committed):
            del proposed, committed

        def observe(self, token):
            del token

        def finish_chunk(self):
            lifecycle.append("finish")

        def discard_chunk(self):
            lifecycle.append("discard")

    monkeypatch.setattr(transcription_module, "HistoryNgramDraft", RecordingHistory)

    def generate(**_kwargs):
        yield torch.tensor([7])

    fake = SimpleNamespace(
        _model=SimpleNamespace(initial_token_id=99, generate=generate),
        _tokenizer=SimpleNamespace(eos_id=1),
        _device=torch.device("cpu"),
    )

    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        list(
            TranscriptionModel._generate_token_stream(
                fake,
                [object()],
                [0.0],
                batch_size=1,
                max_gen_len=1,
                use_sampling=False,
                temperature=1.0,
                cfg_coef=1.0,
                no_eos_is_ok=True,
                prelude_forcing=False,
                _speculative_ngram=True,
            )
        )

    assert lifecycle == ["discard"]


def test_token_stream_omits_speculative_kwargs_after_track_disable(monkeypatch):
    generate_kwargs = []

    class DisablingHistory:
        def __init__(self):
            self.enabled = True

        def start_chunk(self, initial_token):
            del initial_token

        def propose(self):
            return None

        def record_block(self, proposed, committed):
            del proposed, committed

        def observe(self, token):
            del token

        def finish_chunk(self):
            self.enabled = False

        def discard_chunk(self):
            pass

    monkeypatch.setattr(transcription_module, "HistoryNgramDraft", DisablingHistory)

    def generate(**kwargs):
        generate_kwargs.append(kwargs)
        yield torch.tensor([1])

    fake = SimpleNamespace(
        _model=SimpleNamespace(initial_token_id=99, generate=generate),
        _tokenizer=SimpleNamespace(eos_id=1),
        _device=torch.device("cpu"),
    )

    list(
        TranscriptionModel._generate_token_stream(
            fake,
            [object(), object()],
            [0.0, 5.0],
            batch_size=1,
            max_gen_len=16,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=False,
            _speculative_ngram=True,
        )
    )

    private_keys = {
        "_draft_provider",
        "_draft_feedback",
        "_speculative_stop_token",
    }
    assert private_keys <= generate_kwargs[0].keys()
    assert private_keys.isdisjoint(generate_kwargs[1])


def test_token_stream_default_does_not_pass_private_speculative_kwargs():
    def legacy_generate(
        prompt=None,
        conditions=None,
        max_gen_len=16,
        use_sampling=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        cfg_coef=1.0,
        early_stop_on_token=None,
        beam_size=1,
        forbidden_tokens=None,
        profile=False,
    ):
        del (
            prompt,
            conditions,
            max_gen_len,
            use_sampling,
            temp,
            top_k,
            top_p,
            cfg_coef,
            early_stop_on_token,
            beam_size,
            forbidden_tokens,
            profile,
        )
        yield torch.tensor([1])

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=legacy_generate),
        _tokenizer=SimpleNamespace(eos_id=1),
        _device=torch.device("cpu"),
    )

    list(
        TranscriptionModel._generate_token_stream(
            fake,
            [object()],
            [0.0],
            batch_size=1,
            max_gen_len=16,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=False,
        )
    )
