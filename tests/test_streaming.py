"""Stateful module traversal tests."""

from torch import nn

from muscriptor.modules.streaming import StatefulModule, increment_steps, init_states
from muscriptor.modules.transformer import StreamingTransformer


class _CounterStateful(StatefulModule):
    def init_state(self, batch_size: int, sequence_length: int) -> dict:
        return {"offset": 0}

    def increment_step(self, state: dict, increment: int = 1) -> None:
        state["offset"] += increment


def test_increment_steps_uses_full_state_names_for_a_subtree() -> None:
    wrapper = nn.Module()
    wrapper.transformer = StreamingTransformer(
        d_model=8,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
    )
    state = init_states(wrapper, batch_size=1, sequence_length=4)

    increment_steps(wrapper.transformer, state, increment=3)

    assert state["transformer"]["offsets"].tolist() == [3]
    assert state["transformer.layers.0.self_attn"]["offset"] == 3
    assert state["transformer.layers.1.self_attn"]["offset"] == 3


def test_init_states_returns_a_plain_dict() -> None:
    transformer = StreamingTransformer(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )

    state = init_states(transformer, batch_size=1, sequence_length=2)

    assert type(state) is dict


def test_increment_steps_advances_a_shared_module_once() -> None:
    wrapper = nn.Module()
    shared = _CounterStateful()
    wrapper.first = shared
    wrapper.second = shared
    state = init_states(wrapper, batch_size=1, sequence_length=2)

    increment_steps(wrapper, state, increment=4)

    assert state["first"]["offset"] == 4
