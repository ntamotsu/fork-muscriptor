import inspect

import pytest
import torch

import muscriptor.accelerator


def test_synchronize_accepts_an_explicit_device():
    parameter = inspect.signature(muscriptor.accelerator.synchronize).parameters.get(
        "device"
    )

    assert parameter is not None
    assert parameter.default is None


@pytest.mark.skipif(
    not hasattr(torch, "accelerator"), reason="torch.accelerator requires torch>=2.6"
)
def test_synchronize_cpu_does_not_touch_an_available_accelerator(monkeypatch):
    calls = []
    monkeypatch.setattr(muscriptor.accelerator, "_HAS_TORCH_ACCELERATOR", True)
    monkeypatch.setattr(
        muscriptor.accelerator.torch.accelerator,
        "current_device_index",
        lambda: calls.append("current") or 0,
    )
    monkeypatch.setattr(
        muscriptor.accelerator.torch.accelerator,
        "synchronize",
        lambda *_args: calls.append("synchronize"),
    )

    muscriptor.accelerator.synchronize("cpu")

    assert calls == []


@pytest.mark.skipif(
    not hasattr(torch, "accelerator"), reason="torch.accelerator requires torch>=2.6"
)
def test_synchronize_forwards_an_explicit_accelerator_device(monkeypatch):
    calls = []
    monkeypatch.setattr(muscriptor.accelerator, "_HAS_TORCH_ACCELERATOR", True)
    monkeypatch.setattr(
        muscriptor.accelerator.torch.accelerator,
        "synchronize",
        lambda device: calls.append(device),
    )

    muscriptor.accelerator.synchronize("cuda:1")

    assert calls == [muscriptor.accelerator.torch.device("cuda:1")]


@pytest.mark.skipif(
    not hasattr(torch, "accelerator"), reason="torch.accelerator requires torch>=2.6"
)
def test_synchronize_ignores_an_unavailable_accelerator_type(monkeypatch):
    monkeypatch.setattr(muscriptor.accelerator, "_HAS_TORCH_ACCELERATOR", True)

    def unavailable(_device):
        raise ValueError("accelerator type is unavailable")

    monkeypatch.setattr(
        muscriptor.accelerator.torch.accelerator, "synchronize", unavailable
    )

    try:
        muscriptor.accelerator.synchronize("cuda:1")
    except ValueError as error:
        pytest.fail(f"profiling synchronization should be best-effort: {error}")


def test_synchronize_forwards_cuda_device_on_legacy_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(muscriptor.accelerator, "_HAS_TORCH_ACCELERATOR", False)
    monkeypatch.setattr(muscriptor.accelerator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        muscriptor.accelerator.torch.cuda,
        "synchronize",
        lambda device: calls.append(device),
    )

    muscriptor.accelerator.synchronize("cuda:1")

    assert calls == [muscriptor.accelerator.torch.device("cuda:1")]


def test_synchronize_uses_mps_api_on_legacy_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(muscriptor.accelerator, "_HAS_TORCH_ACCELERATOR", False)
    monkeypatch.setattr(
        muscriptor.accelerator.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(
        muscriptor.accelerator.torch.mps,
        "synchronize",
        lambda: calls.append("mps"),
    )

    muscriptor.accelerator.synchronize("mps")

    assert calls == ["mps"]
