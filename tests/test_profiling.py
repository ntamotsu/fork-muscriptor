import inspect

import pytest

import muscriptor.accelerator
from muscriptor.profiling import timed


def test_timed_accepts_the_profiled_device():
    parameter = inspect.signature(timed).parameters.get("device")

    assert parameter is not None
    assert parameter.default is None


def test_timed_disabled_does_not_synchronize_or_write(monkeypatch, capsys):
    synchronize_calls = []
    monkeypatch.setattr(
        muscriptor.accelerator,
        "synchronize",
        lambda device=None: synchronize_calls.append(device),
    )

    with timed(False, "decode", device="cuda:1"):
        pass
    captured = capsys.readouterr()

    assert (synchronize_calls, captured.out, captured.err) == ([], "", "")


def test_timed_forwards_device_and_writes_only_to_stderr(monkeypatch, capsys):
    synchronize_calls = []
    monkeypatch.setattr(
        muscriptor.accelerator,
        "synchronize",
        lambda device=None: synchronize_calls.append(device),
    )

    with timed(True, "decode", device="cuda:1", precision=3):
        pass
    captured = capsys.readouterr()

    assert synchronize_calls == ["cuda:1", "cuda:1"]
    assert captured.out == ""
    assert captured.err.startswith("[muscriptor] decode: ")
    assert captured.err.endswith("s\n")


def test_timed_does_not_mask_an_error_from_the_profiled_block(monkeypatch):
    calls = 0

    def synchronize(_device=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synchronization failed")

    monkeypatch.setattr(muscriptor.accelerator, "synchronize", synchronize)

    with pytest.raises(ValueError, match="profiled work failed"):
        with timed(True, "decode", device="cuda:0"):
            raise ValueError("profiled work failed")
