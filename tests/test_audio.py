"""音声resampler cacheのCPU限定テスト。"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from muscriptor.utils import audio as audio_utils
from muscriptor.utils.resample import ResampleFrac, resample_frac


@pytest.fixture(autouse=True)
def clear_resampler_cache():
    with audio_utils._RESAMPLER_CACHE_LOCK:
        audio_utils._RESAMPLER_CACHE.clear()
    yield
    with audio_utils._RESAMPLER_CACHE_LOCK:
        audio_utils._RESAMPLER_CACHE.clear()


def test_audio_module_keeps_resample_frac_reexport():
    assert audio_utils.resample_frac is resample_frac


def test_resample_reuses_one_prepared_resampler_for_the_same_key(monkeypatch):
    init_calls = 0
    to_calls = 0
    prepared_for = []
    original_init = ResampleFrac.__init__
    original_to = ResampleFrac.to

    def counted_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        original_init(self, *args, **kwargs)

    def counted_to(self, *args, **kwargs):
        nonlocal to_calls
        to_calls += 1
        prepared_for.append((kwargs["device"], kwargs["dtype"]))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(ResampleFrac, "__init__", counted_init)
    monkeypatch.setattr(ResampleFrac, "to", counted_to)
    waveform = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)

    first = audio_utils.resample(waveform, 31, 19)
    second = audio_utils.resample(waveform, 31, 19)

    assert (init_calls, to_calls, torch.equal(first, second)) == (1, 1, True)
    assert prepared_for == [(waveform.device, waveform.dtype)]


def test_resample_evicts_the_least_recently_used_entry_after_eight_keys(
    monkeypatch,
):
    init_calls = 0
    original_init = ResampleFrac.__init__

    def counted_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ResampleFrac, "__init__", counted_init)
    monkeypatch.setattr(ResampleFrac, "forward", lambda _self, waveform: waveform)
    waveform = torch.zeros(32, dtype=torch.float32)
    rate_pairs = [(40 + index, 17) for index in range(9)]

    for orig_freq, new_freq in rate_pairs[:8]:
        audio_utils.resample(waveform, orig_freq, new_freq)
    audio_utils.resample(waveform, *rate_pairs[0])
    audio_utils.resample(waveform, *rate_pairs[8])
    audio_utils.resample(waveform, *rate_pairs[1])
    audio_utils.resample(waveform, *rate_pairs[0])

    assert init_calls == 10


def test_resample_constructs_one_instance_for_concurrent_same_key(monkeypatch):
    worker_count = 4
    init_calls = 0
    to_calls = 0
    start = threading.Barrier(worker_count)
    release_constructor = threading.Event()
    constructor_entered = threading.Condition()
    original_init = ResampleFrac.__init__
    original_to = ResampleFrac.to

    def blocked_init(self, *args, **kwargs):
        nonlocal init_calls
        with constructor_entered:
            init_calls += 1
            constructor_entered.notify_all()
        release_constructor.wait(timeout=2)
        original_init(self, *args, **kwargs)

    def counted_to(self, *args, **kwargs):
        nonlocal to_calls
        to_calls += 1
        return original_to(self, *args, **kwargs)

    def run(waveform):
        start.wait(timeout=2)
        return audio_utils.resample(waveform, 73, 29)

    monkeypatch.setattr(ResampleFrac, "__init__", blocked_init)
    monkeypatch.setattr(ResampleFrac, "to", counted_to)
    monkeypatch.setattr(ResampleFrac, "forward", lambda _self, waveform: waveform)
    waveform = torch.zeros(32, dtype=torch.float32)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run, waveform) for _ in range(worker_count)]
        try:
            with constructor_entered:
                constructor_entered.wait_for(lambda: init_calls > 1, timeout=0.2)
        finally:
            release_constructor.set()
        results = [future.result(timeout=2) for future in futures]

    assert (init_calls, to_calls, all(result is waveform for result in results)) == (
        1,
        1,
        True,
    )


def test_resample_does_not_cache_a_failed_construction(monkeypatch):
    init_calls = 0
    original_init = ResampleFrac.__init__

    def fail_once(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        if init_calls == 1:
            raise RuntimeError("construction failed")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ResampleFrac, "__init__", fail_once)
    monkeypatch.setattr(ResampleFrac, "forward", lambda _self, waveform: waveform)
    waveform = torch.zeros(32, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="construction failed"):
        audio_utils.resample(waveform, 79, 31)
    second = audio_utils.resample(waveform, 79, 31)
    third = audio_utils.resample(waveform, 79, 31)

    assert (init_calls, second is waveform, third is waveform) == (2, True, True)


def test_resample_returns_the_same_tensor_without_touching_cache_for_same_rate(
    monkeypatch,
):
    def unexpected_init(*_args, **_kwargs):
        pytest.fail("同一サンプルレートではresamplerを構築しない")

    waveform = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
    monkeypatch.setattr(ResampleFrac, "forward", lambda _self, value: value)
    audio_utils.resample(waveform, 109, 59)
    cache_before = tuple(audio_utils._RESAMPLER_CACHE.items())
    monkeypatch.setattr(ResampleFrac, "__init__", unexpected_init)

    result = audio_utils.resample(waveform, 123, 123)

    assert result is waveform
    assert tuple(audio_utils._RESAMPLER_CACHE.items()) == cache_before


def test_resample_is_bitwise_equal_to_fresh_resample_frac_on_finite_cpu_float32():
    waveform = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(2, 128)

    expected = resample_frac(waveform, 97, 41)
    first = audio_utils.resample(waveform, 97, 41)
    second = audio_utils.resample(waveform, 97, 41)

    assert torch.isfinite(first).all()
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_resample_separates_cache_entries_by_input_dtype(monkeypatch):
    init_calls = 0
    to_calls = 0
    original_init = ResampleFrac.__init__
    original_to = ResampleFrac.to

    def counted_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        original_init(self, *args, **kwargs)

    def counted_to(self, *args, **kwargs):
        nonlocal to_calls
        to_calls += 1
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(ResampleFrac, "__init__", counted_init)
    monkeypatch.setattr(ResampleFrac, "to", counted_to)
    monkeypatch.setattr(ResampleFrac, "forward", lambda _self, waveform: waveform)
    float32_waveform = torch.zeros(32, dtype=torch.float32)
    float64_waveform = torch.zeros(32, dtype=torch.float64)

    results = [
        audio_utils.resample(float32_waveform, 103, 47),
        audio_utils.resample(float32_waveform, 103, 47),
        audio_utils.resample(float64_waveform, 103, 47),
        audio_utils.resample(float64_waveform, 103, 47),
    ]

    assert (init_calls, to_calls) == (2, 2)
    assert [result.dtype for result in results] == [
        torch.float32,
        torch.float32,
        torch.float64,
        torch.float64,
    ]


def test_resample_runs_cached_forward_outside_the_cache_lock(monkeypatch):
    workers_entered_forward = threading.Barrier(2)
    test_concurrency = threading.Event()

    def overlapping_forward(_self, waveform):
        if test_concurrency.is_set():
            workers_entered_forward.wait(timeout=2)
        return waveform

    monkeypatch.setattr(ResampleFrac, "forward", overlapping_forward)
    waveform = torch.zeros(32, dtype=torch.float32)
    audio_utils.resample(waveform, 107, 53)
    test_concurrency.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(audio_utils.resample, waveform, 107, 53) for _ in range(2)
        ]
        results = [future.result(timeout=2) for future in futures]

    assert all(result is waveform for result in results)
