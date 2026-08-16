"""Tests for muscriptor/utils/beats.py.

All synthetic: the maths is exercised without beat_this or a checkpoint, since
only detect_grid touches the model.
"""

import builtins
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from muscriptor.utils.beats import (
    BAR_OFFSET_MARKER,
    MAX_ONSET_DELAY_S,
    MAX_TEMPO_RESIDUAL,
    MIN_ONSETS,
    BeatDetectionError,
    BeatGrid,
    detect_grid,
    estimate_onset_delay,
    fit_tempo,
    get_onsets_phase,
    infer_beats_per_bar,
    read_bar_offset,
)


def _beats(bpm=120.0, n=64, start=0.0, drift=0.0):
    """Beat times at `bpm`, optionally with a linear tempo ramp of `drift`."""
    t = start + np.arange(n) * (60.0 / bpm)
    if drift:
        span = t[-1] - t[0]
        t = t[0] + (t - t[0]) * (1 + drift * (t - t[0]) / span)
    return t


def _grid(beats, bpm, beats_per_bar=4):
    """A BeatGrid over `beats`, as detect_grid would build it."""
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        first_downbeat=float(beats[0]),
        beats=beats,
    )


def _onsets(beats, subdivision=4, delay=0.0):
    """Onsets on every 1/subdivision of `beats`, `delay` seconds late."""
    fine = np.linspace(beats[0], beats[-1], (len(beats) - 1) * subdivision + 1)
    return fine + delay


def test_short_audio_is_rejected_without_importing_beat_this(monkeypatch):
    imported = []
    real_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name == "beat_this.inference":
            imported.append(name)
            return SimpleNamespace(Audio2Beats=object)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)

    with pytest.raises(BeatDetectionError, match="too short"):
        detect_grid(torch.zeros(1, 15_999), 16_000)

    assert imported == []


def test_detect_grid_keeps_four_positional_arguments(monkeypatch):
    constructor_calls = []
    signals = []

    class FakeAudio2Beats:
        def __init__(self, *, checkpoint_path, device, dbn):
            constructor_calls.append((checkpoint_path, device, dbn))

        def __call__(self, signal, sample_rate):
            assert isinstance(signal, np.ndarray)
            assert sample_rate == 16_000
            signals.append(signal)
            beats = [index * 0.5 for index in range(8)]
            return beats, beats[::4]

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "beat_this.inference":
            return SimpleNamespace(Audio2Beats=FakeAudio2Beats)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    wav = torch.stack((torch.zeros(16_000), torch.ones(16_000)))
    detect_grid(wav, 16_000, "custom-checkpoint", "cpu")

    assert constructor_calls == [("custom-checkpoint", "cpu", False)]
    np.testing.assert_array_equal(signals, [np.full(16_000, 0.5)])


@pytest.mark.parametrize(
    ("checkpoint", "device"),
    (("small0", "cpu"), ("final0", "cuda:0")),
)
def test_detect_grid_rejects_configuration_ignored_by_a_supplied_detector(
    checkpoint,
    device,
):
    detector_calls = []

    def detector(_signal, _sample_rate):
        detector_calls.append(True)
        beats = _beats(n=8)
        return beats, beats[::4]

    with pytest.raises(ValueError, match="supplied detector"):
        detect_grid(
            torch.zeros(1, 16_000),
            16_000,
            checkpoint,
            device,
            detector=detector,
        )
    assert detector_calls == []


def test_fit_tempo_recovers_tempo():
    bpm, residual = fit_tempo(_beats(103.437))
    assert abs(bpm - 103.437) / 103.437 < 1e-3
    assert residual < 1e-6


def test_tempo_residual_gate_rejects_drifting_beats():
    """A 15% tempo ramp must fail the constant-tempo gate; steady beats pass."""
    for drift, should_pass in ((0.0, True), (0.15, False)):
        bpm, residual = fit_tempo(_beats(96.0, drift=drift))
        passes = residual < MAX_TEMPO_RESIDUAL * (60.0 / bpm)
        assert passes is should_pass


def test_infer_beats_per_bar_unanimous():
    beats = _beats(103.437, n=200)
    downbeats = beats[::4]
    assert infer_beats_per_bar(beats, downbeats) == 4


def test_infer_beats_per_bar_rejects_inconsistent_bars():
    """The Tears In The Typing Pool case: 2 beats/bar in only ~64% of bars."""
    beat = 0.68
    spacings = [2 * beat] * 50 + [3 * beat] * 24 + [4 * beat] * 5
    downbeats = np.concatenate([[0.0], np.cumsum(spacings)])
    beats = np.arange(0, downbeats[-1] + beat, beat)
    assert infer_beats_per_bar(beats, downbeats) is None


def test_bar_offset_puts_a_bar_line_on_the_first_downbeat():
    grid = BeatGrid(bpm=103.437, beats_per_bar=4, first_downbeat=1.560)
    offset = grid.bar_offset()
    bar = grid.bar_seconds
    assert 0.0 <= offset < bar
    # After the shift the first downbeat sits on a bar boundary.
    shifted = grid.first_downbeat + offset
    assert abs(shifted % bar) < 1e-9 or abs(shifted % bar - bar) < 1e-9


def test_bar_offset_is_forward_only_for_any_downbeat():
    grid = BeatGrid(bpm=90.0, beats_per_bar=3, first_downbeat=0.0)
    for first in np.linspace(0.0, 10.0, 51):
        grid.first_downbeat = float(first)
        assert 0.0 <= grid.bar_offset() < grid.bar_seconds


def test_bar_offset_is_zero_without_a_meter():
    grid = BeatGrid(bpm=120.0, beats_per_bar=None, first_downbeat=3.7)
    assert grid.bar_seconds is None
    assert grid.bar_offset() == 0.0


def test_onset_phase_places_onsets_in_continuous_beats():
    beats = _beats(120.0, n=5)  # every 0.5 s
    phase = get_onsets_phase([0.0, 0.25, 0.5, 2.0], beats)
    np.testing.assert_allclose(phase, [0.0, 0.5, 1.0, 4.0])


def test_onset_phase_drops_onsets_outside_the_tracked_span():
    beats = _beats(120.0, n=5, start=1.0)  # 1.0 s … 3.0 s
    assert len(get_onsets_phase([0.5, 1.0, 2.0, 3.5], beats)) == 2


def test_onset_phase_counts_each_onset_time_once():
    """A chord is one vote, not one per note."""
    beats = _beats(120.0, n=5)
    assert len(get_onsets_phase([1.0, 1.0, 1.0004, 1.5], beats)) == 2


def test_estimate_onset_delay_recovers_a_known_delay():
    beats = _beats(126.0)
    measured = estimate_onset_delay(_onsets(beats, delay=0.012), _grid(beats, 126.0))
    assert measured is not None
    # 1 ms of slack for the millisecond rounding in onset_phase.
    assert abs(measured.seconds - 0.012) < 0.001
    assert measured.subdivision == 4
    assert measured.concentration > 0.99


def test_estimate_onset_delay_recovers_a_negative_delay():
    beats = _beats(126.0)
    measured = estimate_onset_delay(_onsets(beats, delay=-0.015), _grid(beats, 126.0))
    assert measured is not None
    assert abs(measured.seconds + 0.015) < 0.001


def test_estimate_onset_delay_reads_sparse_onsets():
    """Eighth notes fit the 1/2 grid and every multiple of it, all saying the same."""
    beats = _beats(120.0)
    measured = estimate_onset_delay(
        _onsets(beats, subdivision=2, delay=0.008), _grid(beats, 120.0)
    )
    assert measured is not None
    assert measured.subdivision % 2 == 0
    assert abs(measured.seconds - 0.008) < 0.001


def test_estimate_onset_delay_needs_enough_onsets():
    beats = _beats(120.0)
    onsets = _onsets(beats, subdivision=1, delay=0.01)[: MIN_ONSETS - 1]
    assert estimate_onset_delay(onsets, _grid(beats, 120.0)) is None


def test_estimate_onset_delay_rejects_onsets_off_the_grid():
    """Onsets scattered through the beat have no phase worth reading."""
    beats = _beats(120.0)
    rng = np.random.default_rng(0)
    onsets = rng.uniform(beats[0], beats[-1], 400)
    assert estimate_onset_delay(onsets, _grid(beats, 120.0)) is None


def test_estimate_onset_delay_refuses_an_implausible_offset():
    """Well past the cap is a badly chosen grid, not a transcription delay.

    It takes a coarse grid to even express such an offset — a slow song, and
    onsets no finer than half-beats — which is the case the cap is there for.
    """
    beats = _beats(60.0)
    onsets = _onsets(beats, subdivision=2, delay=1.5 * MAX_ONSET_DELAY_S)
    assert estimate_onset_delay(onsets, _grid(beats, 60.0)) is None


def test_estimate_onset_delay_reads_the_offset_modulo_one_subdivision():
    """Landing a subdivision late is indistinguishable from landing on time.

    The phase angle wraps, so the correction can only ever be a fraction of a
    subdivision — it fixes where notes sit inside the beat, never which beat.
    """
    beats = _beats(120.0)  # 1/4 beat = 125 ms, the spacing of the onsets below
    measured = estimate_onset_delay(
        _onsets(beats, delay=0.125 + 0.01), _grid(beats, 120.0)
    )
    assert measured is not None
    # A whole 125 ms subdivision of the delay is invisible; the 10 ms is what
    # comes back. Any grid the onsets exactly fit reports the same, so which one
    # is picked is not part of the answer.
    assert abs(measured.seconds - 0.01) < 0.001


def test_estimate_onset_delay_survives_beat_quantization():
    """Beats reported on beat_this's 20 ms frame grid still resolve a 10 ms delay.

    The tracker only ever reports multiples of 20 ms, which alone could not
    express the delay; the true beat phase drifting against that grid dithers the
    rounding, so the average over a whole song still lands on it.
    """
    exact = _beats(126.0, n=200)
    quantized = np.round(exact / 0.02) * 0.02
    grid = _grid(quantized, fit_tempo(quantized)[0])  # bpm as detect_grid fits it
    measured = estimate_onset_delay(_onsets(exact, delay=0.010), grid)
    assert measured is not None
    assert abs(measured.seconds - 0.010) < 0.003


def test_with_onset_delay_measures_how_late_the_notes_are():
    beats = _beats(126.0, start=0.4)
    grid = BeatGrid(bpm=126.0, beats_per_bar=4, first_downbeat=0.4, beats=beats)
    measured = grid.with_onset_delay(_onsets(beats, delay=0.012))
    assert abs(measured.onset_delay - 0.012) < 0.001
    # Only the lag is filled in; the grid itself is left exactly as detected.
    assert measured == dataclasses.replace(grid, onset_delay=measured.onset_delay)


def test_with_onset_delay_is_zero_without_tracked_beats():
    """A hand-built grid carries no beats, so there is nothing to measure."""
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=1.0)
    assert grid.with_onset_delay(_onsets(_beats(120.0), delay=0.02)).onset_delay == 0.0


def test_with_onset_delay_is_zero_without_usable_onsets():
    """Measured and found nothing is 0.0, not None — it won't be measured again."""
    beats = _beats(120.0)
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=1.0, beats=beats)
    assert grid.with_onset_delay([]).onset_delay == 0.0
    assert grid.with_onset_delay([0.5, 1.0, 1.5]).onset_delay == 0.0


class _FakeMidi:
    def __init__(self, texts):
        self.tracks = [[_FakeMarker(t) for t in texts]]


class _FakeMarker:
    type = "marker"

    def __init__(self, text):
        self.text = text


def test_read_bar_offset():
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}0.7945"])) == 0.7945
    assert read_bar_offset(_FakeMidi([])) == 0.0
    assert read_bar_offset(_FakeMidi(["some other marker"])) == 0.0
    assert read_bar_offset(_FakeMidi([f"{BAR_OFFSET_MARKER}nonsense"])) == 0.0
