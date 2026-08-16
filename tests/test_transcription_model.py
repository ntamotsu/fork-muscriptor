"""Tests for TranscriptionModel._generate_token_stream.

These check the streaming contract of the token stream without a real model:
a fake `generate()` yields one row (`[batch]`) per timestep and records how
many timesteps have been pulled, so we can assert that a chunk's events come
out *as soon as that chunk finishes* — before the rest of the batch is even
generated.
"""

import copy
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import ModuleType, SimpleNamespace

import pytest
import torch

from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.beats import BeatDetectionError, _LazyAudio2Beats

EOS = 99


def _run(
    batches,
    *,
    batch_size,
    seek_times,
    no_eos_is_ok=False,
    beam_size=1,
    generate_calls=None,
    closed_calls=None,
):
    """Drive _generate_token_stream with a fake model.

    ``batches`` is one list of rows per expected ``generate()`` call; each row
    is the per-chunk token for one timestep. Returns ``(stream, pulled)`` where
    ``pulled`` grows by one entry every time the fake yields a timestep, so its
    length is how far generation has progressed.
    """
    pulled: list[list[int]] = []
    calls = iter(batches)

    def generate(**kwargs):
        try:
            if generate_calls is not None:
                generate_calls.append(kwargs)
            for row in next(calls):
                pulled.append(row)
                yield torch.tensor(row)
        finally:
            if closed_calls is not None:
                closed_calls.append(True)

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    conditions = [object()] * len(seek_times)
    stream = TranscriptionModel._generate_token_stream(
        fake,
        conditions,
        seek_times,
        batch_size,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=no_eos_is_ok,
        # The fake tokenizer has no vocab; prelude forcing has its own tests
        # (test_prelude_forcing.py).
        prelude_forcing=False,
        beam_size=beam_size,
    )
    return stream, pulled


def _stream_from_steps(steps):
    """既成のstep iteratorをfake model経由でtoken streamへ接続する。"""
    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=lambda **_kwargs: steps),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    return TranscriptionModel._generate_token_stream(
        fake,
        [object()],
        [0.0],
        batch_size=1,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=False,
        prelude_forcing=False,
    )


# ---------------------------------------------------------------------------
# Emitted as soon as possible
# ---------------------------------------------------------------------------


def test_first_chunk_streams_before_the_batch_finishes():
    # batch of 2 chunks: chunk 0 ends at row 2, chunk 1 only at row 4.
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, pulled = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, 5.0)
    assert len(pulled) == 0  # the boundary is emitted before any generation
    assert next(it) == 10
    assert len(pulled) == 1  # first token after a single timestep
    assert next(it) == 11
    # Chunk 0 is fully streamed having generated only its own timesteps —
    # chunk 1 (which finishes at row 4) has not been generated to completion.
    assert len(pulled) == 2


def test_single_chunk_streams_token_by_token():
    rows = [[10], [11], [12], [EOS]]
    stream, pulled = _run([rows], batch_size=1, seek_times=[0.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, None)
    assert len(pulled) == 0
    for expected, count in [(10, 1), (11, 2), (12, 3)]:
        assert next(it) == expected
        assert len(pulled) == count


def test_single_beam_delegates_eos_stop_to_the_host():
    generate_calls = []
    stream, _ = _run(
        [[[10], [EOS]]],
        batch_size=1,
        seek_times=[0.0],
        generate_calls=generate_calls,
    )

    list(stream)

    assert [call["early_stop_on_token"] for call in generate_calls] == [None]


def test_single_beam_stops_and_closes_after_staggered_eos():
    rows = [[10, 20], [11, EOS], [12, 777], [EOS, 888], [889, 890]]
    closed_calls = []
    stream, pulled = _run(
        [rows],
        batch_size=2,
        seek_times=[0.0, 5.0],
        closed_calls=closed_calls,
    )

    events = list(stream)

    assert (events, pulled, closed_calls) == (
        [
            ChunkBoundary(0.0, 5.0),
            10,
            11,
            12,
            ChunkBoundary(5.0, None),
            20,
            ProgressEvent(completed=2, total=2),
        ],
        rows[:4],
        [True],
    )


def test_closing_token_stream_closes_active_model_steps():
    class ClosableSteps:
        def __init__(self):
            self.rows = iter([torch.tensor([10]), torch.tensor([EOS])])
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.rows)

        def close(self):
            self.closed = True

    steps = ClosableSteps()
    stream = _stream_from_steps(steps)

    assert next(stream) == ChunkBoundary(0.0, None)
    assert next(stream) == 10
    stream.close()

    assert steps.closed is True


def test_model_step_error_is_propagated_and_iterator_is_closed():
    class FailingSteps:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return torch.tensor([10])
            raise RuntimeError("fake generation failure")

        def close(self):
            self.closed = True

    steps = FailingSteps()
    stream = _stream_from_steps(steps)

    assert next(stream) == ChunkBoundary(0.0, None)
    assert next(stream) == 10
    with pytest.raises(RuntimeError, match="fake generation failure"):
        next(stream)

    assert steps.closed is True


def test_token_stream_accepts_model_steps_without_close():
    steps = iter([torch.tensor([EOS])])

    assert list(_stream_from_steps(steps)) == [
        ChunkBoundary(0.0, None),
        ProgressEvent(completed=1, total=1),
    ]


def test_beam_search_keeps_eos_stopping_inside_the_model():
    rows = [[10], [EOS], [777]]
    generate_calls = []
    stream, pulled = _run(
        [rows],
        batch_size=1,
        seek_times=[0.0],
        beam_size=2,
        generate_calls=generate_calls,
    )

    events = list(stream)

    assert (
        events,
        pulled,
        [call["early_stop_on_token"] for call in generate_calls],
    ) == (
        [
            ChunkBoundary(0.0, None),
            10,
            ProgressEvent(completed=1, total=1),
        ],
        rows,
        [EOS],
    )


# ---------------------------------------------------------------------------
# Ordering and buffering
# ---------------------------------------------------------------------------


def test_full_stream_order_for_a_batch():
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        22,
        23,
        # End of the (only) batch: both chunks done.
        ProgressEvent(completed=2, total=2),
    ]


def test_later_chunk_finishing_first_is_buffered_until_its_turn():
    # chunk 1 hits EOS (row 1) before chunk 0 (row 3); its tokens must wait.
    rows = [[10, 20], [11, EOS], [12, 88], [EOS, 88]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        12,
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


def test_chunks_across_multiple_batches_stay_in_order():
    # batch_size=1 → one generate() call per chunk.
    batches = [[[10], [11], [EOS]], [[20], [EOS]]]
    stream, _ = _run(batches, batch_size=1, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        # batch_size=1 => a completion anchor trails each chunk.
        ProgressEvent(completed=1, total=2),
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Missing EOS
# ---------------------------------------------------------------------------


def test_missing_eos_raises_by_default():
    rows = [[10, 20], [11, 21]]  # neither chunk emits EOS
    closed_calls = []
    stream, _ = _run(
        [rows],
        batch_size=2,
        seek_times=[0.0, 5.0],
        closed_calls=closed_calls,
    )
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)
    assert closed_calls == [True]


def test_missing_eos_warns_and_still_emits_when_allowed():
    rows = [[10, 20], [11, 21]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0], no_eos_is_ok=True)
    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        events = list(stream)
    assert events == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Tempo detection modes
# ---------------------------------------------------------------------------


class _FakeAudio(TranscriptionModel):
    """Just enough of a model for detect_beat_grid_for's mode dispatch."""

    _load_wav = staticmethod(lambda tensor, sr: tensor)

    def __init__(self):
        super().__init__(
            model=object(),
            tokenizer=SimpleNamespace(group_program_map={}),
            device=torch.device("cpu"),
        )


def _install_fake_beat_this(monkeypatch, audio2beats):
    beat_this = ModuleType("beat_this")
    inference = ModuleType("beat_this.inference")
    inference.Audio2Beats = audio2beats
    beat_this.inference = inference
    monkeypatch.setitem(sys.modules, "beat_this", beat_this)
    monkeypatch.setitem(sys.modules, "beat_this.inference", inference)


def _tempo_model():
    return TranscriptionModel(
        model=object(),
        tokenizer=SimpleNamespace(group_program_map={}),
        device=torch.device("cpu"),
    )


def _fake_beat_predictions():
    beats = [index * 0.5 for index in range(8)]
    return beats, beats[::4]


class _MinimalTranscriber(TranscriptionModel):
    @staticmethod
    def _resolve_batch_size(_batch_size, _prelude_forcing):
        return 1

    @staticmethod
    def _load_wav(tensor, _sample_rate):
        return tensor

    @staticmethod
    def _build_conditions(_chunk, _instrument_group):
        return [object()]

    def __init__(self):
        super().__init__(
            model=object(),
            tokenizer=SimpleNamespace(
                group_program_map={},
                _vocab=[],
                frame_rate=100,
            ),
            device=torch.device("cpu"),
        )
        self.profile_seen = None

    def _generate_token_stream(self, *_args, profile=False, **_kwargs):
        self.profile_seen = profile
        return iter(())


class _PreparedAudioRecorder(_MinimalTranscriber):
    def __init__(self):
        super().__init__()
        self.load_calls = 0
        self.prepared = torch.ones(1, 100)
        self.transcription_wavs = []

    def _load_wav(self, _audio, _sample_rate):
        self.load_calls += 1
        return self.prepared

    def _transcribe_prepared(self, wav, **_kwargs):
        self.transcription_wavs.append(wav)
        yield ProgressEvent(completed=0, total=0)

    @staticmethod
    def events_to_midi_bytes(events, *, beat_grid):
        list(events)
        assert beat_grid == "beat-grid"
        return b"MIDI"


def test_transcribe_keeps_progress_on_stderr_without_profiling_sync(
    monkeypatch, capsys
):
    synchronize_calls = []
    monkeypatch.setattr(
        "muscriptor.transcription_model.muscriptor.accelerator.synchronize",
        lambda device=None: synchronize_calls.append(device),
    )
    events = list(_MinimalTranscriber().transcribe((torch.zeros(1, 100), 16_000)))
    captured = capsys.readouterr()

    assert (len(events), synchronize_calls, captured.out) == (1, [], "")
    assert "[muscriptor] audio:" in captured.err


def test_transcribe_can_suppress_progress_output_for_benchmarks(capsys):
    events = list(
        _MinimalTranscriber().transcribe(
            (torch.zeros(1, 100), 16_000),
            log_progress=False,
        )
    )

    captured = capsys.readouterr()
    assert len(events) == 1
    assert (captured.out, captured.err) == ("", "")


def test_transcribe_remains_lazy_when_using_a_prepared_audio_path():
    model = _PreparedAudioRecorder()

    events = model.transcribe((torch.zeros(1, 100), 16_000), log_progress=False)

    assert model.load_calls == 0
    assert list(events) == [ProgressEvent(completed=0, total=0)]
    assert model.load_calls == 1
    assert model.transcription_wavs == [model.prepared]


def test_midi_prepares_audio_once_and_shares_the_same_tensor(monkeypatch):
    model = _PreparedAudioRecorder()
    beat_wavs = []

    def fake_detect_grid(wav, _sample_rate, **_kwargs):
        beat_wavs.append(wav)
        return "beat-grid"

    monkeypatch.setattr("muscriptor.transcription_model.detect_grid", fake_detect_grid)

    result = model.transcribe_to_midi(
        (torch.zeros(2, 44_100), 44_100),
        detect_tempo=True,
        log_progress=False,
    )

    assert result == b"MIDI"
    assert model.load_calls == 1
    assert beat_wavs == [model.prepared]
    assert model.transcription_wavs == [model.prepared]


def test_inherited_midi_dispatches_to_public_overrides_without_private_state():
    calls = []
    audio = (torch.zeros(1, 16_000), 16_000)

    class PublicOnlyTranscriber(TranscriptionModel):
        def __init__(self):
            pass

        def detect_beat_grid_for(self, received_audio, mode="best-effort"):
            calls.append(("detect", received_audio, mode))
            return "public-grid"

        def transcribe(self, received_audio, **kwargs):
            calls.append(("transcribe", received_audio, kwargs))
            return iter(("public-event",))

        def events_to_midi_bytes(self, events, beat_grid=None):
            calls.append(("midi", list(events), beat_grid))
            return b"PUBLIC_MIDI"

    result = PublicOnlyTranscriber().transcribe_to_midi(
        audio,
        use_sampling=True,
        temperature=0.5,
        cfg_coef=2.0,
        instruments=["piano"],
        batch_size=3,
        no_eos_is_ok=False,
        beam_size=2,
        prelude_forcing=False,
        detect_tempo=True,
        profile=True,
        log_progress=False,
    )

    assert result == b"PUBLIC_MIDI"
    assert calls == [
        ("detect", audio, True),
        (
            "transcribe",
            audio,
            {
                "use_sampling": True,
                "temperature": 0.5,
                "cfg_coef": 2.0,
                "instruments": ["piano"],
                "batch_size": 3,
                "no_eos_is_ok": False,
                "beam_size": 2,
                "prelude_forcing": False,
                "profile": True,
                "log_progress": False,
            },
        ),
        ("midi", ["public-event"], "public-grid"),
    ]


def test_midi_detects_public_entrypoints_replaced_on_one_instance(monkeypatch):
    model = _PreparedAudioRecorder()
    calls = []
    audio = (torch.zeros(1, 16_000), 16_000)

    def detect(received_audio, mode="best-effort"):
        calls.append(("detect", received_audio, mode))
        return "beat-grid"

    def transcribe(received_audio, **_kwargs):
        calls.append(("transcribe", received_audio))
        return iter(("public-event",))

    monkeypatch.setattr(model, "detect_beat_grid_for", detect)
    monkeypatch.setattr(model, "transcribe", transcribe)

    result = model.transcribe_to_midi(audio, detect_tempo=True)

    assert result == b"MIDI"
    assert calls == [("detect", audio, True), ("transcribe", audio)]
    assert (model.load_calls, model.transcription_wavs) == (0, [])


def test_transcribe_forwards_profile_to_token_generation(monkeypatch):
    monkeypatch.setattr(
        "muscriptor.transcription_model.muscriptor.accelerator.synchronize",
        lambda _device=None: None,
    )
    model = _MinimalTranscriber()

    list(model.transcribe((torch.zeros(1, 100), 16_000), profile=True))

    assert model.profile_seen is True


def test_transcribe_profile_writes_timings_only_to_stderr(monkeypatch, capsys):
    synchronize_calls = []
    monkeypatch.setattr(
        "muscriptor.transcription_model.muscriptor.accelerator.synchronize",
        lambda device=None: synchronize_calls.append(device),
    )

    list(_MinimalTranscriber().transcribe((torch.zeros(1, 100), 16_000), profile=True))
    captured = capsys.readouterr()

    assert captured.out == ""
    assert synchronize_calls
    assert set(synchronize_calls) == {torch.device("cpu")}
    assert "[muscriptor] load audio:" in captured.err
    assert "[muscriptor] build conditions:" in captured.err
    assert "[muscriptor] generate total:" not in captured.err
    assert "[muscriptor] transcribe total:" not in captured.err


def test_detect_tempo_modes(monkeypatch):
    def boom(*args, **kwargs):
        raise BeatDetectionError("no fixed tempo")

    monkeypatch.setattr("muscriptor.transcription_model.detect_grid", boom)
    model = _FakeAudio()
    # false: never even calls the detector.
    assert model.detect_beat_grid_for((None, None), False) is None
    # best-effort: swallows the failure, no grid written.
    assert model.detect_beat_grid_for((None, None), "best-effort") is None
    # true: the caller wanted to know.
    with pytest.raises(BeatDetectionError):
        model.detect_beat_grid_for((None, None), True)


def test_tempo_detector_is_lazily_reused_by_one_model(monkeypatch):
    constructor_calls = []
    inference_calls = []

    class FakeAudio2Beats:
        def __init__(self, *, checkpoint_path, device, dbn):
            constructor_calls.append((checkpoint_path, device, dbn))

        def __call__(self, signal, sample_rate):
            inference_calls.append((signal, sample_rate))
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    model = _tempo_model()
    wav = torch.zeros(1, 16_000)

    model.detect_beat_grid_for((wav, 16_000), True)
    model.detect_beat_grid_for((wav, 16_000), True)

    assert constructor_calls == [("final0", "cpu", False)]
    assert len(inference_calls) == 2


@pytest.mark.parametrize(
    "clone",
    (copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))),
    ids=("deepcopy", "pickle"),
)
def test_lazy_tempo_detector_serialization_restores_an_empty_cpu_cache(
    monkeypatch,
    clone,
):
    constructor_configs = []

    class FakeAudio2Beats:
        def __init__(self, **kwargs):
            constructor_configs.append((kwargs["checkpoint_path"], kwargs["device"]))

        def __call__(self, _signal, _sample_rate):
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    detector = _LazyAudio2Beats(checkpoint="custom", device="cpu")
    signal = torch.zeros(16_000).numpy()
    detector(signal, 16_000)

    restored = clone(detector)
    assert restored._detector is None
    assert restored._lock is not detector._lock
    restored(signal, 16_000)

    assert constructor_configs == [("custom", "cpu"), ("custom", "cpu")]


def test_tempo_detector_constructor_failure_is_retried(monkeypatch):
    constructor_calls = 0
    inference_calls = 0

    class FakeAudio2Beats:
        def __init__(self, **_kwargs):
            nonlocal constructor_calls
            constructor_calls += 1
            if constructor_calls == 1:
                raise RuntimeError("constructor failed")

        def __call__(self, _signal, _sample_rate):
            nonlocal inference_calls
            inference_calls += 1
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    model = _tempo_model()
    audio = (torch.zeros(1, 16_000), 16_000)

    with pytest.raises(RuntimeError, match="constructor failed"):
        model.detect_beat_grid_for(audio, True)
    model.detect_beat_grid_for(audio, True)
    model.detect_beat_grid_for(audio, True)

    assert (constructor_calls, inference_calls) == (2, 2)


def test_tempo_detector_is_kept_after_an_input_specific_failure(monkeypatch):
    constructor_calls = 0
    inference_calls = 0

    class FakeAudio2Beats:
        def __init__(self, **_kwargs):
            nonlocal constructor_calls
            constructor_calls += 1

        def __call__(self, _signal, _sample_rate):
            nonlocal inference_calls
            inference_calls += 1
            if inference_calls == 1:
                raise BeatDetectionError("no beat for this input")
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    model = _tempo_model()
    audio = (torch.zeros(1, 16_000), 16_000)

    with pytest.raises(BeatDetectionError, match="this input"):
        model.detect_beat_grid_for(audio, True)
    model.detect_beat_grid_for(audio, True)

    assert (constructor_calls, inference_calls) == (1, 2)


def test_tempo_detector_is_not_shared_between_models(monkeypatch):
    detectors = []

    class FakeAudio2Beats:
        def __init__(self, **_kwargs):
            detectors.append(self)

        def __call__(self, _signal, _sample_rate):
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    audio = (torch.zeros(1, 16_000), 16_000)

    _tempo_model().detect_beat_grid_for(audio, True)
    _tempo_model().detect_beat_grid_for(audio, True)

    assert len(detectors) == 2
    assert detectors[0] is not detectors[1]


def test_tempo_detector_is_not_built_when_disabled_or_audio_is_too_short(monkeypatch):
    constructor_calls = []

    class FakeAudio2Beats:
        def __init__(self, **_kwargs):
            constructor_calls.append(True)

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    model = _tempo_model()

    assert model.detect_beat_grid_for((torch.zeros(1, 16_000), 16_000), False) is None
    with pytest.raises(BeatDetectionError, match="too short"):
        model.detect_beat_grid_for((torch.zeros(1, 15_999), 16_000), True)

    assert constructor_calls == []


def test_tempo_detector_concurrent_first_use_builds_once_and_serializes_inference(
    monkeypatch,
):
    workers = 8
    start = Barrier(workers)
    state_lock = Lock()
    constructor_calls = 0
    inference_calls = 0
    active_inferences = 0
    max_active_inferences = 0

    class FakeAudio2Beats:
        def __init__(self, **_kwargs):
            nonlocal constructor_calls
            time.sleep(0.01)
            with state_lock:
                constructor_calls += 1

        def __call__(self, _signal, _sample_rate):
            nonlocal inference_calls, active_inferences, max_active_inferences
            with state_lock:
                inference_calls += 1
                active_inferences += 1
                max_active_inferences = max(max_active_inferences, active_inferences)
            time.sleep(0.01)
            with state_lock:
                active_inferences -= 1
            return _fake_beat_predictions()

    _install_fake_beat_this(monkeypatch, FakeAudio2Beats)
    model = _tempo_model()
    audio = (torch.zeros(1, 16_000), 16_000)

    def detect(_index):
        start.wait(timeout=5)
        return model.detect_beat_grid_for(audio, True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        grids = list(pool.map(detect, range(workers)))

    assert len(grids) == workers
    assert (constructor_calls, inference_calls, max_active_inferences) == (
        1,
        workers,
        1,
    )
