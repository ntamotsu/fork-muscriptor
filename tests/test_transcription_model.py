"""Tests for TranscriptionModel._generate_token_stream.

These check the streaming contract of the token stream without a real model:
a fake `generate()` yields one row (`[batch]`) per timestep and records how
many timesteps have been pulled, so we can assert that a chunk's events come
out *as soon as that chunk finishes* — before the rest of the batch is even
generated.
"""

from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.beats import BeatDetectionError

EOS = 99


def _run(batches, *, batch_size, seek_times, no_eos_is_ok=False):
    """Drive _generate_token_stream with a fake model.

    ``batches`` is one list of rows per expected ``generate()`` call; each row
    is the per-chunk token for one timestep. Returns ``(stream, pulled)`` where
    ``pulled`` grows by one entry every time the fake yields a timestep, so its
    length is how far generation has progressed.
    """
    pulled: list[list[int]] = []
    calls = iter(batches)

    def generate(**kwargs):
        for row in next(calls):
            pulled.append(row)
            yield torch.tensor(row)

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
    )
    return stream, pulled


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
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)


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


class _FakeAudio:
    """Just enough of a model for detect_beat_grid_for's mode dispatch."""

    _load_wav = staticmethod(lambda tensor, sr: tensor)
    detect_beat_grid_for = TranscriptionModel.detect_beat_grid_for


class _MinimalTranscriber:
    transcribe = TranscriptionModel.transcribe
    _device = torch.device("cpu")
    _tokenizer = SimpleNamespace(_vocab=[], frame_rate=100)
    _instrument_for_program = staticmethod(lambda _program: "piano")

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
        self.profile_seen = None

    def _generate_token_stream(self, *_args, profile=False, **_kwargs):
        self.profile_seen = profile
        return iter(())


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
