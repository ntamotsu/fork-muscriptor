"""Hermetic tests for the FastAPI transcription server.

Uses a fake transcriber so no weights / audio decoding is required.
"""

import base64
import inspect
import io
import json
import threading
import time
import wave
import zipfile
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
from fastapi.testclient import TestClient

import muscriptor.server as server_module
from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.server import create_app, event_to_dict
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.sheets import MuseScoreError, MuseScoreNotFoundError

FAKE_MIDI = b"FAKE_MIDI_BYTES"


def _bind_base_audio_entrypoints(model):
    """prepared高速経路を表すmockへ基底の公開entrypointを束縛する。"""
    model.transcribe = TranscriptionModel.transcribe.__get__(model, TranscriptionModel)
    model.detect_beat_grid_for = TranscriptionModel.detect_beat_grid_for.__get__(
        model, TranscriptionModel
    )


def make_model(events=(), midi=FAKE_MIDI):
    """A mock standing in for TranscriptionModel.

    Autospec'd against the real class so the mock fakes isinstance and keeps
    method signatures in sync with what the server calls — but no weights or
    audio decoding are loaded.
    """
    model = create_autospec(TranscriptionModel, instance=True)
    model._device = "cpu"
    model._transcribe_prepared.return_value = list(events)
    model._detect_beat_grid_prepared.return_value = None
    model.events_to_midi_bytes.return_value = midi
    model.transcribe_to_midi.return_value = midi
    _bind_base_audio_entrypoints(model)
    return model


def _wav_bytes(tmp_path: Path) -> bytes:
    """Write a tiny silent WAV so the upload payload is a real file."""
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return p.read_bytes()


def _flac_bytes(sample_rate: int = 22050, n_frames: int = 1600) -> bytes:
    """Encode a tiny silent mono FLAC in memory via soundfile (non-WAV path)."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(n_frames, dtype="float32"), sample_rate, format="FLAC")
    return buf.getvalue()


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE `data: <json>` lines into a list of dicts."""
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), f"unexpected SSE chunk: {chunk!r}"
        out.append(json.loads(chunk[len("data: ") :]))
    return out


def test_event_to_dict_start_and_end():
    start = NoteStartEvent(pitch=60, start_time=0.5, index=0, instrument="piano")
    end = NoteEndEvent(end_time=1.5, start_event=start)
    assert event_to_dict(start) == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.5,
        "index": 0,
        "instrument": "piano",
    }
    assert event_to_dict(end) == {
        "type": "end",
        "end_time": 1.5,
        "start_event_index": 0,
    }


def test_create_app_accepts_an_opt_in_profile_setting():
    parameter = inspect.signature(create_app).parameters.get("profile")

    assert parameter is not None
    assert parameter.default is False


def test_transcribe_streams_sse_events(tmp_path):
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    s1 = NoteStartEvent(pitch=64, start_time=0.1, index=1, instrument="guitar")
    events = [
        s0,
        NoteEndEvent(end_time=0.5, start_event=s0),
        s1,
        NoteEndEvent(end_time=0.6, start_event=s1),
    ]
    model = make_model(events)
    client = TestClient(create_app(model))

    payload = _wav_bytes(tmp_path)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    parsed = _parse_sse(resp.text)
    # Note events, then a trailing base64-encoded MIDI event.
    assert parsed[:-1] == [event_to_dict(e) for e in events]
    assert parsed[-1] == {
        "type": "transcription_complete",
        "data": base64.b64encode(FAKE_MIDI).decode("ascii"),
        "beat_grid": None,
    }
    assert model._transcribe_prepared.call_count == 1


def test_transcribe_stream_supports_a_public_only_adapter(tmp_path):
    start = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    end = NoteEndEvent(end_time=0.5, start_event=start)

    class PublicOnlyAdapter:
        def __init__(self):
            self.calls = []

        def transcribe(
            self,
            audio,
            *,
            instruments,
            batch_size,
            no_eos_is_ok,
            profile,
        ):
            self.calls.append(
                (
                    "transcribe",
                    audio,
                    instruments,
                    batch_size,
                    no_eos_is_ok,
                    profile,
                )
            )
            return iter((start, end))

        def detect_beat_grid_for(self, audio, mode="best-effort"):
            self.calls.append(("detect", audio, mode))
            return None

        def events_to_midi_bytes(self, events, beat_grid=None):
            self.calls.append(("midi", list(events), beat_grid))
            return FAKE_MIDI

    model = PublicOnlyAdapter()
    client = TestClient(create_app(model, profile=True))

    response = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin"], "detect_tempo": "false"},
    )

    assert response.status_code == 200
    assert _parse_sse(response.text)[-1]["type"] == "transcription_complete"
    assert [call[0] for call in model.calls] == ["transcribe", "detect", "midi"]
    assert model.calls[0][2:] == (["violin"], 1, True, True)
    assert model.calls[1][2] is False
    assert model.calls[2][1:] == ([start, end], None)


def test_transcribe_stream_honors_one_public_override_on_a_subclass(tmp_path):
    class PublicTranscribeOverride(TranscriptionModel):
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **kwargs):
            self.calls.append(("transcribe", audio, kwargs))
            return iter(())

        def events_to_midi_bytes(self, events, beat_grid=None):
            self.calls.append(("midi", list(events), beat_grid))
            return FAKE_MIDI

    model = PublicTranscribeOverride()
    response = TestClient(create_app(model)).post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"detect_tempo": "false"},
    )

    assert response.status_code == 200
    assert [call[0] for call in model.calls] == ["transcribe", "midi"]
    assert model.calls[0][2]["profile"] is False
    assert model.calls[1][1:] == ([], None)


def test_transcribe_stream_preserves_profile_and_progress_logging(tmp_path, capsys):
    model = make_model()
    client = TestClient(create_app(model, profile=True))

    response = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )

    assert response.status_code == 200
    kwargs = model._transcribe_prepared.call_args.kwargs
    profile_output = capsys.readouterr().err
    assert (
        kwargs.get("profile"),
        kwargs.get("log_progress"),
        profile_output.count("[muscriptor] load audio:"),
    ) == (True, True, 1)


def test_transcribe_prepares_audio_once_and_reuses_the_same_tensor(tmp_path):
    import torch

    model = make_model()
    prepared = torch.zeros(1, 1600)
    model._prepare_audio.return_value = prepared
    client = TestClient(create_app(model))

    response = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )

    transcribed = (
        model._transcribe_prepared.call_args.args[0]
        if model._transcribe_prepared.called
        else None
    )
    detected = (
        model._detect_beat_grid_prepared.call_args.args[0]
        if model._detect_beat_grid_prepared.called
        else None
    )
    assert (
        response.status_code,
        model._prepare_audio.call_count,
        transcribed is prepared,
        detected is prepared,
    ) == (200, 1, True, True)


def test_transcribe_passes_false_to_prepared_tempo_detection(tmp_path):
    import torch

    model = make_model()
    prepared = torch.zeros(1, 1600)
    model._prepare_audio.return_value = prepared
    client = TestClient(create_app(model))

    response = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"detect_tempo": "false"},
    )

    detected_wav, mode = model._detect_beat_grid_prepared.call_args.args
    assert (response.status_code, detected_wav is prepared, mode) == (
        200,
        True,
        False,
    )


def test_transcribe_propagates_model_errors_and_releases_the_lock(tmp_path):
    model = make_model()

    def fail(*_args, **_kwargs):
        raise RuntimeError("fake transcription failure")
        yield

    model._transcribe_prepared.side_effect = fail
    client = TestClient(create_app(model))
    request = {"files": {"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")}}

    with pytest.raises(RuntimeError, match="fake transcription failure"):
        client.post("/transcribe", **request)

    model._transcribe_prepared.side_effect = None
    model._transcribe_prepared.return_value = []
    response = client.post("/transcribe", **request)
    assert response.status_code == 200


def test_transcribe_sends_beat_grid(tmp_path):
    """The detected grid rides along with the final MIDI event, for the UI's bar lines."""
    model = make_model()
    model._detect_beat_grid_prepared.return_value = BeatGrid(
        # A real detected grid carries `beats`, which must not reach the JSON.
        bpm=123.5,
        beats_per_bar=4,
        first_downbeat=0.75,
        beats=np.array([0.75, 1.24]),
    )
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert (
        _parse_sse(resp.text)[-1]["beat_grid"],
        model._detect_beat_grid_prepared.call_args.args[1],
    ) == (
        {
            "bpm": 123.5,
            "beats_per_bar": 4,
            "first_downbeat": 0.75,
            # Two beats and a handful of notes are far too little to measure a lag
            # from, so the UI is told to shift its notes by nothing.
            "onset_delay": 0.0,
        },
        "best-effort",
    )


def test_transcribe_forwards_progress(tmp_path):
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    events = [
        ProgressEvent(completed=0, total=2),
        s0,
        NoteEndEvent(end_time=0.5, start_event=s0),
        ProgressEvent(completed=1, total=2),
        ProgressEvent(completed=2, total=2),
    ]
    model = make_model(events)
    client = TestClient(create_app(model))

    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)

    # Progress events surface as their own SSE type, interleaved with notes.
    assert {"type": "progress", "completed": 0, "total": 2} in parsed
    assert {"type": "progress", "completed": 2, "total": 2} in parsed
    # ...but are kept out of the note list the MIDI file is built from.
    (built,) = model.events_to_midi_bytes.call_args.args
    assert all(not isinstance(e, ProgressEvent) for e in built)


def test_transcribe_empty_stream(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    # No notes, but the trailing MIDI event is still emitted.
    assert _parse_sse(resp.text) == [
        {
            "type": "transcription_complete",
            "data": base64.b64encode(FAKE_MIDI).decode("ascii"),
            "beat_grid": None,
        }
    ]


def test_transcribe_missing_file():
    client = TestClient(create_app(make_model()))
    resp = client.post("/transcribe")
    assert resp.status_code == 422


def test_transcribe_passes_tensor_not_path(tmp_path):
    """Server must hand the model an in-memory (tensor, sr) tuple — no disk."""
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    audio = model._prepare_audio.call_args.args[0]
    assert isinstance(audio, tuple)
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16000
    assert tensor.shape[-1] == 1600  # samples we wrote


def test_transcribe_rejects_invalid_wav():
    # An undecodable upload is the client's fault: the endpoint reports 400.
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe",
        files={"file": ("garbage.wav", b"not a wav at all", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_accepts_non_wav_audio():
    """A non-WAV upload (FLAC) decodes via soundfile and reaches the model."""
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("clip.flac", _flac_bytes(sample_rate=22050), "audio/flac")},
    )
    assert resp.status_code == 200
    audio = model._prepare_audio.call_args.args[0]
    assert isinstance(audio, tuple)
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    # Decoded by soundfile, not resampled by the server — sample rate preserved.
    assert sr == 22050
    assert tensor.shape[-1] == 1600


def test_transcribe_rejects_undecodable_file():
    """Bytes that are neither WAV nor anything libsndfile reads → 400."""
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe",
        files={"file": ("mystery.mp3", b"\x00\x01 not audio \x02\x03", "audio/mpeg")},
    )
    assert resp.status_code == 400


def test_transcribe_passes_instruments(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin", "drums"]},
    )
    assert resp.status_code == 200
    assert model._transcribe_prepared.call_args.kwargs["instruments"] == [
        "violin",
        "drums",
    ]


def test_transcribe_midi_returns_bytes_with_headers(tmp_path):
    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/midi"
    assert resp.headers["content-disposition"] == 'attachment; filename="result.mid"'
    assert resp.content == FAKE_MIDI


def test_transcribe_midi_forwards_server_profile_setting(tmp_path):
    model = make_model()
    client = TestClient(create_app(model, profile=True))

    response = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )

    assert response.status_code == 200
    assert model.transcribe_to_midi.call_args.kwargs.get("profile") is True


def test_transcribe_midi_passes_tensor_and_instruments(tmp_path):
    import torch

    model = make_model()
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["violin", "drums"]},
    )
    assert resp.status_code == 200
    (audio,) = model.transcribe_to_midi.call_args.args
    assert model.transcribe_to_midi.call_args.kwargs["instruments"] == [
        "violin",
        "drums",
    ]
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16000
    assert tensor.shape[-1] == 1600


def test_transcribe_midi_rejects_invalid_wav():
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("garbage.wav", b"not a wav at all", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_midi_rejects_unknown_instrument(tmp_path):
    client = TestClient(create_app(make_model()), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
        data={"instruments": ["not_a_real_instrument"]},
    )
    assert resp.status_code == 400


def test_transcribe_midi_rejects_audio_over_duration_limit(tmp_path, monkeypatch):
    """Audio longer than the 15-minute cap is rejected with 413, before the
    model is ever touched."""
    monkeypatch.setattr(server_module, "_MAX_TRANSCRIBE_MIDI_DURATION_S", 0.05)
    model = make_model()
    client = TestClient(create_app(model), raise_server_exceptions=False)
    resp = client.post(
        "/transcribe/midi",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 413
    model.transcribe_to_midi.assert_not_called()


def _blocking_transcribe_model(first_reached: threading.Event, gate: threading.Event):
    """最初のprepared採譜だけ途中で止め、2 requestを確実に競合させるmock。"""
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    e0 = NoteEndEvent(end_time=0.5, start_event=s0)
    call_lock = threading.Lock()
    calls = [0]

    def side_effect(*args, **kwargs):
        with call_lock:
            calls[0] += 1
            first = calls[0] == 1
        yield s0
        if first:
            first_reached.set()
            assert gate.wait(timeout=10), "gate never opened"
        yield e0

    model = create_autospec(TranscriptionModel, instance=True)
    model._device = "cpu"
    model._prepare_audio.side_effect = lambda audio: audio[0]
    model._transcribe_prepared.side_effect = side_effect
    model.events_to_midi_bytes.return_value = FAKE_MIDI
    model._detect_beat_grid_prepared.return_value = None
    _bind_base_audio_entrypoints(model)
    return model, s0


def _blocking_public_adapter(
    first_reached: threading.Event,
    gate: threading.Event,
    closed: threading.Event,
):
    start = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    end = NoteEndEvent(end_time=0.5, start_event=start)

    class BlockingIterator:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            if self.index == 1:
                return start
            if self.index == 2:
                first_reached.set()
                assert gate.wait(timeout=10), "gate never opened"
                return end
            raise StopIteration

        def close(self):
            closed.set()

    class PublicOnlyAdapter:
        def __init__(self):
            self.calls = 0
            self.call_lock = threading.Lock()

        def transcribe(self, _audio, **_kwargs):
            with self.call_lock:
                self.calls += 1
                first = self.calls == 1
            return BlockingIterator() if first else iter((start, end))

        @staticmethod
        def detect_beat_grid_for(_audio, _mode="best-effort"):
            return None

        @staticmethod
        def events_to_midi_bytes(_events, beat_grid=None):
            assert beat_grid is None
            return FAKE_MIDI

    return PublicOnlyAdapter(), start


def _post_transcribe(app, payload, client_id, out, name):
    # A fresh TestClient per thread (httpx.Client isn't for concurrent use); the
    # underlying app — and its lock — is shared, which is what we're exercising.
    client = TestClient(app)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
        headers={"X-Client-Id": client_id},
    )
    out[name] = _parse_sse(resp.text)


def test_concurrent_different_clients_do_not_preempt(tmp_path):
    """The reported bug: a second window (different client id) must NOT stop the
    transcription already in progress. It also must not wait around for the
    lock: it gets an immediate 503 (so a caller retrying against another
    machine, e.g. behind Traefik, doesn't sit on the connection)."""
    first_reached = threading.Event()
    gate = threading.Event()
    model, s0 = _blocking_transcribe_model(first_reached, gate)
    app = create_app(model)
    payload = _wav_bytes(tmp_path)
    out: dict[str, list] = {}

    a = threading.Thread(
        target=_post_transcribe, args=(app, payload, "tab-A", out, "A")
    )
    a.start()
    assert first_reached.wait(timeout=5)  # A holds the lock, mid-stream

    # B, a different client, is refused immediately rather than queued.
    client_b = TestClient(app)
    started = time.monotonic()
    resp_b = client_b.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
        headers={"X-Client-Id": "tab-B"},
    )
    assert time.monotonic() - started < 2.0  # no ~60s (or even multi-second) wait
    assert resp_b.status_code == 503

    gate.set()
    a.join(timeout=10)

    # A ran to completion (ends with the assembled MIDI event) — not preempted.
    assert out["A"][0] == event_to_dict(s0)
    assert out["A"][-1] == {
        "type": "transcription_complete",
        "data": base64.b64encode(FAKE_MIDI).decode("ascii"),
        "beat_grid": None,
    }

    # Once A has released the lock, B's retry (as the frontend would send)
    # succeeds normally.
    resp_b_retry = client_b.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
        headers={"X-Client-Id": "tab-B"},
    )
    assert _parse_sse(resp_b_retry.text)[-1]["type"] == "transcription_complete"
    assert model._transcribe_prepared.call_count == 2


def test_concurrent_same_client_preempts(tmp_path):
    """A resubmit from the SAME client id still preempts the in-flight run so a
    stale stream stops instead of finishing."""
    first_reached = threading.Event()
    gate = threading.Event()
    model, s0 = _blocking_transcribe_model(first_reached, gate)
    app = create_app(model)
    payload = _wav_bytes(tmp_path)
    out: dict[str, list] = {}

    a = threading.Thread(
        target=_post_transcribe, args=(app, payload, "same-tab", out, "A")
    )
    a.start()
    assert first_reached.wait(timeout=5)

    b = threading.Thread(
        target=_post_transcribe, args=(app, payload, "same-tab", out, "B")
    )
    b.start()
    # Let B reach the lock and signal A's cancel (same id → preempt) before A
    # resumes past the gate.
    time.sleep(0.5)
    gate.set()
    a.join(timeout=10)
    b.join(timeout=10)

    # A was preempted: it streamed its first note but never the trailing MIDI.
    assert out["A"] == [event_to_dict(s0)]
    # B (the resubmit) ran to completion.
    assert out["B"][-1]["type"] == "transcription_complete"
    assert model._transcribe_prepared.call_count == 2


def test_same_client_preemption_closes_a_public_transcription_iterator(tmp_path):
    first_reached = threading.Event()
    gate = threading.Event()
    closed = threading.Event()
    model, _start = _blocking_public_adapter(first_reached, gate, closed)
    app = create_app(model)
    payload = _wav_bytes(tmp_path)
    out = {}

    first = threading.Thread(
        target=_post_transcribe,
        args=(app, payload, "same-tab", out, "first"),
    )
    first.start()
    assert first_reached.wait(timeout=5)

    replacement = threading.Thread(
        target=_post_transcribe,
        args=(app, payload, "same-tab", out, "replacement"),
    )
    replacement.start()
    time.sleep(0.5)
    gate.set()
    first.join(timeout=10)
    replacement.join(timeout=10)

    assert closed.is_set()
    assert out["replacement"][-1]["type"] == "transcription_complete"


# ---- /sheets ------------------------------------------------------------
# MuseScore is never invoked here: `write_sheets` is replaced with a stub that
# writes the same shape of output, so these run on machines without it.

FAKE_PDF = b"%PDF-1.4 fake"


def _fake_write_sheets(midi_bytes, out_dir, musescore=None):
    """Stand-in for write_sheets: the real file set, with dummy contents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in (
        ("score.mid", midi_bytes),
        ("score.musicxml", b"<score-partwise/>"),
        ("full_score.pdf", FAKE_PDF),
        ("01_electric_guitar.pdf", FAKE_PDF),
        ("01_electric_guitar_tab.pdf", FAKE_PDF),
    ):
        path = out_dir / name
        path.write_bytes(payload)
        written.append(path)
    return written


def _sheets_client(monkeypatch, write_sheets=_fake_write_sheets, **kwargs):
    monkeypatch.setattr(server_module, "write_sheets", write_sheets)
    return TestClient(create_app(make_model()), **kwargs)


def test_sheets_returns_every_engraved_file(monkeypatch):
    client = _sheets_client(monkeypatch)
    resp = client.post("/sheets", files={"midi": ("in.mid", FAKE_MIDI, "audio/midi")})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["content-disposition"] == 'attachment; filename="sheets.zip"'

    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        # Flattened to bare filenames, in the order write_sheets emits them —
        # the frontend lists them as they come out of the archive.
        assert archive.namelist() == [
            "score.mid",
            "score.musicxml",
            "full_score.pdf",
            "01_electric_guitar.pdf",
            "01_electric_guitar_tab.pdf",
        ]
        assert archive.read("score.mid") == FAKE_MIDI
        assert archive.read("full_score.pdf") == FAKE_PDF


def test_sheets_zip_is_stored_not_deflated(monkeypatch):
    """The browser unpacks this archive itself; every member must be stored."""
    client = _sheets_client(monkeypatch)
    resp = client.post("/sheets", files={"midi": ("in.mid", FAKE_MIDI, "audio/midi")})
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        assert [i.compress_type for i in archive.infolist()] == [
            zipfile.ZIP_STORED
        ] * len(archive.infolist())


def test_sheets_passes_the_uploaded_midi_through(monkeypatch):
    seen = {}

    def spy(midi_bytes, out_dir, musescore=None):
        seen["midi"] = midi_bytes
        return _fake_write_sheets(midi_bytes, out_dir)

    client = _sheets_client(monkeypatch, write_sheets=spy)
    resp = client.post("/sheets", files={"midi": ("in.mid", b"MThd...", "audio/midi")})
    assert resp.status_code == 200
    assert seen["midi"] == b"MThd..."


def test_sheets_leaves_nothing_on_disk(monkeypatch):
    """The scratch directory write_sheets rendered into is gone afterwards."""
    dirs = []

    def spy(midi_bytes, out_dir, musescore=None):
        dirs.append(out_dir)
        return _fake_write_sheets(midi_bytes, out_dir)

    client = _sheets_client(monkeypatch, write_sheets=spy)
    assert (
        client.post(
            "/sheets", files={"midi": ("in.mid", FAKE_MIDI, "audio/midi")}
        ).status_code
        == 200
    )
    assert not dirs[0].exists()


def test_sheets_without_musescore_is_503(monkeypatch):
    """A server with no MuseScore is a deployment problem, not a bad request —
    and the install hint has to reach the client."""

    def missing(midi_bytes, out_dir, musescore=None):
        raise MuseScoreNotFoundError("MuseScore was not found. Downloads: ...")

    client = _sheets_client(monkeypatch, write_sheets=missing)
    resp = client.post("/sheets", files={"midi": ("in.mid", FAKE_MIDI, "audio/midi")})
    assert resp.status_code == 503
    assert "MuseScore was not found" in resp.json()["detail"]


def test_sheets_reports_a_musescore_failure(monkeypatch):
    def broken(midi_bytes, out_dir, musescore=None):
        raise MuseScoreError("MuseScore failed to import the MIDI file.")

    client = _sheets_client(
        monkeypatch, write_sheets=broken, raise_server_exceptions=False
    )
    resp = client.post("/sheets", files={"midi": ("in.mid", b"not midi", "audio/midi")})
    assert resp.status_code == 500
    assert "import the MIDI file" in resp.json()["detail"]


def test_sheets_missing_file(monkeypatch):
    client = _sheets_client(monkeypatch)
    assert client.post("/sheets").status_code == 422


def test_sheets_engraves_concurrently(monkeypatch):
    """Engraving is not serialized: it costs a handful of short MuseScore
    processes and none of the GPU, so a second caller runs straight away
    instead of waiting for — or being refused because of — the first."""
    first_reached = threading.Event()
    gate = threading.Event()
    calls = []

    def blocks_the_first_caller(midi_bytes, out_dir, musescore=None):
        calls.append(1)
        if len(calls) == 1:
            first_reached.set()
            assert gate.wait(timeout=10), "gate never opened"
        return _fake_write_sheets(midi_bytes, out_dir)

    monkeypatch.setattr(server_module, "write_sheets", blocks_the_first_caller)
    app = create_app(make_model())
    files = {"midi": ("in.mid", FAKE_MIDI, "audio/midi")}
    out = {}

    def post_first():
        out["first"] = TestClient(app).post("/sheets", files=files).status_code

    a = threading.Thread(target=post_first)
    a.start()
    assert first_reached.wait(timeout=5)

    # The first engrave is still inside MuseScore at this point.
    started = time.monotonic()
    assert TestClient(app).post("/sheets", files=files).status_code == 200
    assert time.monotonic() - started < 5.0, "second engrave waited on the first"

    gate.set()
    a.join(timeout=10)
    assert out["first"] == 200
