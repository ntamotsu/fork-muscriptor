"""実weightやacceleratorを使わないbenchmark CLIの境界テスト。"""

import json
import os
import types
from pathlib import Path

import pytest
import torch
import typer
from typer.testing import CliRunner

import muscriptor.benchmark_cli as benchmark_cli
from muscriptor.benchmark import (
    BenchmarkEnvironment,
    BenchmarkProtocol,
    BenchmarkReport,
    BenchmarkSample,
    BenchmarkWorkload,
    EventCounts,
    GateStatus,
    GoldenPolicy,
    compare_reports,
)
from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.generation_telemetry import ChunkGenerationStats


class _FakeModel:
    load_kwargs: dict | None = None
    transcribe_kwargs: dict | None = None

    @classmethod
    def load_model(cls, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def transcribe(self, audio, **kwargs):
        type(self).transcribe_kwargs = kwargs
        wav, sample_rate = audio
        assert wav.shape == (1, 16_000)
        assert sample_rate == 16_000
        start = NoteStartEvent(60, 0.1, 7, "piano")
        yield ProgressEvent(0, 1)
        yield start
        yield NoteEndEvent(0.4, start)
        yield ProgressEvent(1, 1)


class _MutatingModel(_FakeModel):
    audio_path: Path

    def transcribe(self, audio, **kwargs):
        type(self).audio_path.write_bytes(b"changed while running")
        yield from super().transcribe(audio, **kwargs)


class _NoEosModel(_FakeModel):
    def transcribe(self, _audio, **kwargs):
        type(self).transcribe_kwargs = kwargs
        if False:
            yield
        raise RuntimeError("chunk 0 did not emit EOS within 2000 tokens")


class _TelemetryFakeModel(_FakeModel):
    prepared_kwargs: list[dict] = []

    def transcribe(self, _audio, **_kwargs):
        pytest.fail("telemetry must use canonical prepared audio directly")

    def _transcribe_prepared(self, wav, **kwargs):
        assert wav.shape == (1, 16_000)
        observer = kwargs.pop("_generation_observer")
        type(self).prepared_kwargs.append(kwargs)
        observer(
            ChunkGenerationStats(
                chunk_index=0,
                seek_time_us=0,
                prompt_tokens=0,
                observed_rows=3,
                generated_rows=3,
                eos_step=3,
                max_gen_len=2000,
                hit_generation_limit=False,
            )
        )
        start = NoteStartEvent(60, 0.1, 7, "piano")
        yield ProgressEvent(0, 1)
        yield start
        yield NoteEndEvent(0.4, start)
        yield ProgressEvent(1, 1)

    def _generate_token_stream(self, *_args, **_kwargs):
        raise AssertionError("the fake prepared seam does not decode tokens")


def _run_args(audio: Path, weights: Path, output: Path) -> list[str]:
    return [
        "run",
        str(audio),
        "--model",
        str(weights),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--environment",
        "test-cpu",
        "--warmup-runs",
        "1",
        "--runs",
        "2",
    ]


def _hf_snapshot(tmp_path: Path):
    """HF cacheと同じsnapshotからblobへのsymlink構造を作る。"""
    cache = tmp_path / "models--owner--muscriptor"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    weights_blob = blobs / "weights-hash"
    config_blob = blobs / "config-hash"
    weights_blob.write_bytes(b"weight bytes")
    config_blob.write_text(
        json.dumps({"dim": 32, "num_heads": 2, "num_layers": 1, "card": 1393})
    )
    snapshot_weights = snapshot / "model.safetensors"
    snapshot_config = snapshot / "config.json"
    snapshot_weights.symlink_to("../../blobs/weights-hash")
    snapshot_config.symlink_to("../../blobs/config-hash")
    return cache, snapshot_weights, snapshot_config, weights_blob, config_blob


def test_run_requires_an_explicit_device(tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "run",
            str(audio),
            "--model",
            str(weights),
            "--output",
            str(tmp_path / "report.json"),
            "--environment",
            "test-cpu",
            "--dtype",
            "float32",
        ],
    )

    assert result.exit_code != 0
    assert "--device" in result.output


def test_run_requires_an_explicit_dtype(tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "run",
            str(audio),
            "--model",
            str(weights),
            "--output",
            str(tmp_path / "report.json"),
            "--environment",
            "test-cpu",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "--dtype" in result.output


def test_run_writes_a_report_without_real_inference(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    config = tmp_path / "config.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    config.write_text(
        json.dumps({"dim": 64, "num_heads": 4, "num_layers": 2, "card": 1393})
    )
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel, raising=False)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_source_metadata",
        lambda: {"git_commit": "deadbeef", "git_dirty": False},
        raising=False,
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, weights, output),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert payload["protocol"]["generation_telemetry"] == "none"
    assert all(
        sample["chunk_generation_stats"] is None for sample in payload["samples"]
    )
    assert payload["environment"]["name"] == "test-cpu"
    assert payload["source"]["decoding_implementation"] == "scalar-v1"
    assert payload["workload"]["parameters"]["audio"]["num_samples"] == 16_000
    assert payload["workload"]["parameters"]["model"]["config"] == {
        "effective": {"card": 1393, "dim": 64, "num_heads": 4, "num_layers": 2},
        "file_sha256": benchmark_cli._sha256_file(config),
        "source": "adjacent-config.json",
    }
    assert payload["summary"]["output_stable"] is True
    assert _FakeModel.load_kwargs == {
        "weights_path": weights,
        "device": torch.device("cpu"),
        "dtype": "float32",
    }
    assert _FakeModel.transcribe_kwargs == {
        "use_sampling": False,
        "temperature": 1.0,
        "cfg_coef": 1.0,
        "instruments": None,
        "batch_size": 1,
        "no_eos_is_ok": False,
        "beam_size": 1,
        "prelude_forcing": True,
        "profile": False,
        "log_progress": False,
    }


def test_run_enables_speculative_decoding_without_changing_semantic_workload(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_require_base_speculative_model",
        lambda _model: None,
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, output), "--speculative-decoding"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert _FakeModel.transcribe_kwargs == {
        **benchmark_cli._TRANSCRIBE_PARAMETERS,
        "speculative_decoding": True,
    }
    assert payload["workload"]["parameters"]["transcribe"] == (
        benchmark_cli._TRANSCRIBE_PARAMETERS
    )
    assert payload["protocol"]["generation_telemetry"] == "none"
    assert payload["source"]["decoding_implementation"] == "history-ngram-v1"


def test_speculative_benchmark_help_warns_that_some_audio_can_be_slower():
    result = CliRunner().invoke(benchmark_cli.app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "slower when" in " ".join(result.output.split())


def test_run_rejects_speculative_decoding_outside_the_loaded_model_matrix(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    model = object.__new__(benchmark_cli.TranscriptionModel)
    model._device = torch.device("cpu")
    model._model = object()
    monkeypatch.setattr(
        benchmark_cli.TranscriptionModel,
        "load_model",
        classmethod(lambda _cls, **_kwargs: model),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, output), "--speculative-decoding"],
    )

    assert result.exit_code != 0
    assert "--speculative-decoding" in result.output
    assert "large float16 model on MPS" in result.output
    assert not output.exists()


def test_scalar_and_speculative_reports_remain_comparison_compatible(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    scalar_output = tmp_path / "scalar.json"
    speculative_output = tmp_path / "speculative.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_require_base_speculative_model",
        lambda _model: None,
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    scalar_result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, weights, scalar_output),
    )
    speculative_result = CliRunner().invoke(
        benchmark_cli.app,
        [
            *_run_args(audio, weights, speculative_output),
            "--speculative-decoding",
        ],
    )

    assert scalar_result.exit_code == 0, scalar_result.output
    assert speculative_result.exit_code == 0, speculative_result.output
    scalar = BenchmarkReport.from_json(scalar_output.read_text())
    speculative = BenchmarkReport.from_json(speculative_output.read_text())
    comparison = compare_reports(
        speculative,
        scalar,
        GoldenPolicy(max_median_regression_percent=1_000_000),
    )
    assert speculative.workload == scalar.workload
    assert speculative.protocol == scalar.protocol
    assert comparison.status is GateStatus.PASS


def test_run_forwards_canonical_instruments_and_allow_no_eos_to_the_workload(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            *_run_args(audio, weights, output),
            "--instruments",
            "drums, chromatic_percussion, orchestra_hit",
            "--allow-no-eos",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert (
        _FakeModel.transcribe_kwargs,
        payload["workload"]["parameters"]["transcribe"],
    ) == (
        {
            **benchmark_cli._TRANSCRIBE_PARAMETERS,
            "instruments": [
                "drums",
                "chromatic_percussion",
                "orchestra_hit",
            ],
            "no_eos_is_ok": True,
        },
    ) * 2


def test_generation_telemetry_rejects_fake_or_overridden_model_implementations(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    output.write_text("keep existing report")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            *_run_args(audio, weights, output),
            "--generation-telemetry",
            "--force",
        ],
    )

    assert result.exit_code != 0
    assert "generation-telemetry" in result.output
    assert "TranscriptionModel" in result.output
    assert output.read_text() == "keep existing report"


def test_strict_eos_failure_keeps_an_existing_force_target(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    output.write_text("keep existing report")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _NoEosModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, output), "--force"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert _NoEosModel.transcribe_kwargs["no_eos_is_ok"] is False
    assert output.read_text() == "keep existing report"


@pytest.mark.parametrize(
    ("override_scope", "method_name"),
    [
        (scope, method_name)
        for scope in ("class", "instance")
        for method_name in (
            "transcribe",
            "_transcribe_prepared",
            "_generate_token_stream",
        )
    ],
)
@pytest.mark.parametrize(
    "validator_name",
    [
        "_require_base_generation_telemetry_model",
        "_require_base_speculative_model",
    ],
)
def test_private_benchmark_paths_detect_class_and_instance_method_overrides(
    override_scope,
    method_name,
    validator_name,
):
    def override(self, *_args, **_kwargs):
        return None

    if override_scope == "class":
        model_type = type(
            "OverriddenTranscriptionModel",
            (benchmark_cli.TranscriptionModel,),
            {method_name: override},
        )
        model = object.__new__(model_type)
    else:
        model = object.__new__(benchmark_cli.TranscriptionModel)
        setattr(model, method_name, types.MethodType(override, model))

    with pytest.raises(typer.BadParameter, match=method_name):
        getattr(benchmark_cli, validator_name)(model)


@pytest.mark.parametrize("override_scope", ["class", "instance"])
@pytest.mark.parametrize(
    "method_name",
    ["_validate_speculative_request", "_validate_speculative_ngram"],
)
def test_speculative_benchmark_detects_validator_overrides(
    override_scope,
    method_name,
):
    def override(self, **_kwargs):
        return None

    if override_scope == "class":
        model_type = type(
            "OverriddenTranscriptionModel",
            (benchmark_cli.TranscriptionModel,),
            {method_name: override},
        )
        model = object.__new__(model_type)
    else:
        model = object.__new__(benchmark_cli.TranscriptionModel)
        setattr(model, method_name, types.MethodType(override, model))

    with pytest.raises(typer.BadParameter, match=method_name):
        benchmark_cli._require_base_speculative_model(model)


def test_generation_telemetry_uses_prepared_audio_and_records_v2_chunk_stats(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    _TelemetryFakeModel.prepared_kwargs = []
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _TelemetryFakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "_require_base_generation_telemetry_model",
        lambda _model: None,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            *_run_args(audio, weights, output),
            "--generation-telemetry",
            "--allow-no-eos",
            "--instruments",
            "drums,chromatic_percussion,orchestra_hit",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    expected_parameters = {
        **benchmark_cli._TRANSCRIBE_PARAMETERS,
        "instruments": ["drums", "chromatic_percussion", "orchestra_hit"],
        "no_eos_is_ok": True,
    }
    assert (
        payload["protocol"]["generation_telemetry"],
        [sample["chunk_generation_stats"] for sample in payload["samples"]],
        _TelemetryFakeModel.prepared_kwargs,
    ) == (
        "selected-output-v1",
        [
            [
                {
                    "chunk_index": 0,
                    "seek_time_us": 0,
                    "prompt_tokens": 0,
                    "observed_rows": 3,
                    "generated_rows": 3,
                    "eos_step": 3,
                    "max_gen_len": 2000,
                    "hit_generation_limit": False,
                }
            ]
        ]
        * 2,
        [expected_parameters] * 3,
    )


def test_speculative_generation_telemetry_uses_the_private_prepared_opt_in(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    weights.write_bytes(b"weight bytes")
    _TelemetryFakeModel.prepared_kwargs = []
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _TelemetryFakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "_require_base_generation_telemetry_model",
        lambda _model: None,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_require_base_speculative_model",
        lambda _model: None,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            *_run_args(audio, weights, output),
            "--generation-telemetry",
            "--speculative-decoding",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert (
        _TelemetryFakeModel.prepared_kwargs
        == [{**benchmark_cli._TRANSCRIBE_PARAMETERS, "_speculative_ngram": True}] * 3
    )
    assert payload["workload"]["parameters"]["transcribe"] == (
        benchmark_cli._TRANSCRIBE_PARAMETERS
    )
    assert payload["protocol"]["generation_telemetry"] == "selected-output-v1"
    assert payload["source"]["decoding_implementation"] == "history-ngram-v1"


def test_run_preserves_hf_snapshot_paths_for_adjacent_symlinked_config(
    monkeypatch, tmp_path
):
    cache, snapshot_weights, _snapshot_config, _weights_blob, config_blob = (
        _hf_snapshot(tmp_path)
    )
    (cache / "snapshots" / "other").mkdir()
    audio = tmp_path / "audio.wav"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})
    relative_weights = (
        cache.relative_to(tmp_path)
        / "snapshots"
        / "other"
        / ".."
        / "revision"
        / "model.safetensors"
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, relative_weights, output),
    )

    payload = json.loads(output.read_text()) if output.exists() else {}
    assert (
        result.exit_code,
        payload.get("workload", {})
        .get("parameters", {})
        .get("model", {})
        .get("config"),
        _FakeModel.load_kwargs,
    ) == (
        0,
        {
            "effective": {
                "card": 1393,
                "dim": 32,
                "num_heads": 2,
                "num_layers": 1,
            },
            "file_sha256": benchmark_cli._sha256_file(config_blob),
            "source": "adjacent-config.json",
        },
        {
            "weights_path": snapshot_weights,
            "device": torch.device("cpu"),
            "dtype": "float32",
        },
    )


def test_run_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    output.write_text("keep me")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel, raising=False)

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, weights, output),
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text() == "keep me"


def test_report_round_trip_uses_utf8_for_non_ascii_labels(monkeypatch, tmp_path):
    audio = tmp_path / "楽曲.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})
    args = _run_args(audio, weights, output)
    args[args.index("--environment") + 1] = "日本語-M4"
    args.extend(["--audio-id", "長尺の楽曲"])

    run_result = CliRunner().invoke(benchmark_cli.app, args)
    compare_result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "compare",
            str(output),
            str(output),
            "--max-regression-percent",
            "0",
        ],
    )

    assert run_result.exit_code == 0, run_result.output
    assert compare_result.exit_code == 0, compare_result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["environment"]["name"] == "日本語-M4"
    assert payload["workload"]["parameters"]["audio"]["id"] == "長尺の楽曲"


@pytest.mark.parametrize("output_source", ["audio", "weights", "config"])
def test_force_never_overwrites_an_input_file(output_source, monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"dim": 64, "num_heads": 4, "num_layers": 2, "card": 1393})
    )
    output = {"audio": audio, "weights": weights, "config": config}[output_source]
    original = output.read_bytes()

    def must_not_load(_path):
        pytest.fail("input/output collision must be rejected before loading")

    monkeypatch.setattr(benchmark_cli, "load_audio", must_not_load)
    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, output), "--force"],
    )

    assert result.exit_code != 0
    assert "must differ" in result.output
    assert output.read_bytes() == original


@pytest.mark.parametrize("output_source", ["weights-target", "config-target"])
def test_force_never_overwrites_an_hf_snapshot_symlink_target(
    output_source, monkeypatch, tmp_path
):
    _cache, snapshot_weights, _snapshot_config, weights_blob, config_blob = (
        _hf_snapshot(tmp_path)
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = {
        "weights-target": weights_blob,
        "config-target": config_blob,
    }[output_source]
    original = output.read_bytes()

    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: pytest.fail("resolved input target must fail before loading"),
    )
    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, snapshot_weights, output), "--force"],
    )

    assert (
        result.exit_code != 0,
        "must differ" in result.output,
        output.read_bytes(),
    ) == (True, True, original)


def test_output_never_occupies_a_missing_companion_config_path(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: pytest.fail("reserved config path must fail before loading"),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, config), "--force"],
    )

    assert result.exit_code != 0
    assert "must differ" in result.output
    assert not config.exists()


@pytest.mark.parametrize("device", ["meta", "xpu:0", "cpu:1", "mps:0", "cuda"])
def test_run_rejects_devices_outside_the_benchmark_matrix(
    device, monkeypatch, tmp_path
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    args = _run_args(audio, weights, output)
    args[args.index("--device") + 1] = device
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: pytest.fail("invalid device must fail before audio loading"),
    )

    result = CliRunner().invoke(benchmark_cli.app, args)

    assert result.exit_code != 0
    assert "device" in result.output.lower()


def test_cpu_environment_metadata_does_not_probe_gpu_backends(monkeypatch):
    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("CPU metadata must not probe CUDA, cuDNN, or MPS")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_probe)
    monkeypatch.setattr(torch.backends.cudnn, "version", unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)

    metadata = benchmark_cli._environment_metadata(torch.device("cpu"), "float32")

    device = metadata["device"]
    assert device["type"] == "cpu"
    assert device["index"] is None
    assert isinstance(device["name"], str) and device["name"]
    assert device["transformer_dtype"] == "float32"
    assert device["cuda_runtime"] is None
    assert device["cudnn_version"] is None


def test_device_availability_errors_are_reported_without_initializing_audio(
    monkeypatch,
):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(typer.BadParameter, match="MPS is not available"):
        benchmark_cli._parse_device("mps")
    with pytest.raises(typer.BadParameter, match="CUDA is not available"):
        benchmark_cli._parse_device("cuda:0")


def test_run_rejects_a_symbolic_link_output_before_loading(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    target = tmp_path / "target.json"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    target.write_text("keep")
    output.symlink_to(target)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: pytest.fail("symlink must be rejected before loading"),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        [*_run_args(audio, weights, output), "--force"],
    )

    assert result.exit_code != 0
    assert "symbolic link" in result.output
    assert target.read_text() == "keep"


def test_compare_rejects_an_oversized_report_before_reading(tmp_path):
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as file:
        file.seek(benchmark_cli._MAX_REPORT_BYTES)
        file.write(b"x")

    with pytest.raises(ValueError, match="exceeds"):
        benchmark_cli._read_report(oversized)


def test_run_rejects_an_input_changed_during_measurement(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"original audio")
    weights.write_bytes(b"weights")
    _MutatingModel.audio_path = audio
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _MutatingModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, weights, output),
    )

    assert result.exit_code != 0
    assert "changed while the benchmark ran" in result.output
    assert not output.exists()


@pytest.mark.parametrize("retargeted_input", ["weights", "config"])
def test_run_rejects_an_hf_snapshot_symlink_retargeted_during_measurement(
    retargeted_input, monkeypatch, tmp_path
):
    cache, snapshot_weights, snapshot_config, weights_blob, config_blob = _hf_snapshot(
        tmp_path
    )
    blobs = cache / "blobs"
    if retargeted_input == "weights":
        symlink_path = snapshot_weights
        original_blob = weights_blob
        new_blob = blobs / "weights-new"
        new_target = "../../blobs/weights-new"
    else:
        symlink_path = snapshot_config
        original_blob = config_blob
        new_blob = blobs / "config-new"
        new_target = "../../blobs/config-new"
    new_blob.write_bytes(original_blob.read_bytes())
    original_metadata = original_blob.stat()
    new_metadata = new_blob.stat()
    os.utime(
        new_blob,
        ns=(new_metadata.st_atime_ns, original_metadata.st_mtime_ns),
    )

    class RetargetingModel(_FakeModel):
        def transcribe(self, audio, **kwargs):
            symlink_path.unlink()
            symlink_path.symlink_to(new_target)
            yield from super().transcribe(audio, **kwargs)

    audio = tmp_path / "audio.wav"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", RetargetingModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )
    monkeypatch.setattr(benchmark_cli, "_source_metadata", lambda: {})

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, snapshot_weights, output),
    )

    assert (
        result.exit_code != 0,
        "changed while the benchmark ran" in result.output,
        output.exists(),
    ) == (True, True, False)


def test_run_does_not_write_a_report_if_strict_synchronization_fails(
    monkeypatch, tmp_path
):
    audio = tmp_path / "audio.wav"
    weights = tmp_path / "model.safetensors"
    output = tmp_path / "report.json"
    audio.write_bytes(b"audio")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(benchmark_cli, "TranscriptionModel", _FakeModel)
    monkeypatch.setattr(
        benchmark_cli,
        "load_audio",
        lambda _path: torch.zeros(1, 16_000),
    )

    def fail_synchronization(_device):
        raise RuntimeError("simulated synchronization failure")

    monkeypatch.setattr(
        benchmark_cli,
        "_strict_synchronize",
        fail_synchronization,
        raising=False,
    )

    result = CliRunner().invoke(
        benchmark_cli.app,
        _run_args(audio, weights, output),
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "synchronization failure" in str(result.exception)
    assert not output.exists()


def test_atomic_write_keeps_the_old_report_if_replace_fails(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text("old report")

    def fail_replace(*_args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(benchmark_cli.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated storage failure"):
        benchmark_cli._atomic_write_text(output, "new report", force=True)

    assert output.read_text() == "old report"
    assert list(tmp_path.iterdir()) == [output]


def test_default_atomic_publish_failure_is_a_cli_error(monkeypatch, tmp_path):
    output = tmp_path / "report.json"

    def fail_link(*_args):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(benchmark_cli.os, "link", fail_link)

    with pytest.raises(typer.BadParameter, match="Cannot atomically publish"):
        benchmark_cli._atomic_write_text(output, "report", force=False)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def _report(
    wall_time_ns: int,
    *,
    environment: str = "same-environment",
) -> BenchmarkReport:
    return BenchmarkReport(
        workload=BenchmarkWorkload("same-workload"),
        environment=BenchmarkEnvironment(environment),
        protocol=BenchmarkProtocol(warmup_runs=1, measured_runs=1),
        samples=(
            BenchmarkSample(
                index=0,
                wall_time_ns=wall_time_ns,
                first_progress_ns=1,
                first_note_ns=2,
                progress_milestones=(),
                event_counts=EventCounts(note_start=1, note_end=1, progress=0),
                stream_digest="same-stream",
                note_digest="same-notes",
            ),
        ),
    )


def test_compare_prints_machine_readable_pass_result(tmp_path):
    golden = tmp_path / "golden.json"
    candidate = tmp_path / "candidate.json"
    golden.write_text(_report(100).to_json())
    candidate.write_text(_report(102).to_json())

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "compare",
            str(candidate),
            str(golden),
            "--max-regression-percent",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "median_ratio": 1.02,
        "reasons": [],
        "status": "pass",
    }


def test_compare_exits_one_on_a_timing_regression(tmp_path):
    golden = tmp_path / "golden.json"
    candidate = tmp_path / "candidate.json"
    golden.write_text(_report(100).to_json())
    candidate.write_text(_report(103).to_json())

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "compare",
            str(candidate),
            str(golden),
            "--max-regression-percent",
            "2",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "fail"


def test_compare_uses_a_distinct_exit_code_for_incompatible_reports(tmp_path):
    golden = tmp_path / "golden.json"
    candidate = tmp_path / "candidate.json"
    golden.write_text(_report(100, environment="m4").to_json())
    candidate.write_text(_report(100, environment="t4").to_json())

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "compare",
            str(candidate),
            str(golden),
            "--max-regression-percent",
            "0",
        ],
    )

    assert result.exit_code == 3
    assert json.loads(result.stdout)["status"] == "incompatible"


@pytest.mark.parametrize("threshold", ["nan", "inf"])
def test_compare_rejects_non_finite_regression_thresholds(threshold, tmp_path):
    golden = tmp_path / "golden.json"
    candidate = tmp_path / "candidate.json"
    golden.write_text(_report(100).to_json())
    candidate.write_text(_report(100).to_json())

    result = CliRunner().invoke(
        benchmark_cli.app,
        [
            "compare",
            str(candidate),
            str(golden),
            "--max-regression-percent",
            threshold,
        ],
    )

    assert result.exit_code != 0
    assert "finite" in result.output
