"""再現可能な推論ベンチマークを実行・比較するCLI。"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import stat
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import torch
import typer

from muscriptor.benchmark import (
    BenchmarkEnvironment,
    BenchmarkProtocol,
    BenchmarkReport,
    BenchmarkWorkload,
    GateStatus,
    GoldenPolicy,
    compare_reports,
    run_benchmark,
)
from muscriptor.transcription_model import TranscriptionModel, _resolve_config
from muscriptor.tokenizer.mt3 import resolve_instrument_names
from muscriptor.utils.audio import load_audio


app = typer.Typer(
    add_completion=False,
    help="Run and compare reproducible muscriptor inference benchmarks.",
)


_TRANSCRIBE_PARAMETERS = {
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
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_BASE_TRANSCRIPTION_METHODS = {
    name: inspect.getattr_static(TranscriptionModel, name)
    for name in ("transcribe", "_transcribe_prepared", "_generate_token_stream")
}


@app.callback()
def _root() -> None:
    """ベンチマークの実行または既存レポートの比較を選ぶ。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(str(tuple(canonical.shape)).encode())
    digest.update(memoryview(canonical.numpy()).cast("B"))
    return digest.hexdigest()


def _absolute_lexical(path: Path) -> Path:
    """最終symlinkを解決せず、pathを絶対化してlexicalに正規化する。"""
    return Path(os.path.abspath(path))


def _file_snapshot(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise typer.BadParameter(f"Benchmark inputs must be regular files: {path}")
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _ensure_unchanged(path: Path, expected: tuple[int, int, int, int]) -> None:
    if _file_snapshot(path) != expected:
        raise typer.BadParameter(f"Input changed while the benchmark ran: {path}")


def _ensure_distinct_output(output: Path, *inputs: Path) -> None:
    if output.is_symlink():
        raise typer.BadParameter("Benchmark output must not be a symbolic link")
    resolved_output = output.resolve(strict=False)
    for input_path in inputs:
        same_path = resolved_output == input_path.resolve()
        same_file = (
            output.exists()
            and input_path.exists()
            and os.path.samefile(output, input_path)
        )
        if same_path or same_file:
            raise typer.BadParameter(
                "Benchmark output must differ from the audio, model, and config inputs"
            )


def _prepare_output(output: Path, *, force: bool) -> None:
    if output.exists() and not force:
        raise typer.BadParameter(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    Path(temporary).unlink()


def _atomic_write_text(output: Path, text: str, *, force: bool) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        if force:
            os.replace(temporary_path, output)
        else:
            try:
                os.link(temporary_path, output)
            except FileExistsError as error:
                raise typer.BadParameter(f"Output already exists: {output}") from error
            except OSError as error:
                raise typer.BadParameter(
                    f"Cannot atomically publish benchmark report: {error}"
                ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_report(path: Path) -> BenchmarkReport:
    if path.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError(f"Benchmark report exceeds {_MAX_REPORT_BYTES} bytes")
    return BenchmarkReport.from_json(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _source_metadata() -> dict[str, str | bool | None]:
    status = _git_output("status", "--porcelain")
    diff = _git_output("diff", "--binary", "HEAD", "--")
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
        ),
        "muscriptor_source_sha256": _source_tree_sha256(),
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return platform.platform()
    return platform.processor() or platform.machine()


def _environment_metadata(device: torch.device, dtype: str) -> dict:
    cuda_selected = device.type == "cuda"
    cuda_metadata = None
    if cuda_selected:
        cuda_metadata = {
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "model_autocast_dtype": "float16",
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": {
            "type": device.type,
            "index": device.index,
            "name": _device_name(device),
            "transformer_dtype": dtype,
            "cuda_runtime": torch.version.cuda if cuda_selected else None,
            "cudnn_version": (
                torch.backends.cudnn.version() if cuda_selected else None
            ),
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_compute": cuda_metadata,
    }


def _parse_device(value: str) -> torch.device:
    if value == "auto":
        raise typer.BadParameter(
            "Specify cpu, mps, or an indexed device such as cuda:0"
        )
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise typer.BadParameter(f"Invalid torch device: {value}") from error
    if device.type not in {"cpu", "mps", "cuda"}:
        raise typer.BadParameter(f"Unsupported benchmark device: {value}")
    if device.type in {"cpu", "mps"} and device.index is not None:
        raise typer.BadParameter(f"Unsupported benchmark device: {value}")
    if device.type == "cuda" and device.index is None:
        raise typer.BadParameter(
            "Unsupported benchmark device; use an explicit index such as cuda:0"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise typer.BadParameter("MPS is not available on this machine")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise typer.BadParameter("CUDA is not available on this machine")
        if device.index >= torch.cuda.device_count():
            raise typer.BadParameter(
                f"CUDA device index is out of range: {device.index}"
            )
    return device


def _strict_synchronize(device: torch.device) -> None:
    """計測境界では同期失敗を握り潰さず、report作成を中止する。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _require_base_generation_telemetry_model(model: object) -> None:
    overridden: list[str] = []
    for name, base_method in _BASE_TRANSCRIPTION_METHODS.items():
        method = getattr(model, name, None)
        if (
            getattr(method, "__self__", None) is not model
            or getattr(method, "__func__", None) is not base_method
        ):
            overridden.append(name)
    if overridden:
        raise typer.BadParameter(
            "--generation-telemetry requires base TranscriptionModel "
            "implementations for transcribe, _transcribe_prepared, and "
            f"_generate_token_stream; overridden or missing: {', '.join(overridden)}"
        )


@app.command("run")
def run_command(
    audio_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Audio file to decode before the timed region.",
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option(
            "--model",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Local model weights; remote aliases are intentionally rejected.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination for the JSON report."),
    ],
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Explicit torch device: cpu, mps, or an indexed CUDA device.",
        ),
    ],
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            help="Explicit transformer dtype: float32, float16, or bfloat16.",
        ),
    ],
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            help="Stable label for this machine and software environment.",
        ),
    ],
    model_label: Annotated[
        str | None,
        typer.Option("--model-label", help="Human-readable model identifier."),
    ] = None,
    audio_id: Annotated[
        str | None,
        typer.Option("--audio-id", help="Stable, path-independent audio identifier."),
    ] = None,
    warmup_runs: Annotated[
        int,
        typer.Option("--warmup-runs", min=0, help="Unrecorded warmup runs."),
    ] = 1,
    runs: Annotated[
        int,
        typer.Option("--runs", min=1, help="Measured runs."),
    ] = 5,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing report."),
    ] = False,
    instruments: Annotated[
        str | None,
        typer.Option(
            "--instruments",
            help="Comma-separated expected instrument group names.",
        ),
    ] = None,
    allow_no_eos: Annotated[
        bool,
        typer.Option(
            "--allow-no-eos",
            help="Record runs whose chunks reach the generation limit without EOS.",
        ),
    ] = False,
    generation_telemetry: Annotated[
        bool,
        typer.Option(
            "--generation-telemetry",
            help="Record per-chunk selected-output generation statistics.",
        ),
    ] = False,
) -> None:
    """モデルを一度だけ読み込み、転写streamを最後まで消費する時間を測る。"""
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise typer.BadParameter(
            "dtype must be one of: float32, float16, bfloat16", param_hint="--dtype"
        )

    instrument_names: list[str] | None = None
    if instruments is not None:
        try:
            instrument_names = resolve_instrument_names(
                token for token in instruments.split(",") if token.strip()
            )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--instruments") from error
    transcribe_parameters = {
        **_TRANSCRIBE_PARAMETERS,
        "instruments": instrument_names,
        "no_eos_is_ok": allow_no_eos,
    }

    torch_device = _parse_device(device)
    audio_file = audio_file.resolve()
    model_path = _absolute_lexical(model_path)
    config_path = model_path.parent / "config.json"
    config_input = config_path if config_path.exists() else None
    protected_inputs = [
        audio_file,
        model_path,
        config_path,
    ]
    _ensure_distinct_output(output, *protected_inputs)
    _prepare_output(output, force=force)
    audio_snapshot = _file_snapshot(audio_file)
    model_snapshot = _file_snapshot(model_path)
    config_snapshot = _file_snapshot(config_input) if config_input is not None else None
    audio_sha256 = _sha256_file(audio_file)
    model_sha256 = _sha256_file(model_path)
    effective_config = asdict(_resolve_config(model_path, model_path))
    config_metadata = {
        "source": (
            "adjacent-config.json" if config_input is not None else "embedded-fallback"
        ),
        "file_sha256": (
            _sha256_file(config_input) if config_input is not None else None
        ),
        "effective": effective_config,
    }
    prepared_audio = load_audio(audio_file).to(dtype=torch.float32)
    normalized_pcm_sha256 = _sha256_tensor(prepared_audio)
    wav = prepared_audio.to(device=torch_device)
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=torch_device,
        dtype=dtype,
    )
    if generation_telemetry:
        _require_base_generation_telemetry_model(model)

    model_name = model_label or model_path.stem
    input_name = audio_id or audio_file.name
    workload = BenchmarkWorkload(
        name=f"{model_name}:{input_name}",
        parameters={
            "model": {
                "label": model_name,
                "weights_sha256": model_sha256,
                "weights_bytes": model_path.stat().st_size,
                "config": config_metadata,
            },
            "audio": {
                "id": input_name,
                "file_sha256": audio_sha256,
                "normalized_pcm_sha256": normalized_pcm_sha256,
                "sample_rate_hz": 16_000,
                "num_samples": wav.shape[-1],
                "duration_seconds": wav.shape[-1] / 16_000,
            },
            "transcribe": transcribe_parameters,
        },
    )
    benchmark_environment = BenchmarkEnvironment(
        name=environment,
        metadata=_environment_metadata(torch_device, dtype),
    )
    protocol = BenchmarkProtocol(
        warmup_runs=warmup_runs,
        measured_runs=runs,
        generation_telemetry=("selected-output-v1" if generation_telemetry else "none"),
    )

    def instrumented_stream(observer):
        return model._transcribe_prepared(
            wav,
            **transcribe_parameters,
            _generation_observer=observer,
        )

    report = run_benchmark(
        lambda: model.transcribe((wav, 16_000), **transcribe_parameters),
        workload=workload,
        environment=benchmark_environment,
        protocol=protocol,
        synchronize=lambda: _strict_synchronize(torch_device),
        source=_source_metadata(),
        instrumented_event_stream_factory=(
            instrumented_stream if generation_telemetry else None
        ),
    )
    _ensure_unchanged(audio_file, audio_snapshot)
    _ensure_unchanged(model_path, model_snapshot)
    if config_input is not None:
        _ensure_unchanged(config_input, config_snapshot)
    elif config_path.exists():
        raise typer.BadParameter(
            f"Input changed while the benchmark ran: {config_path} appeared"
        )
    _atomic_write_text(output, report.to_json() + "\n", force=force)
    typer.echo(f"Wrote benchmark report to {output}", err=True)


@app.command("compare")
def compare_command(
    candidate: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    golden: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    max_regression_percent: Annotated[
        float,
        typer.Option(
            "--max-regression-percent",
            min=0,
            help="Maximum allowed increase in median wall time.",
        ),
    ],
) -> None:
    """同じ条件で作られたcandidateとgoldenを比較する。"""
    try:
        candidate_report = _read_report(candidate)
        golden_report = _read_report(golden)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise typer.BadParameter(f"Invalid benchmark report: {error}") from error

    try:
        policy = GoldenPolicy(max_regression_percent)
    except ValueError as error:
        raise typer.BadParameter(
            str(error), param_hint="--max-regression-percent"
        ) from error
    result = compare_reports(candidate_report, golden_report, policy)
    typer.echo(
        json.dumps(
            {
                "status": result.status.value,
                "reasons": list(result.reasons),
                "median_ratio": result.median_ratio,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if result.status is GateStatus.FAIL:
        raise typer.Exit(1)
    if result.status is GateStatus.INCOMPATIBLE:
        raise typer.Exit(3)


def main() -> None:
    app()
