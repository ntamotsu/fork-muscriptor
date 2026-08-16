"""モデル実装に依存しない、再現可能な採譜ベンチマーク用ヘルパー。"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent


TranscriptionEvent = NoteStartEvent | NoteEndEvent | ProgressEvent
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class BenchmarkWorkload:
    """計測対象を識別する名前とパラメーター。"""

    name: str
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """golden比較を同一条件に制限するハードウェア・ソフトウェア情報。"""

    name: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkProtocol:
    """モデルと正規化済み音声を事前準備するwarm計測プロトコル。"""

    scope: str = "model-loaded-audio-predecoded-on-device"
    warmup_runs: int = 1
    measured_runs: int = 5

    def __post_init__(self) -> None:
        _require_integer("warmup_runs", self.warmup_runs, minimum=0)
        _require_integer("measured_runs", self.measured_runs, minimum=1)


@dataclass(frozen=True)
class ProgressMilestone:
    completed: int
    total: int
    elapsed_ns: int

    def __post_init__(self) -> None:
        _require_integer("completed", self.completed, minimum=0)
        _require_integer("total", self.total, minimum=0)
        _require_integer("elapsed_ns", self.elapsed_ns, minimum=0)
        if self.completed > self.total:
            raise ValueError("completed must not exceed total")


@dataclass(frozen=True)
class EventCounts:
    note_start: int
    note_end: int
    progress: int

    def __post_init__(self) -> None:
        _require_integer("note_start", self.note_start, minimum=0)
        _require_integer("note_end", self.note_end, minimum=0)
        _require_integer("progress", self.progress, minimum=0)


@dataclass(frozen=True)
class BenchmarkSample:
    index: int
    wall_time_ns: int
    first_progress_ns: int | None
    first_note_ns: int | None
    progress_milestones: tuple[ProgressMilestone, ...]
    event_counts: EventCounts
    stream_digest: str
    note_digest: str

    def __post_init__(self) -> None:
        _require_integer("index", self.index, minimum=0)
        _require_integer("wall_time_ns", self.wall_time_ns, minimum=1)
        for field_name, value in (
            ("first_progress_ns", self.first_progress_ns),
            ("first_note_ns", self.first_note_ns),
        ):
            if value is not None:
                _require_integer(field_name, value, minimum=0)
                if value > self.wall_time_ns:
                    raise ValueError(f"{field_name} must not exceed wall_time_ns")
        if any(
            milestone.elapsed_ns > self.wall_time_ns
            for milestone in self.progress_milestones
        ):
            raise ValueError("progress milestone must not exceed wall_time_ns")


@dataclass(frozen=True)
class BenchmarkReport:
    workload: BenchmarkWorkload
    environment: BenchmarkEnvironment
    protocol: BenchmarkProtocol
    samples: tuple[BenchmarkSample, ...]
    # commit、dirty状態、日時は来歴であり、比較互換性の条件には含めない。
    source: Mapping[str, JSONValue] = field(default_factory=dict)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if len(self.samples) != self.protocol.measured_runs:
            raise ValueError(
                "samples must contain exactly protocol.measured_runs entries"
            )
        if tuple(sample.index for sample in self.samples) != tuple(
            range(len(self.samples))
        ):
            raise ValueError("sample indexes must be contiguous and start at zero")

    @property
    def median_wall_time_ns(self) -> float:
        return float(statistics.median(sample.wall_time_ns for sample in self.samples))

    @property
    def output_stable(self) -> bool:
        return self.stream_digest is not None and self.note_digest is not None

    @property
    def stream_digest(self) -> str | None:
        return _common_digest(sample.stream_digest for sample in self.samples)

    @property
    def note_digest(self) -> str | None:
        return _common_digest(sample.note_digest for sample in self.samples)

    def to_json(self) -> str:
        """reportをcanonical JSONへ直列化する。"""
        return json.dumps(
            self._to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkReport:
        """canonical JSONからreportを復元する。"""
        data = json.loads(payload, parse_constant=_reject_non_finite_json)
        if not isinstance(data, dict):
            raise ValueError("Benchmark report must be a JSON object")
        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported benchmark schema_version: {data.get('schema_version')!r}"
            )

        workload = _mapping(data, "workload")
        environment = _mapping(data, "environment")
        protocol = _mapping(data, "protocol")
        sample_data = data.get("samples")
        if not isinstance(sample_data, list):
            raise ValueError("Benchmark report field 'samples' must be a list")

        samples: list[BenchmarkSample] = []
        for raw_sample in sample_data:
            if not isinstance(raw_sample, dict):
                raise ValueError("Every benchmark sample must be a JSON object")
            raw_counts = _mapping(raw_sample, "event_counts")
            raw_milestones = raw_sample.get("progress_milestones")
            if not isinstance(raw_milestones, list):
                raise ValueError("Sample progress_milestones must be a list")
            samples.append(
                BenchmarkSample(
                    index=raw_sample["index"],
                    wall_time_ns=raw_sample["wall_time_ns"],
                    first_progress_ns=raw_sample["first_progress_ns"],
                    first_note_ns=raw_sample["first_note_ns"],
                    progress_milestones=tuple(
                        ProgressMilestone(
                            completed=milestone["completed"],
                            total=milestone["total"],
                            elapsed_ns=milestone["elapsed_ns"],
                        )
                        for milestone in raw_milestones
                    ),
                    event_counts=EventCounts(
                        note_start=raw_counts["note_start"],
                        note_end=raw_counts["note_end"],
                        progress=raw_counts["progress"],
                    ),
                    stream_digest=raw_sample["stream_digest"],
                    note_digest=raw_sample["note_digest"],
                )
            )

        report = cls(
            workload=BenchmarkWorkload(
                name=workload["name"],
                parameters=_mapping(workload, "parameters"),
            ),
            environment=BenchmarkEnvironment(
                name=environment["name"],
                metadata=_mapping(environment, "metadata"),
            ),
            protocol=BenchmarkProtocol(
                scope=protocol["scope"],
                warmup_runs=protocol["warmup_runs"],
                measured_runs=protocol["measured_runs"],
            ),
            samples=tuple(samples),
            source=_mapping(data, "source"),
        )
        if data.get("summary") != report._summary_dict():
            raise ValueError("Benchmark report summary does not match its samples")
        return report

    def _summary_dict(self) -> dict[str, JSONValue]:
        return {
            "median_wall_time_ns": self.median_wall_time_ns,
            "output_stable": self.output_stable,
            "stream_digest": self.stream_digest,
            "note_digest": self.note_digest,
        }

    def _to_dict(self) -> dict[str, JSONValue]:
        data = asdict(self)
        data["summary"] = self._summary_dict()
        return data


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class GoldenPolicy:
    max_median_regression_percent: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_median_regression_percent, bool)
            or not isinstance(self.max_median_regression_percent, (int, float))
            or not math.isfinite(self.max_median_regression_percent)
            or self.max_median_regression_percent < 0
        ):
            raise ValueError(
                "max_median_regression_percent must be finite and non-negative"
            )


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    reasons: tuple[str, ...] = ()
    median_ratio: float | None = None

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS


def compare_reports(
    candidate: BenchmarkReport,
    golden: BenchmarkReport,
    policy: GoldenPolicy,
) -> GateResult:
    """互換なreport同士の出力同等性と中央値回帰を判定する。"""
    incompatibilities: list[str] = []
    if candidate.environment != golden.environment:
        incompatibilities.append("environment_mismatch")
    if candidate.workload != golden.workload:
        incompatibilities.append("workload_mismatch")
    if candidate.protocol != golden.protocol:
        incompatibilities.append("protocol_mismatch")
    if incompatibilities:
        return GateResult(GateStatus.INCOMPATIBLE, tuple(incompatibilities))

    quality_failures: list[str] = []
    if not golden.output_stable:
        quality_failures.append("golden_output_unstable")
    if not candidate.output_stable:
        quality_failures.append("candidate_output_unstable")
    if not quality_failures:
        if candidate.stream_digest != golden.stream_digest:
            quality_failures.append("stream_digest_mismatch")
        if candidate.note_digest != golden.note_digest:
            quality_failures.append("note_digest_mismatch")
    if quality_failures:
        return GateResult(GateStatus.FAIL, tuple(quality_failures))

    golden_median = golden.median_wall_time_ns
    candidate_median = candidate.median_wall_time_ns
    median_ratio = candidate_median / golden_median
    allowed_median = golden_median * (1 + policy.max_median_regression_percent / 100)
    if candidate_median > allowed_median:
        return GateResult(
            GateStatus.FAIL,
            reasons=("median_wall_time_regression",),
            median_ratio=median_ratio,
        )
    return GateResult(GateStatus.PASS, median_ratio=median_ratio)


def _noop() -> None:
    pass


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _reject_non_finite_json(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _common_digest(digests: Iterable[str]) -> str | None:
    values = set(digests)
    if len(values) != 1:
        return None
    return next(iter(values))


def _mapping(container: Mapping[str, object], key: str) -> dict[str, JSONValue]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark report field {key!r} must be a JSON object")
    return value


def run_benchmark(
    event_stream_factory: Callable[[], Iterable[TranscriptionEvent]],
    *,
    workload: BenchmarkWorkload,
    environment: BenchmarkEnvironment,
    protocol: BenchmarkProtocol = BenchmarkProtocol(),
    source: Mapping[str, JSONValue] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    synchronize: Callable[[], None] = _noop,
) -> BenchmarkReport:
    """遅延イベントstreamに対してwarmupと計測runを実行する。"""
    for _ in range(protocol.warmup_runs):
        synchronize()
        for _event in event_stream_factory():
            pass
        synchronize()

    samples: list[BenchmarkSample] = []
    for index in range(protocol.measured_runs):
        synchronize()
        started_ns = clock_ns()
        events: list[TranscriptionEvent] = []
        milestones: list[ProgressMilestone] = []
        first_progress_ns: int | None = None
        first_note_ns: int | None = None
        note_start_count = 0
        note_end_count = 0
        progress_count = 0

        for event in event_stream_factory():
            events.append(event)
            if isinstance(event, ProgressEvent):
                elapsed_ns = clock_ns() - started_ns
                if first_progress_ns is None:
                    first_progress_ns = elapsed_ns
                milestones.append(
                    ProgressMilestone(event.completed, event.total, elapsed_ns)
                )
                progress_count += 1
            elif isinstance(event, NoteStartEvent):
                if first_note_ns is None:
                    first_note_ns = clock_ns() - started_ns
                note_start_count += 1
            elif isinstance(event, NoteEndEvent):
                note_end_count += 1
            else:
                raise TypeError(f"Unsupported transcription event: {type(event)!r}")

        synchronize()
        wall_time_ns = clock_ns() - started_ns
        samples.append(
            BenchmarkSample(
                index=index,
                wall_time_ns=wall_time_ns,
                first_progress_ns=first_progress_ns,
                first_note_ns=first_note_ns,
                progress_milestones=tuple(milestones),
                event_counts=EventCounts(
                    note_start=note_start_count,
                    note_end=note_end_count,
                    progress=progress_count,
                ),
                stream_digest=canonical_stream_digest(events),
                note_digest=canonical_note_digest(events),
            )
        )
    return BenchmarkReport(
        workload, environment, protocol, tuple(samples), source or {}
    )


def _time_us(seconds: float) -> int:
    return round(seconds * 1_000_000)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_stream_digest(events: Iterable[TranscriptionEvent]) -> str:
    """公開採譜イベントstreamから安定したdigestを返す。"""
    records: list[dict[str, int | str]] = []
    for event in events:
        if isinstance(event, NoteStartEvent):
            records.append(
                {
                    "type": "note_start",
                    "pitch": event.pitch,
                    "start_time_us": _time_us(event.start_time),
                    "index": event.index,
                    "instrument": event.instrument,
                }
            )
        elif isinstance(event, NoteEndEvent):
            records.append(
                {
                    "type": "note_end",
                    "end_time_us": _time_us(event.end_time),
                    "start_event_index": event.start_event_index,
                }
            )
        elif isinstance(event, ProgressEvent):
            records.append(
                {
                    "type": "progress",
                    "completed": event.completed,
                    "total": event.total,
                }
            )
        else:
            raise TypeError(f"Unsupported transcription event: {type(event)!r}")
    return _digest(records)


def canonical_note_digest(events: Iterable[TranscriptionEvent]) -> str:
    """イベントstream内の完結したnoteから安定したdigestを返す。"""
    starts: dict[int, NoteStartEvent] = {}
    notes: list[tuple[str, int, int, int]] = []
    for event in events:
        if isinstance(event, NoteStartEvent):
            if event.index in starts:
                raise ValueError(f"Duplicate note start index: {event.index}")
            starts[event.index] = event
        elif isinstance(event, NoteEndEvent):
            try:
                start = starts.pop(event.start_event_index)
            except KeyError as exc:
                raise ValueError(
                    f"Note end references unknown start index: {event.start_event_index}"
                ) from exc
            notes.append(
                (
                    start.instrument,
                    start.pitch,
                    _time_us(start.start_time),
                    _time_us(event.end_time),
                )
            )
        elif not isinstance(event, ProgressEvent):
            raise TypeError(f"Unsupported transcription event: {type(event)!r}")
    if starts:
        raise ValueError(f"Unclosed note start indexes: {sorted(starts)}")
    return _digest(sorted(notes))
