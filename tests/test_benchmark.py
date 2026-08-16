"""ベンチマークharnessとgolden gateの隔離されたテスト。"""

import json
from dataclasses import replace

import pytest

from muscriptor.benchmark import (
    BenchmarkEnvironment,
    BenchmarkProtocol,
    BenchmarkReport,
    BenchmarkSample,
    BenchmarkWorkload,
    EventCounts,
    GateResult,
    GateStatus,
    GoldenPolicy,
    ProgressMilestone,
    canonical_note_digest,
    canonical_stream_digest,
    compare_reports,
    run_benchmark,
)
from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent


def _note_stream(
    pitch: int = 60,
    *,
    index: int = 0,
    include_progress: bool = True,
):
    start = NoteStartEvent(
        pitch=pitch,
        start_time=0.25,
        index=index,
        instrument="piano",
    )
    events = [start, NoteEndEvent(end_time=0.75, start_event=start)]
    if include_progress:
        return [
            ProgressEvent(completed=0, total=1),
            *events,
            ProgressEvent(completed=1, total=1),
        ]
    return events


class _ManualClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def advance(self, nanoseconds: int):
        self.now_ns += nanoseconds


def test_protocol_rejects_non_positive_measured_runs():
    with pytest.raises(ValueError, match="measured_runs"):
        BenchmarkProtocol(measured_runs=0)


def test_protocol_rejects_negative_warmup_runs():
    with pytest.raises(ValueError, match="warmup_runs"):
        BenchmarkProtocol(warmup_runs=-1)


@pytest.mark.parametrize(
    "wall_time_ns",
    [0, -1, 1.5, True, float("nan"), float("inf")],
)
def test_sample_rejects_non_positive_wall_time(wall_time_ns):
    with pytest.raises(ValueError, match="wall_time_ns"):
        BenchmarkSample(
            index=0,
            wall_time_ns=wall_time_ns,
            first_progress_ns=None,
            first_note_ns=None,
            progress_milestones=(),
            event_counts=EventCounts(note_start=0, note_end=0, progress=0),
            stream_digest="stream",
            note_digest="notes",
        )


@pytest.mark.parametrize("value", [-1, 1.5, True, float("inf")])
def test_event_counts_require_non_negative_integers(value):
    with pytest.raises(ValueError, match="note_start"):
        EventCounts(note_start=value, note_end=0, progress=0)


@pytest.mark.parametrize("value", [-1, 1.5, True, float("inf")])
def test_progress_milestones_require_non_negative_integers(value):
    with pytest.raises(ValueError, match="elapsed_ns"):
        ProgressMilestone(completed=0, total=1, elapsed_ns=value)


def test_report_rejects_a_sample_count_that_differs_from_the_protocol():
    sample = BenchmarkSample(
        index=0,
        wall_time_ns=1,
        first_progress_ns=None,
        first_note_ns=None,
        progress_milestones=(),
        event_counts=EventCounts(note_start=0, note_end=0, progress=0),
        stream_digest="stream",
        note_digest="notes",
    )

    with pytest.raises(ValueError, match="samples"):
        BenchmarkReport(
            workload=BenchmarkWorkload("fake"),
            environment=BenchmarkEnvironment("cpu-test"),
            protocol=BenchmarkProtocol(measured_runs=2),
            samples=(sample,),
        )


@pytest.mark.parametrize(
    "percent",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_golden_policy_requires_a_finite_non_negative_percent(percent):
    with pytest.raises(ValueError, match="max_median_regression_percent"):
        GoldenPolicy(percent)


def test_stream_digest_changes_when_public_event_data_changes():
    assert canonical_stream_digest(_note_stream(60)) != canonical_stream_digest(
        _note_stream(61)
    )


def test_note_digest_changes_when_note_semantics_change():
    assert canonical_note_digest(_note_stream(60)) != canonical_note_digest(
        _note_stream(61)
    )


def test_note_digest_ignores_progress_and_stream_indexes():
    assert canonical_note_digest(_note_stream()) == canonical_note_digest(
        _note_stream(index=99, include_progress=False)
    )


def test_run_benchmark_discards_warmups_and_consumes_every_lazy_stream():
    calls = 0
    yielded = 0

    def stream():
        nonlocal calls, yielded
        calls += 1
        for event in _note_stream():
            yielded += 1
            yield event

    run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=2, measured_runs=3),
    )

    assert (calls, yielded) == (5, 20)


def test_run_benchmark_synchronizes_outside_each_timed_lazy_stream():
    actions: list[str] = []
    clock_values = iter((0, 1))

    def stream():
        actions.append("stream-start")
        actions.append("stream-end")
        yield from ()

    def synchronize():
        actions.append("sync")

    def clock_ns():
        actions.append("clock")
        return next(clock_values)

    run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=1, measured_runs=1),
        synchronize=synchronize,
        clock_ns=clock_ns,
    )

    assert actions == [
        "sync",
        "stream-start",
        "stream-end",
        "sync",
        "sync",
        "clock",
        "stream-start",
        "stream-end",
        "sync",
        "clock",
    ]


def test_run_benchmark_records_elapsed_time_progress_and_event_counts():
    clock = _ManualClock()
    events = _note_stream()

    def stream():
        for delay_ns, event in zip((10, 20, 30, 40), events, strict=True):
            clock.advance(delay_ns)
            yield event

    def synchronize():
        clock.advance(5)

    report = run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        synchronize=synchronize,
        clock_ns=clock,
    )

    assert report.samples == (
        BenchmarkSample(
            index=0,
            wall_time_ns=105,
            first_progress_ns=10,
            first_note_ns=30,
            progress_milestones=(
                ProgressMilestone(completed=0, total=1, elapsed_ns=10),
                ProgressMilestone(completed=1, total=1, elapsed_ns=100),
            ),
            event_counts=EventCounts(note_start=1, note_end=1, progress=2),
            stream_digest=canonical_stream_digest(events),
            note_digest=canonical_note_digest(events),
        ),
    )


def test_report_summarizes_median_and_stable_output():
    clock = _ManualClock()
    durations = iter((30, 10, 20))

    def stream():
        clock.advance(next(durations))
        yield from _note_stream()

    report = run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=3),
        clock_ns=clock,
    )

    assert (
        report.median_wall_time_ns,
        report.output_stable,
        report.stream_digest,
        report.note_digest,
    ) == (
        20,
        True,
        canonical_stream_digest(_note_stream()),
        canonical_note_digest(_note_stream()),
    )


def test_report_marks_varying_output_as_unstable():
    clock = _ManualClock()
    pitches = iter((60, 61))

    def stream():
        clock.advance(1)
        yield from _note_stream(next(pitches))

    report = run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=2),
        clock_ns=clock,
    )

    assert (report.output_stable, report.stream_digest, report.note_digest) == (
        False,
        None,
        None,
    )


def test_report_v1_round_trips_as_canonical_json():
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload(
            "short-song",
            {"model": {"weights_sha256": "abc"}, "batch_size": 1},
        ),
        environment=BenchmarkEnvironment(
            "m4-test",
            {"device": {"type": "mps"}, "torch_version": "test"},
        ),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        source={
            "git_commit": "deadbeef",
            "git_dirty": False,
            "created_at_utc": "2026-08-16T00:00:00Z",
        },
        clock_ns=iter((10, 20, 30, 40, 50)).__next__,
    )

    payload = report.to_json()
    restored = type(report).from_json(payload)

    assert (restored, restored.to_json()) == (report, payload)


def test_report_v1_rejects_unknown_schema_version():
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        type(report).from_json(json.dumps(data))


def test_report_v1_rejects_non_positive_wall_time_from_json():
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["samples"][0]["wall_time_ns"] = 0
    data["summary"]["median_wall_time_ns"] = 0.0

    with pytest.raises(ValueError, match="wall_time_ns"):
        type(report).from_json(json.dumps(data))


@pytest.mark.parametrize("wall_time_ns", [1.5, True, float("inf")])
def test_report_v1_rejects_non_integer_wall_time_from_json(wall_time_ns):
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["samples"][0]["wall_time_ns"] = wall_time_ns
    data["summary"]["median_wall_time_ns"] = float(wall_time_ns)

    with pytest.raises(ValueError, match="wall_time_ns|non-finite"):
        type(report).from_json(json.dumps(data))


def _completed_report(
    *,
    environment: str = "m4",
    workload: str = "short-song",
    scope: str = "model-loaded-audio-predecoded-on-device",
    wall_times: tuple[int, ...] = (100, 100, 100),
    stream_digest: str = "stream-a",
    note_digest: str = "notes-a",
    source: dict | None = None,
):
    samples = tuple(
        BenchmarkSample(
            index=index,
            wall_time_ns=wall_time,
            first_progress_ns=10,
            first_note_ns=20,
            progress_milestones=(),
            event_counts=EventCounts(note_start=1, note_end=1, progress=0),
            stream_digest=stream_digest,
            note_digest=note_digest,
        )
        for index, wall_time in enumerate(wall_times)
    )
    return BenchmarkReport(
        workload=BenchmarkWorkload(workload),
        environment=BenchmarkEnvironment(environment),
        protocol=BenchmarkProtocol(
            scope=scope,
            warmup_runs=1,
            measured_runs=len(samples),
        ),
        samples=samples,
        source=source or {},
    )


def test_golden_gate_rejects_a_different_environment():
    golden = _completed_report(environment="m4")
    candidate = _completed_report(environment="t4")

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.INCOMPATIBLE,
        reasons=("environment_mismatch",),
    )


def test_golden_gate_rejects_a_different_workload():
    golden = _completed_report(workload="short-song")
    candidate = _completed_report(workload="long-song")

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.INCOMPATIBLE,
        reasons=("workload_mismatch",),
    )


def test_golden_gate_rejects_a_different_protocol():
    golden = _completed_report()
    candidate = _completed_report(scope="different-timing-boundary")

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.INCOMPATIBLE,
        reasons=("protocol_mismatch",),
    )


def test_golden_gate_checks_output_before_timing():
    golden = _completed_report(wall_times=(100, 100, 100))
    candidate = _completed_report(
        wall_times=(50, 50, 50),
        stream_digest="stream-b",
        note_digest="notes-b",
    )

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.FAIL,
        reasons=("stream_digest_mismatch", "note_digest_mismatch"),
    )


def test_golden_gate_accepts_the_regression_budget_boundary():
    golden = _completed_report(wall_times=(90, 100, 110))
    candidate = _completed_report(wall_times=(95, 105, 115))

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(GateStatus.PASS, median_ratio=1.05)


def test_golden_gate_fails_above_the_regression_budget():
    golden = _completed_report(wall_times=(90, 100, 110))
    candidate = _completed_report(wall_times=(96, 106, 116))

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.FAIL,
        reasons=("median_wall_time_regression",),
        median_ratio=1.06,
    )


def test_golden_gate_ignores_source_provenance_for_compatibility():
    golden = _completed_report(
        source={"git_commit": "before", "created_at_utc": "earlier"}
    )
    candidate = _completed_report(
        source={"git_commit": "after", "created_at_utc": "later"}
    )

    result = compare_reports(candidate, golden, GoldenPolicy(0.0))

    assert result == GateResult(GateStatus.PASS, median_ratio=1.0)


def test_golden_gate_rejects_unstable_candidate_output():
    golden = _completed_report()
    changed_sample = replace(golden.samples[1], stream_digest="stream-b")
    candidate = replace(
        golden,
        samples=(golden.samples[0], changed_sample, golden.samples[2]),
    )

    result = compare_reports(candidate, golden, GoldenPolicy(5.0))

    assert result == GateResult(
        GateStatus.FAIL,
        reasons=("candidate_output_unstable",),
    )
