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
from muscriptor.generation_telemetry import ChunkGenerationStats


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


def _chunk_stats(chunk_index: int = 0) -> ChunkGenerationStats:
    return ChunkGenerationStats(
        chunk_index=chunk_index,
        seek_time_us=chunk_index * 5_000_000,
        prompt_tokens=0,
        observed_rows=3,
        generated_rows=3,
        eos_step=3,
        max_gen_len=2000,
        hit_generation_limit=False,
    )


def test_protocol_rejects_non_positive_measured_runs():
    with pytest.raises(ValueError, match="measured_runs"):
        BenchmarkProtocol(measured_runs=0)


def test_protocol_rejects_negative_warmup_runs():
    with pytest.raises(ValueError, match="warmup_runs"):
        BenchmarkProtocol(warmup_runs=-1)


@pytest.mark.parametrize(
    "generation_telemetry",
    [True, 1, [], "unknown"],
)
def test_protocol_rejects_invalid_generation_telemetry(generation_telemetry):
    with pytest.raises(ValueError, match="generation_telemetry"):
        BenchmarkProtocol(generation_telemetry=generation_telemetry)


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


@pytest.mark.parametrize(
    "chunk_generation_stats",
    [
        [_chunk_stats()],
        (_chunk_stats(), object()),
        (_chunk_stats(0), _chunk_stats(2)),
        (_chunk_stats(0), _chunk_stats(0)),
    ],
)
def test_sample_requires_contiguous_typed_chunk_generation_stats(
    chunk_generation_stats,
):
    with pytest.raises(ValueError, match="chunk_generation_stats|chunk_index"):
        BenchmarkSample(
            index=0,
            wall_time_ns=1,
            first_progress_ns=None,
            first_note_ns=None,
            progress_milestones=(),
            event_counts=EventCounts(note_start=0, note_end=0, progress=0),
            stream_digest="stream",
            note_digest="notes",
            chunk_generation_stats=chunk_generation_stats,
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
    ("generation_telemetry", "chunk_generation_stats"),
    [
        ("none", ()),
        ("selected-output-v1", None),
    ],
)
def test_report_requires_protocol_and_sample_telemetry_to_match(
    generation_telemetry,
    chunk_generation_stats,
):
    sample = BenchmarkSample(
        index=0,
        wall_time_ns=1,
        first_progress_ns=None,
        first_note_ns=None,
        progress_milestones=(),
        event_counts=EventCounts(note_start=0, note_end=0, progress=0),
        stream_digest="stream",
        note_digest="notes",
        chunk_generation_stats=chunk_generation_stats,
    )

    with pytest.raises(ValueError, match="generation_telemetry"):
        BenchmarkReport(
            workload=BenchmarkWorkload("fake"),
            environment=BenchmarkEnvironment("cpu-test"),
            protocol=BenchmarkProtocol(
                warmup_runs=0,
                measured_runs=1,
                generation_telemetry=generation_telemetry,
            ),
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


def test_run_benchmark_discards_warmup_telemetry_and_records_measured_runs():
    calls = 0

    def unexpected_plain_stream():
        pytest.fail("instrumented runs must use the observer-aware factory")

    def instrumented_stream(observer):
        nonlocal calls
        calls += 1
        observer(
            ChunkGenerationStats(
                chunk_index=0,
                seek_time_us=calls,
                prompt_tokens=0,
                observed_rows=3,
                generated_rows=3,
                eos_step=3,
                max_gen_len=2000,
                hit_generation_limit=False,
            )
        )
        yield from _note_stream()

    report = run_benchmark(
        unexpected_plain_stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(
            warmup_runs=2,
            measured_runs=2,
            generation_telemetry="selected-output-v1",
        ),
        instrumented_event_stream_factory=instrumented_stream,
    )

    assert (
        calls,
        tuple(
            sample.chunk_generation_stats[0].seek_time_us for sample in report.samples
        ),
    ) == (4, (3, 4))


def test_run_benchmark_rejects_missing_stats_for_instrumented_progress():
    def instrumented_stream(_observer):
        yield from _note_stream()

    with pytest.raises(ValueError, match="chunk_generation_stats count"):
        run_benchmark(
            lambda: _note_stream(),
            workload=BenchmarkWorkload("fake"),
            environment=BenchmarkEnvironment("cpu-test"),
            protocol=BenchmarkProtocol(
                warmup_runs=0,
                measured_runs=1,
                generation_telemetry="selected-output-v1",
            ),
            instrumented_event_stream_factory=instrumented_stream,
            clock_ns=iter((0, 1, 2, 3, 4)).__next__,
        )


def test_run_benchmark_releases_warmup_telemetry_before_the_final_sync():
    actions: list[str] = []
    stream_calls = 0
    sync_count = 0

    class ObservedStats:
        def __del__(self):
            actions.append("release-stats")

    def instrumented_stream(observer):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            observer(ObservedStats())
            yield from ()
        else:
            yield ProgressEvent(completed=0, total=0)

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    run_benchmark(
        lambda: (),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(
            warmup_runs=1,
            measured_runs=1,
            generation_telemetry="selected-output-v1",
        ),
        instrumented_event_stream_factory=instrumented_stream,
        synchronize=synchronize,
        clock_ns=iter((0, 1, 2)).__next__,
    )

    assert actions.index("release-stats") < actions.index("sync-2")


@pytest.mark.parametrize(
    ("generation_telemetry", "use_instrumented_factory"),
    [
        ("none", True),
        ("selected-output-v1", False),
    ],
)
def test_run_benchmark_requires_factory_and_protocol_telemetry_to_match(
    generation_telemetry,
    use_instrumented_factory,
):
    calls = []

    def plain_stream():
        calls.append("plain")
        return ()

    def instrumented_stream(_observer):
        calls.append("instrumented")
        return ()

    with pytest.raises(ValueError, match="generation_telemetry"):
        run_benchmark(
            plain_stream,
            workload=BenchmarkWorkload("fake"),
            environment=BenchmarkEnvironment("cpu-test"),
            protocol=BenchmarkProtocol(
                warmup_runs=0,
                measured_runs=1,
                generation_telemetry=generation_telemetry,
            ),
            instrumented_event_stream_factory=(
                instrumented_stream if use_instrumented_factory else None
            ),
        )

    assert calls == []


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


def test_run_benchmark_releases_warmup_events_before_the_final_warmup_sync():
    actions: list[str] = []
    stream_calls = 0
    sync_count = 0

    class ObservedProgressEvent(ProgressEvent):
        def __del__(self):
            actions.append("release")

    def stream():
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield ObservedProgressEvent(completed=0, total=0)

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=1, measured_runs=1),
        synchronize=synchronize,
        clock_ns=iter((0, 1)).__next__,
    )

    assert actions.index("release") < actions.index("sync-2")


def test_run_benchmark_releases_previous_events_before_the_next_start_sync():
    clock = _ManualClock()
    actions: list[str] = []
    sync_count = 0

    class TimedProgressEvent(ProgressEvent):
        def __del__(self):
            actions.append("release")
            clock.advance(100)

    def stream():
        clock.advance(10)
        yield TimedProgressEvent(completed=0, total=0)

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    report = run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=2),
        synchronize=synchronize,
        clock_ns=clock,
    )

    assert (
        tuple(sample.wall_time_ns for sample in report.samples),
        actions.index("release") < actions.index("sync-3"),
        report.stream_digest,
        report.note_digest,
    ) == (
        (10, 10),
        True,
        canonical_stream_digest([ProgressEvent(completed=0, total=0)]),
        canonical_note_digest([ProgressEvent(completed=0, total=0)]),
    )


def test_run_benchmark_releases_previous_event_before_an_empty_run():
    clock = _ManualClock()
    actions: list[str] = []
    stream_calls = 0
    sync_count = 0

    class TimedProgressEvent(ProgressEvent):
        def __del__(self):
            actions.append("release")
            clock.advance(100)

    def stream():
        nonlocal stream_calls
        stream_calls += 1
        clock.advance(10)
        if stream_calls == 1:
            yield TimedProgressEvent(completed=0, total=0)

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    report = run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=2),
        synchronize=synchronize,
        clock_ns=clock,
    )

    assert (
        tuple(sample.wall_time_ns for sample in report.samples),
        actions.index("release") < actions.index("sync-3"),
    ) == ((10, 10), True)


def test_run_benchmark_releases_measured_iterator_before_the_final_sync():
    actions: list[str] = []
    sync_count = 0

    class ObservedIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def __del__(self):
            actions.append("release-stream")

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    run_benchmark(
        ObservedIterator,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        synchronize=synchronize,
        clock_ns=iter((0, 1)).__next__,
    )

    assert actions.index("release-stream") < actions.index("sync-2")


def test_run_benchmark_closes_externally_referenced_iterators_before_sync():
    actions: list[str] = []
    iterators = []
    sync_count = 0

    class ObservedIterator:
        def __init__(self, index):
            self.index = index

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            actions.append(f"close-{self.index}")

    def stream():
        iterator = ObservedIterator(len(iterators))
        iterators.append(iterator)
        return iterator

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    run_benchmark(
        stream,
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=1, measured_runs=1),
        synchronize=synchronize,
        clock_ns=iter((0, 1)).__next__,
    )

    assert (
        actions.index("close-0") < actions.index("sync-2"),
        actions.index("close-1") < actions.index("sync-4"),
    ) == (True, True)


def test_run_benchmark_releases_previous_event_before_a_later_run_fails():
    clock = _ManualClock()
    actions: list[str] = []
    stream_calls = 0
    sync_count = 0

    class ObservedProgressEvent(ProgressEvent):
        def __del__(self):
            actions.append("release")

    def stream():
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            clock.advance(10)
            yield ObservedProgressEvent(completed=0, total=0)
            return
        raise RuntimeError("fake stream failure")

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        actions.append(f"sync-{sync_count}")

    with pytest.raises(RuntimeError, match="fake stream failure"):
        run_benchmark(
            stream,
            workload=BenchmarkWorkload("fake"),
            environment=BenchmarkEnvironment("cpu-test"),
            protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=2),
            synchronize=synchronize,
            clock_ns=clock,
        )

    assert actions.index("release") < actions.index("sync-3")


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


def test_report_v2_round_trips_generation_telemetry_as_canonical_json():
    def instrumented_stream(observer):
        observer(_chunk_stats())
        yield from _note_stream()

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
        protocol=BenchmarkProtocol(
            warmup_runs=0,
            measured_runs=1,
            generation_telemetry="selected-output-v1",
        ),
        source={
            "git_commit": "deadbeef",
            "git_dirty": False,
            "created_at_utc": "2026-08-16T00:00:00Z",
        },
        clock_ns=iter((10, 20, 30, 40, 50)).__next__,
        instrumented_event_stream_factory=instrumented_stream,
    )

    payload = report.to_json()
    restored = type(report).from_json(payload)

    assert (
        restored,
        restored.to_json(),
        restored.samples[0].chunk_generation_stats,
    ) == (report, payload, (_chunk_stats(),))


def test_report_v1_is_migrated_to_v2_without_generation_telemetry():
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["schema_version"] = 1
    del data["protocol"]["generation_telemetry"]
    del data["samples"][0]["chunk_generation_stats"]

    restored = type(report).from_json(json.dumps(data))

    assert (
        restored.schema_version,
        restored.protocol.generation_telemetry,
        restored.samples[0].chunk_generation_stats,
        json.loads(restored.to_json())["schema_version"],
    ) == (2, "none", None, 2)


@pytest.mark.parametrize(
    "v2_field",
    ["protocol.generation_telemetry", "sample.chunk_generation_stats"],
)
def test_report_v1_rejects_v2_fields_instead_of_downgrading_them(v2_field):
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["schema_version"] = 1
    if v2_field.startswith("protocol"):
        del data["samples"][0]["chunk_generation_stats"]
    else:
        del data["protocol"]["generation_telemetry"]

    with pytest.raises(ValueError, match="generation_telemetry|chunk_generation_stats"):
        type(report).from_json(json.dumps(data))


@pytest.mark.parametrize("schema_version", [3, True, [], "2", None])
def test_report_rejects_unknown_schema_version(schema_version):
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        type(report).from_json(json.dumps(data))


@pytest.mark.parametrize(
    "missing_field",
    ["protocol.generation_telemetry", "sample.chunk_generation_stats"],
)
def test_report_v2_rejects_missing_generation_telemetry_fields(missing_field):
    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(warmup_runs=0, measured_runs=1),
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    if missing_field.startswith("protocol"):
        del data["protocol"]["generation_telemetry"]
    else:
        del data["samples"][0]["chunk_generation_stats"]

    with pytest.raises(ValueError, match="generation_telemetry|chunk_generation_stats"):
        type(report).from_json(json.dumps(data))


@pytest.mark.parametrize(
    "raw_stats",
    [
        1,
        {"chunk_index": 0},
        {
            "chunk_index": 0,
            "seek_time_us": 0,
            "prompt_tokens": 0,
            "observed_rows": 3,
            "generated_rows": 3,
            "eos_step": 3,
            "max_gen_len": 2000,
            "hit_generation_limit": False,
            "extra": "rejected",
        },
    ],
)
def test_report_v2_rejects_noncanonical_chunk_generation_stats(raw_stats):
    def instrumented_stream(observer):
        observer(_chunk_stats())
        yield from _note_stream()

    report = run_benchmark(
        lambda: _note_stream(),
        workload=BenchmarkWorkload("fake"),
        environment=BenchmarkEnvironment("cpu-test"),
        protocol=BenchmarkProtocol(
            warmup_runs=0,
            measured_runs=1,
            generation_telemetry="selected-output-v1",
        ),
        instrumented_event_stream_factory=instrumented_stream,
        clock_ns=iter((0, 1, 2, 3, 4)).__next__,
    )
    data = json.loads(report.to_json())
    data["samples"][0]["chunk_generation_stats"] = [raw_stats]

    with pytest.raises(ValueError, match="chunk_generation_stats"):
        type(report).from_json(json.dumps(data))


def test_report_v2_rejects_non_positive_wall_time_from_json():
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
def test_report_v2_rejects_non_integer_wall_time_from_json(wall_time_ns):
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
    generation_telemetry: str = "none",
):
    samples = tuple(
        BenchmarkSample(
            index=index,
            wall_time_ns=wall_time,
            first_progress_ns=10,
            first_note_ns=20,
            progress_milestones=(
                (ProgressMilestone(completed=0, total=0, elapsed_ns=10),)
                if generation_telemetry != "none"
                else ()
            ),
            event_counts=EventCounts(
                note_start=1,
                note_end=1,
                progress=1 if generation_telemetry != "none" else 0,
            ),
            stream_digest=stream_digest,
            note_digest=note_digest,
            chunk_generation_stats=(() if generation_telemetry != "none" else None),
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
            generation_telemetry=generation_telemetry,
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


def test_golden_gate_keeps_generation_telemetry_separate_from_v1_golden():
    legacy_data = json.loads(_completed_report().to_json())
    legacy_data["schema_version"] = 1
    del legacy_data["protocol"]["generation_telemetry"]
    for sample in legacy_data["samples"]:
        del sample["chunk_generation_stats"]
    legacy_golden = BenchmarkReport.from_json(json.dumps(legacy_data))
    instrumented_candidate = _completed_report(
        generation_telemetry="selected-output-v1"
    )

    result = compare_reports(
        instrumented_candidate,
        legacy_golden,
        GoldenPolicy(5.0),
    )

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
