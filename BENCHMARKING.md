# Reproducible inference benchmarks

`muscriptor-benchmark` measures the warm transcription path without changing the
normal `muscriptor` CLI. It writes a machine-readable JSON report containing the
timing samples, public event digests, workload hashes, and an environment
identity.

Reports are currently written as schema v2. The reader still accepts schema v1
reports and migrates them in memory with generation telemetry disabled.

## What is measured

The `model-loaded-audio-predecoded-on-device` protocol prepares the model and a
mono, float32, 16 kHz audio tensor before timing. A plain recorded run then
measures complete consumption of `TranscriptionModel.transcribe(...)`, including
condition construction, autoregressive decoding, and public event construction.
The telemetry variant enters the equivalent prepared-audio path directly, as
described below. Accelerator work is synchronized immediately before the start
clock and after the generator is exhausted.

Model loading, file decoding, resampling, and the initial device transfer are not
included. One warmup and five measured runs are used by default. No benchmark
results or machine-specific golden files are committed to this repository.
The workload uses greedy decoding, batch size 1, prelude forcing, and strict EOS
checking (`no_eos_is_ok=False`) so a truncated generation cannot be reported as a
successful performance sample.

The local weight file and an adjacent `config.json` (when present) are both
fingerprinted and checked again after measurement. The report is not written if an
input changes during the run or if an accelerator synchronization fails.

## Run on Apple Silicon

Use a local weights file so its exact bytes can be fingerprinted. Give the audio
an identifier that remains stable if the file is moved:

```console
uv run muscriptor-benchmark run /path/to/song.wav \
  --model /path/to/model.safetensors \
  --model-label medium \
  --audio-id song-long-v1 \
  --device mps \
  --dtype float16 \
  --environment m4-pro-macos \
  --output artifacts/m4-medium-baseline.json
```

Run the candidate under the same machine state, inputs, options, and software
versions, then compare it with an explicit regression allowance:

```console
uv run muscriptor-benchmark compare \
  artifacts/m4-medium-candidate.json \
  artifacts/m4-medium-baseline.json \
  --max-regression-percent 3
```

The comparison is rejected as incompatible before timing is considered if the
workload, environment, or protocol differs. It also fails if measured runs are
not output-stable or if the candidate changes either the exact public event stream
or the normalized completed notes.

`compare` takes the candidate first and the golden report second. Exit status 0
means pass, 1 means a digest or timing failure, 2 means invalid command input, and
3 means the reports are incompatible. CI should also inspect the JSON `status` and
`reasons` fields rather than relying on the exit status alone. Hashing large audio
and weight files happens outside the timed region but can still take noticeable
wall-clock time.

### Compare scalar and speculative decoding

History n-gram speculative decoding is an experimental, output-preserving
implementation choice for the large float16 model on MPS. The benchmark command
already fixes the other verified settings: greedy decoding, batch size 1, CFG 1,
prelude forcing, and profiling disabled. Unsupported loaded models or devices are
rejected instead of silently falling back to scalar decoding.
The implementation bounds completed-chunk history and automatically returns to
scalar generation for the rest of a track when measured draft savings are too
small. This limits regressions but does not remove the need for representative
benchmarking.

First record the scalar golden with the same large model and arguments shown
below, but omit `--speculative-decoding` and write it to a different output path.
Then record the speculative candidate:

```console
uv run muscriptor-benchmark run /path/to/song.wav \
  --model /path/to/muscriptor-large/model.safetensors \
  --model-label large \
  --audio-id song-long-v1 \
  --device mps \
  --dtype float16 \
  --environment m4-pro-macos \
  --speculative-decoding \
  --output artifacts/m4-large-speculative.json
```

The implementation choice is deliberately excluded from the semantic workload
and protocol identities, so the scalar and speculative reports remain directly
comparable. It is still recorded in source metadata as
`"decoding_implementation": "scalar-v1"` or
`"decoding_implementation": "history-ngram-v1"`. The normal digest checks
therefore remain responsible for proving output equivalence before the timing
result is accepted. Speedup depends on how often prior token history can provide
an accepted draft; material with frequent misses can be slower. Measure it
separately for each workload rather than assuming a uniform speedup.

OS, Python, PyTorch, device, precision, and execution-policy metadata are part of
the environment identity. Regenerate the device-specific golden after changing
any of them, including a macOS update. On MPS, consider `--warmup-runs 2` if the
first warmup still includes shader compilation; the candidate and golden must use
the same value.

## CUDA runs

CUDA reports use their own baseline and are never compared with MPS reports. For
the current T4 policy, use an indexed device and the existing float32-weights plus
autocast path:

```console
uv run muscriptor-benchmark run /path/to/song.wav \
  --model /path/to/model.safetensors \
  --device cuda:0 \
  --dtype float32 \
  --environment colab-t4 \
  --output artifacts/t4-medium-baseline.json
```

Do not treat a timing comparison as a quality evaluation when intentionally
changing decoding, precision, batching, or model semantics. Those changes need a
separate note onset/offset and instrument-quality evaluation.

## Diagnose generation length and missing EOS

Per-chunk generation telemetry is opt-in because its observer callback runs inside
the timed region. Enable it together with the same options for both the golden and
candidate; an instrumented report is intentionally incompatible with both a plain
schema v2 report and a migrated schema v1 report. The protocol records the
instrumentation identity as `selected-output-v1` instead of `none`.

`libsndfile` cannot decode every M4A variant. For the large-model drums-stem
workload used during performance work, first decode the selected audio stream to
a native-rate float WAV. Keep channel mixing and 16 kHz resampling in muscriptor
so this follows the same canonicalization path as other benchmark inputs:

```console
ffmpeg -v error -i /path/to/drums.m4a \
  -map 0:a:0 -vn -sn -dn -c:a pcm_f32le \
  /path/to/song-drums-native-f32.wav
```

Then run the instrumented benchmark:

```console
uv run muscriptor-benchmark run /path/to/song-drums-native-f32.wav \
  --model /path/to/muscriptor-large/model.safetensors \
  --model-label large \
  --audio-id song-drums-native-f32-v1 \
  --device mps \
  --dtype float16 \
  --environment m4-pro-macos \
  --instruments drums,chromatic_percussion,orchestra_hit \
  --generation-telemetry \
  --allow-no-eos \
  --output artifacts/m4-large-drums-telemetry.json
```

Add `--speculative-decoding` to the same command to collect the equivalent
per-chunk telemetry while the history n-gram implementation is enabled. Keep all
other arguments identical when comparing it with the scalar report.

`--allow-no-eos` is a diagnostic opt-in. Without it, strict EOS checking remains
the default: a chunk that reaches the 2000-token limit aborts the command and no
partial report is published (an existing `--force` target is left unchanged).
With it, the warning is retained and the completed run records the affected chunk
with `hit_generation_limit=true`.

Each sample's `chunk_generation_stats` contains one entry per five-second chunk,
in order. `observed_rows` includes prompt echo rows returned by the model;
`generated_rows` excludes them. `eos_step` is the 1-based generated-row position
of the first EOS, or `null`. `prompt_tokens`, `max_gen_len`, `chunk_index`, and
`seek_time_us` make the count and audio position explicit. For batched decoding,
row counts include work performed after an earlier chunk reaches EOS while the
rest of its batch finishes. Beam-search telemetry describes the selected output
replayed by the model, not the internal beam exploration.

Telemetry uses the already canonicalized on-device waveform and the base
`TranscriptionModel` prepared-audio path so the observer can be attached without
re-decoding or transferring audio. The command refuses telemetry when
`transcribe`, `_transcribe_prepared`, or `_generate_token_stream` has been
overridden, because those implementations could change the meaning of the
recorded fields.
