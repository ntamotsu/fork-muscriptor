# Reproducible inference benchmarks

`muscriptor-benchmark` measures the warm transcription path without changing the
normal `muscriptor` CLI. It writes a machine-readable JSON report containing the
timing samples, public event digests, workload hashes, and an environment
identity.

## What is measured

The `model-loaded-audio-predecoded-on-device` protocol prepares the model and a
mono, float32, 16 kHz audio tensor before timing. Each recorded run then measures
complete consumption of `TranscriptionModel.transcribe(...)`, including condition
construction, autoregressive decoding, and public event construction. Accelerator
work is synchronized immediately before the start clock and after the generator is
exhausted.

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
