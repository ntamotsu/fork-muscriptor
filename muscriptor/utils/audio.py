"""Audio loading and resampling utilities. WAV is handled by the stdlib;
other formats fall back to `soundfile`."""

import wave
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from threading import Lock
from typing import IO

import numpy as np
import torch

from muscriptor.utils.resample import (
    ResampleFrac as _ResampleFrac,
    resample_frac as resample_frac,
)


_ResamplerKey = tuple[int, int, torch.device, torch.dtype]

_RESAMPLER_CACHE_MAX_SIZE = 8
# 一般的な44.1 kHz→16 kHzのfloat32 kernel（約0.4 MiB）は十分保持できる。
_RESAMPLER_CACHE_MAX_ENTRY_BYTES = 64 * 1024 * 1024
_RESAMPLER_CACHE_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_RESAMPLER_CACHE: OrderedDict[_ResamplerKey, _ResampleFrac] = OrderedDict()
_RESAMPLER_INFLIGHT: dict[_ResamplerKey, Future[_ResampleFrac]] = {}
_RESAMPLER_CACHE_LOCK = Lock()


def _clear_resampler_cache() -> None:
    """cache済み・構築中のresampler参照を破棄する。"""
    with _RESAMPLER_CACHE_LOCK:
        _RESAMPLER_CACHE.clear()
        _RESAMPLER_INFLIGHT.clear()


def _resampler_buffer_bytes(resampler: _ResampleFrac) -> int:
    """resamplerが保持する全bufferの論理byte数を返す。"""
    return sum(buffer.numel() * buffer.element_size() for buffer in resampler.buffers())


def _cached_resampler_buffer_bytes() -> int:
    """cacheに常駐しているresampler bufferの合計byte数を返す。"""
    return sum(_resampler_buffer_bytes(item) for item in _RESAMPLER_CACHE.values())


def _cache_resampler(
    key: _ResamplerKey,
    resampler: _ResampleFrac,
    buffer_bytes: int,
) -> None:
    """上限内のresamplerをLRUへ追加する。呼出側でlockを保持する。"""
    if buffer_bytes > min(
        _RESAMPLER_CACHE_MAX_ENTRY_BYTES,
        _RESAMPLER_CACHE_MAX_TOTAL_BYTES,
    ):
        return
    while _RESAMPLER_CACHE and (
        len(_RESAMPLER_CACHE) >= _RESAMPLER_CACHE_MAX_SIZE
        or _cached_resampler_buffer_bytes() + buffer_bytes
        > _RESAMPLER_CACHE_MAX_TOTAL_BYTES
    ):
        _RESAMPLER_CACHE.popitem(last=False)
    _RESAMPLER_CACHE[key] = resampler


def _build_resampler(
    key: _ResamplerKey,
    future: Future[_ResampleFrac],
) -> _ResampleFrac:
    """global lock外でresamplerを構築し、結果を同一keyのwaiterへ共有する。"""
    orig_freq, new_freq, device, dtype = key
    try:
        resampler = _ResampleFrac(orig_freq, new_freq).to(
            device=device,
            dtype=dtype,
        )
        buffer_bytes = _resampler_buffer_bytes(resampler)
        with _RESAMPLER_CACHE_LOCK:
            if _RESAMPLER_INFLIGHT.get(key) is future:
                del _RESAMPLER_INFLIGHT[key]
                _cache_resampler(key, resampler, buffer_bytes)
    except BaseException as error:
        with _RESAMPLER_CACHE_LOCK:
            if _RESAMPLER_INFLIGHT.get(key) is future:
                del _RESAMPLER_INFLIGHT[key]
        future.set_exception(error)
        raise
    future.set_result(resampler)
    return resampler


def _get_resampler(key: _ResamplerKey) -> _ResampleFrac:
    """cacheまたはkey別in-flightからresamplerを取得する。"""
    with _RESAMPLER_CACHE_LOCK:
        resampler = _RESAMPLER_CACHE.get(key)
        if resampler is not None:
            _RESAMPLER_CACHE.move_to_end(key)
            return resampler

        future = _RESAMPLER_INFLIGHT.get(key)
        should_build = future is None
        if should_build:
            future = Future()
            _RESAMPLER_INFLIGHT[key] = future

    assert future is not None
    if should_build:
        return _build_resampler(key, future)
    return future.result()


def _read_wav_file(source) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV file using the stdlib `wave` module.

    `source` may be a filesystem path or a binary file-like object.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    if hasattr(source, "read"):
        opened = wave.open(source, "rb")
    else:
        opened = wave.open(str(source), "rb")
    with opened as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        as_int32 = np.where(as_int32 >= (1 << 23), as_int32 - (1 << 24), as_int32)
        data = as_int32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    data = data.reshape(-1, n_channels)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def _read_non_wav_file(source: str | Path | IO[bytes]) -> tuple[torch.Tensor, int]:
    """Load a non-WAV audio file using `soundfile`.

    `source` may be a filesystem path or a binary file-like object (e.g. an
    ``io.BytesIO`` of an uploaded file), since libsndfile reads either.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    target = str(source) if isinstance(source, (str, Path)) else source
    data, sample_rate = sf.read(target, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sample_rate


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    """最終次元を対象に、再利用可能なsinc kernelでリサンプリングする。"""
    if orig_freq == new_freq:
        return waveform
    orig_freq = int(orig_freq)
    new_freq = int(new_freq)
    key = (orig_freq, new_freq, waveform.device, waveform.dtype)
    resampler = _get_resampler(key)
    # forwardはcacheを更新しないため、lock外で並行実行できる。
    return resampler(waveform)


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return a mono float32 tensor at target_sr.

    PCM WAV files are read with the stdlib `wave` module. Other formats (mp3,
    flac, ogg, m4a, …) are decoded via `soundfile`. Dispatch is by content, not
    file extension, so misnamed files (e.g. an MP3 upload saved as .wav) still
    load.

    Returns:
        Tensor of shape [1, T] at target_sr.
    """
    filepath = Path(path)
    try:
        wav, sr = _read_wav_file(str(filepath))
    except (wave.Error, EOFError):
        wav, sr = _read_non_wav_file(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav
