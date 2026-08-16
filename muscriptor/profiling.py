"""推論中の任意計測を共通化する小さなヘルパー。"""

import contextlib
import sys
import time
from collections.abc import Iterator

import torch

import muscriptor.accelerator


@contextlib.contextmanager
def timed(
    enabled: bool,
    label: str,
    *,
    device: torch.device | str | None = None,
    precision: int = 2,
) -> Iterator[None]:
    """有効時だけ同期を挟んで処理時間をstderrへ出す。"""
    if not enabled:
        yield
        return

    muscriptor.accelerator.synchronize(device)
    started_at = time.perf_counter()
    try:
        yield
    except BaseException:
        raise
    else:
        muscriptor.accelerator.synchronize(device)
        elapsed = time.perf_counter() - started_at
        print(f"[muscriptor] {label}: {elapsed:.{precision}f}s", file=sys.stderr)


def message(enabled: bool, *values: object) -> None:
    """有効時だけ診断情報をstderrへ出す。"""
    if enabled:
        print(*values, file=sys.stderr)
