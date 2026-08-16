"""履歴だけを使うgreedy speculative decoding用のdraft提案。"""

from __future__ import annotations


class HistoryNgramDraft:
    """観測済みtoken列の最長suffix一致から固定長の続きを提案する。"""

    def __init__(
        self,
        *,
        block_size: int = 4,
        min_context: int = 2,
        max_context: int = 16,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if min_context < 1:
            raise ValueError("min_context must be positive")
        if max_context < min_context:
            raise ValueError("max_context must be at least min_context")
        self.block_size = block_size
        self.min_context = min_context
        self.max_context = max_context
        self._completed: list[tuple[int, ...]] = []
        self._current: list[int] | None = None

    def start_chunk(self, initial_token: int) -> None:
        if self._current is not None:
            raise RuntimeError("the previous chunk is still active")
        self._current = [int(initial_token)]

    def observe(self, token: int) -> None:
        if self._current is None:
            raise RuntimeError("no chunk is active")
        self._current.append(int(token))

    def finish_chunk(self) -> None:
        if self._current is None:
            raise RuntimeError("no chunk is active")
        self._completed.append(tuple(self._current))
        self._current = None

    def discard_chunk(self) -> None:
        self._current = None

    def propose(self) -> tuple[int, ...] | None:
        current = self._current
        if current is None:
            raise RuntimeError("no chunk is active")
        if len(current) < self.min_context:
            return None

        sources: list[tuple[int, ...] | list[int]] = [current]
        sources.extend(reversed(self._completed))
        largest_context = min(self.max_context, len(current))
        for context_size in range(
            largest_context,
            self.min_context - 1,
            -1,
        ):
            suffix = current[-context_size:]
            for source in sources:
                latest_start = len(source) - context_size - self.block_size
                for start in range(latest_start, -1, -1):
                    if list(source[start : start + context_size]) != suffix:
                        continue
                    draft_start = start + context_size
                    return tuple(source[draft_start : draft_start + self.block_size])
        return None
