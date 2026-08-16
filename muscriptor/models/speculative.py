"""履歴だけを使うgreedy speculative decoding用のdraft提案。"""

from __future__ import annotations


class HistoryNgramDraft:
    """観測済みtoken列の最長suffix一致から固定長の続きを提案する。"""

    # K=4のblock検証は実測でscalar生成約1.26回分。8回で12 token未満しか
    # 確定できない区間は、そのchunkだけ通常生成へ戻す。
    _VERIFICATION_WINDOW = 8
    _MIN_COMMITTED_TOKENS = 12
    # 候補探索のmissも含め、4 chunk・256回以上観測してscalar呼出し削減が
    # 探索回数の12.5%未満なら、そのtrackの残りは通常生成へ戻す。
    _MIN_EVIDENCE_CHUNKS = 4
    _MIN_PROPOSAL_CALLS = 256

    def __init__(
        self,
        *,
        block_size: int = 4,
        min_context: int = 2,
        max_context: int = 16,
        max_completed_chunks: int = 8,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if min_context < 1:
            raise ValueError("min_context must be positive")
        if max_context < min_context:
            raise ValueError("max_context must be at least min_context")
        if max_completed_chunks < 1:
            raise ValueError("max_completed_chunks must be positive")
        self.block_size = block_size
        self.min_context = min_context
        self.max_context = max_context
        self.max_completed_chunks = max_completed_chunks
        self._completed: list[tuple[int, ...]] = []
        self._current: list[int] | None = None
        self._recent_commits: list[int] = []
        self._drafting_enabled = True
        self._track_disabled = False
        self._evidence_chunks = 0
        self._proposal_calls = 0
        self._verified_blocks = 0
        self._committed_tokens = 0
        self._chunk_proposal_calls = 0
        self._chunk_verified_blocks = 0
        self._chunk_committed_tokens = 0

    @property
    def enabled(self) -> bool:
        """このtrackでdraft探索を続けるかを返す。"""
        return not self._track_disabled

    def start_chunk(self, initial_token: int) -> None:
        if self._current is not None:
            raise RuntimeError("the previous chunk is still active")
        self._current = [int(initial_token)]
        self._recent_commits = []
        self._drafting_enabled = self.enabled
        self._chunk_proposal_calls = 0
        self._chunk_verified_blocks = 0
        self._chunk_committed_tokens = 0

    def observe(self, token: int) -> None:
        if self._current is None:
            raise RuntimeError("no chunk is active")
        self._current.append(int(token))

    def record_block(self, proposed: int, committed: int) -> None:
        """実際に検証したblockの確定token数を記録する。"""
        del proposed
        self._chunk_verified_blocks += 1
        self._chunk_committed_tokens += committed
        self._recent_commits.append(committed)
        if len(self._recent_commits) > self._VERIFICATION_WINDOW:
            del self._recent_commits[0]
        if (
            len(self._recent_commits) == self._VERIFICATION_WINDOW
            and sum(self._recent_commits) < self._MIN_COMMITTED_TOKENS
        ):
            self._drafting_enabled = False

    def finish_chunk(self) -> None:
        if self._current is None:
            raise RuntimeError("no chunk is active")
        if self._chunk_proposal_calls:
            self._evidence_chunks += 1
            self._proposal_calls += self._chunk_proposal_calls
            self._verified_blocks += self._chunk_verified_blocks
            self._committed_tokens += self._chunk_committed_tokens
        saved_scalar_calls = self._committed_tokens - self._verified_blocks
        if (
            self._evidence_chunks >= self._MIN_EVIDENCE_CHUNKS
            and self._proposal_calls >= self._MIN_PROPOSAL_CALLS
            and 8 * saved_scalar_calls < self._proposal_calls
        ):
            self._track_disabled = True
            self._completed.clear()
        if self.enabled and len(self._current) >= self.min_context + self.block_size:
            self._completed.append(tuple(self._current))
            overflow = len(self._completed) - self.max_completed_chunks
            if overflow > 0:
                del self._completed[:overflow]
        self._current = None

    def discard_chunk(self) -> None:
        self._current = None
        self._chunk_proposal_calls = 0
        self._chunk_verified_blocks = 0
        self._chunk_committed_tokens = 0

    def propose(self) -> tuple[int, ...] | None:
        current = self._current
        if current is None:
            raise RuntimeError("no chunk is active")
        if not self._drafting_enabled:
            return None
        if len(current) < self.min_context:
            return None
        self._chunk_proposal_calls += 1

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
