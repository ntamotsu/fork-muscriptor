"""自己回帰生成の軽量な観測データ型。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkGenerationStats:
    """完了した1 chunkのselected-output生成統計。

    ``observed_rows`` はmodelのstep iteratorから実際に取得したrow数で、
    prompt echoを含む。``generated_rows`` はそこから``prompt_tokens``を
    除いたrow数、``eos_step`` は生成row内で最初にEOSを観測した1-originの
    位置である。batch生成では、先にEOSへ到達したchunkについても、同じ
    batchの完了まで取得したrowを両row数へ含めるため、EOS後に生じた実際の
    batch計算量を把握できる。

    beam searchでは内部beamの探索過程ではなく、modelが最後にreplayする
    selected outputだけを観測する。
    """

    chunk_index: int
    seek_time_us: int
    prompt_tokens: int
    observed_rows: int
    generated_rows: int
    eos_step: int | None
    max_gen_len: int
    hit_generation_limit: bool

    def __post_init__(self) -> None:
        nonnegative_integer_fields = (
            "chunk_index",
            "seek_time_us",
            "prompt_tokens",
            "observed_rows",
            "generated_rows",
        )
        for name in nonnegative_integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int (bool is not accepted)")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if type(self.max_gen_len) is not int:
            raise TypeError("max_gen_len must be an int (bool is not accepted)")
        if self.max_gen_len < 1:
            raise ValueError("max_gen_len must be positive")

        if self.eos_step is not None:
            if type(self.eos_step) is not int:
                raise TypeError(
                    "eos_step must be an int or None (bool is not accepted)"
                )
            if not 1 <= self.eos_step <= self.generated_rows:
                raise ValueError("eos_step must be within generated_rows")

        if type(self.hit_generation_limit) is not bool:
            raise TypeError("hit_generation_limit must be a bool")
        if self.observed_rows != self.prompt_tokens + self.generated_rows:
            raise ValueError("observed_rows must equal prompt_tokens + generated_rows")

        expected_hit = self.eos_step is None and self.observed_rows >= self.max_gen_len
        if self.hit_generation_limit is not expected_hit:
            raise ValueError("hit_generation_limit is inconsistent with EOS and rows")
