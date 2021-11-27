from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    hidden_size: int = 18
    batch_size: int = 5_000
    lr: float = 0.1
    min_tokens: int = 10
    max_tokens: int = 20
    lexicon_size: int = 10_000
