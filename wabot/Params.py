from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    hidden_size: int = 16
    batch_size: int = 10_000
    lr: float = 0.1
    min_tokens: int = 5
