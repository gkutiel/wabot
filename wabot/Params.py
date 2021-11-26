from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    hidden_size: int = 16
    batch_size: int = 128
    lr: float = 0.1
    min_tokens: int = 5
