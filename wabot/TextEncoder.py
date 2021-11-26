import torch

from typing import Dict, List, Union
from torch import nn, Tensor

from wabot.parser import Tokenizer


class TextEncoder(nn.Module):
    def __init__(
            self, *,
            hidden_size: int = 16,
            tokenizer: Tokenizer,
            lexicon: Dict[str, int]):
        super().__init__()

        self.hidden_size = hidden_size

        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.embedding = nn.Embedding(
            num_embeddings=len(lexicon),
            embedding_dim=hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, text: str) -> Tensor:
        tokens = self.tokenizer(text)
        tokens = [self.lexicon[t] for t in tokens]
        embedding = self.embedding(torch.tensor(tokens))
        embedding = embedding[None, ...]
        _, h = self.gru(embedding)

        return h.squeeze()
