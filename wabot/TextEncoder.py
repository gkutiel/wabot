import torch

from typing import Dict, List, Union, cast
from torch import nn, Tensor
from wabot.Params import Params

from wabot.parser import Tokenizer


class TextEncoder(nn.Module):
    def __init__(
            self, *,
            params: Params,
            tokenizer: Tokenizer = Tokenizer(),
            lexicon: Dict[str, int]):

        super().__init__()

        hidden_size = params.hidden_size
        self.hidden_size = hidden_size

        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.embedding = nn.Embedding(
            num_embeddings=len(lexicon) + 1,
            embedding_dim=hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True)

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer(text)
        return [self.lexicon[t] for t in tokens if t in self.lexicon]

    def forward(self, input: Union[str, List[int], Tensor]) -> Tensor:
        if isinstance(input, str):
            tokens = torch.tensor(self.tokenize(input), dtype=torch.int)
        elif isinstance(input, list):
            tokens = torch.tensor(input, dtype=torch.int)
        else:
            tokens = input

        embedding = self.embedding(tokens)

        if embedding.dim() < 3:
            embedding = embedding[None, ...]

        assert embedding.dim() == 3

        _, h = self.gru(embedding)

        return h.squeeze()
