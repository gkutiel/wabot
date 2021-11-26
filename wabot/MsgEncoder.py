from fasttext import FastText
import torch
from typing import List, Union
from torch import nn, Tensor

from wabot.parser import Msg


class MsgEncoder(nn.Module):
    def __init__(self, model: FastText._FastText, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = model
        self.gru = nn.GRU(
            input_size=model.get_dimension(),
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, text: str, h0: Union[Tensor, None] = None) -> Tensor:
        words, _ = self.model.get_line(text)
        input = [self.model[w] for w in words]
        input = [torch.tensor(t, dtype=torch.float) for t in input]
        input = torch.stack(input)
        input = input[None, ...]

        if h0 is None:
            h0 = torch.zeros(1, 1, self.hidden_size)

        _, encoding = self.gru(input, h0)
        return encoding.squeeze()
