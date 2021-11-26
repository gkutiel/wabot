import torch
import pytorch_lightning as pl

from typing import Dict, List

from torch import nn
from torch import Tensor
from wabot.data import Data

from wabot.parser import Tokenizer
from wabot.TextEncoder import TextEncoder


class SimpleSenderPredictor(pl.LightningModule):
    def __init__(
            self, *,
            text_encoder: TextEncoder,
            num_senders: int) -> None:

        super().__init__()

        self.text_encoder = text_encoder

        self.linear = nn.Linear(text_encoder.hidden_size, num_senders)
        self.softmax = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, text: str) -> Tensor:
        encoded = self.text_encoder(text)
        logits = self.linear(encoded)

        return self.softmax(logits).squeeze()

    def training_step(self, batch: List[Data], batch_idx):
        assert len(batch) == 1

        d = batch[0]
        pred = self(d.text)
        target = torch.tensor(d.sender_id)
        loss = self.loss(pred[None, ...], target[None, ...])
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)
