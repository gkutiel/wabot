import torch
import torchmetrics

import pytorch_lightning as pl

from typing import Dict, List, Tuple, Union

from torch import nn
from torch import Tensor
from wabot.Params import Params

from wabot.parser import Tokenizer
from wabot.TextEncoder import TextEncoder


class SimpleSenderPredictor(pl.LightningModule):
    def __init__(
            self, *,
            params: Params,
            text_encoder: TextEncoder,
            num_senders: int) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.lr = params.lr
        self.text_encoder = text_encoder

        self.linear = nn.Linear(text_encoder.hidden_size, num_senders)
        self.softmax = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens: Union[str, List[int], Tensor]) -> Tensor:
        encoded = self.text_encoder(tokens)
        logits = self.linear(encoded)

        return self.softmax(logits)

    def training_step(self, batch: List[Tensor], batch_idx):
        sid, tokens = batch
        pred = self(tokens)
        loss = self.loss(pred, sid)
        self.log('train/loss', loss)
        self.log('train/acc', torchmetrics.functional.accuracy(pred, sid))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr)

    def predict(self, text: Union[str, List[int]]):
        with torch.no_grad():
            self.eval()
            return self(text)
