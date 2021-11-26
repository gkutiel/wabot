import torch
import pytorch_lightning as pl

from typing import List

from torch import nn
from torch import Tensor

from wabot.parser import Msg
from wabot.MsgEncoder import MsgEncoder


class SimpleSenderPredictor(pl.LightningModule):
    def __init__(self, senders: List[str], msg_encoder: MsgEncoder) -> None:
        super().__init__()
        self.senders = senders
        self.msg_encoder = msg_encoder

        self.ln = nn.Linear(msg_encoder.hidden_size, len(senders))
        self.sm = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, text: str) -> Tensor:
        encoded = self.msg_encoder(text)
        logits = self.ln(encoded)

        return self.sm(logits)

    def training_step(self, msgs: List[Msg], batch_idx):
        assert len(msgs) == 1
        msg = msgs[0]

        pred = self(msg.sender + ':' + msg.text)
        target = torch.tensor(self.senders.index(msg.sender))
        loss = self.loss(pred[None, ...], target[None, ...])
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)
