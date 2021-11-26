from typing import List
import pytorch_lightning as pl
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

    def training_step(self, sample, batch_idx):
        sender, text = sample

        return self.loss(self(text), self.senders.index(sender))
