import fasttext
import json

from pytorch_lightning import Trainer, seed_everything
from wabot.MsgEncoder import MsgEncoder
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.data import msg_data_loader
from wabot.parser import get_senders
from wabot.tools import get_messages


def train_simple_predictor():
    seed_everything(42)

    msgs = get_messages()
    senders = get_senders(msgs)
    model = fasttext.load_model('model.bin')
    msg_encoder = MsgEncoder(model)

    predictor = SimpleSenderPredictor(
        senders=senders,
        msg_encoder=msg_encoder)

    trainer = Trainer(
        log_every_n_steps=1,
        accumulate_grad_batches=512,
        max_epochs=3)

    trainer.fit(predictor, msg_data_loader(msgs))
