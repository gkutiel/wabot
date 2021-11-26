import fasttext
import json

from pytorch_lightning import Trainer, seed_everything
from wabot.MsgEncoder import MsgEncoder
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.data import msg_data_loader
from wabot.parser import get_senders
from wabot.parser import get_messages


def train_simple_predictor():
    seed_everything(42)

    msgs = get_messages()
    msgs = [m for m in msgs if len(m.text) > 340]

    with open('train.json', 'w') as f:
        for m in msgs:
            json.dump(m.to_dict(), f)
            print(file=f)

    senders = get_senders(msgs)
    print(f'Senders: {senders}')

    model = fasttext.load_model('model.bin')
    msg_encoder = MsgEncoder(model, hidden_size=16)

    predictor = SimpleSenderPredictor(
        senders=senders,
        msg_encoder=msg_encoder)

    trainer = Trainer(
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        max_epochs=10)

    trainer.fit(predictor, msg_data_loader(msgs, shuffle=True))
