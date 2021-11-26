import json

from pytorch_lightning import Trainer, seed_everything
from wabot.MsgEncoder import TextEncoder
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.data import Data, msg_data_loader
from wabot.parser import build_lexicon, get_senders, Tokenizer, get_tokens
from wabot.parser import get_messages


def train_simple_predictor():
    seed_everything(42)

    msgs = get_messages()
    tokens = get_tokens(msgs)
    lexicon = build_lexicon(tokens, size=1000)
    senders = get_senders(msgs)
    # print(f'Senders: {senders}')

    # msg_encoder = MsgEncoder(model, hidden_size=16)

    # predictor = SimpleSenderPredictor(
    #     senders=senders,
    #     msg_encoder=msg_encoder)

    # trainer = Trainer(
    #     log_every_n_steps=1,
    #     accumulate_grad_batches=1,
    #     max_epochs=10)

    # trainer.fit(predictor, msg_data_loader(msgs, shuffle=True))


if __name__ == '__main__':
    train_simple_predictor()
