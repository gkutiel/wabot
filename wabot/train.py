from pytorch_lightning import Trainer, seed_everything

from wabot.data import sender_dataloader
from wabot.parser import Tokenizer, build_lexicon, get_messages, get_senders, get_tokens
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.TextEncoder import TextEncoder


def train_simple_predictor():
    seed_everything(42)

    msgs = get_messages()

    text_encoder = TextEncoder(lexicon=build_lexicon(get_tokens(msgs), size=10000))

    senders = get_senders(msgs)

    dl = sender_dataloader(
        msgs=msgs,
        senders=senders,
        text_encoder=text_encoder)

    trainer = Trainer(
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        max_epochs=10)

    model = SimpleSenderPredictor(
        text_encoder=text_encoder,
        num_senders=len(senders))

    trainer.fit(model, dl)


if __name__ == '__main__':
    train_simple_predictor()
