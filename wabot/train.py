import optuna

from optuna.samplers import TPESampler
from dataclasses import dataclass
from pytorch_lightning import Trainer, seed_everything
from wabot.Params import Params

from wabot.data import sender_dataloader
from wabot.parser import build_lexicon, get_messages, get_senders, get_tokens
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.TextEncoder import TextEncoder

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train(params: Params):
    seed_everything(42)

    msgs = get_messages()

    text_encoder = TextEncoder(
        params=params,
        lexicon=build_lexicon(get_tokens(msgs)))

    senders = get_senders(msgs)

    dl = sender_dataloader(
        params=params,
        msgs=msgs,
        senders=senders,
        text_encoder=text_encoder)

    monitor = 'train/acc'

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode='max',
        save_weights_only=True)

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode='max',
        min_delta=1e-6,
        patience=5)

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        max_epochs=30)

    model = SimpleSenderPredictor(
        params=params,
        text_encoder=text_encoder,
        num_senders=len(senders))

    trainer.fit(model, dl)

    return checkpoint_callback.best_model_score.item()


def objective(trial):
    return train(params=Params(
        hidden_size=trial.suggest_int('hidden_size', 16, 32, step=2),
        # batch_size=trial.suggest_int('batch_size', 128, 1024, step=128),
        lr=trial.suggest_float('lr', 0.001, 0.1, step=0.001)))


def train_hp():
    sampler = TPESampler(seed=8979)
    study = optuna.create_study(sampler=sampler)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
