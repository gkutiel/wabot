import re

from datetime import datetime
from wabot.TextEncoder import TextEncoder
from wabot.parser import Tokenizer
from wabot.train import Params


def test_SimpleSenderPredictor():
    from wabot.SimpleSenderPredictor import SimpleSenderPredictor
    text_encoder = TextEncoder(
        params=Params(),
        tokenizer=Tokenizer(sub=re.compile(r' ')),
        lexicon={'a': 0, 'b': 1, 'c': 2})

    predictor = SimpleSenderPredictor(
        params=Params(),
        num_senders=3,
        text_encoder=text_encoder)

    pred = predictor('a b c')
    assert pred.shape == (3,)
    assert 0.99 <= pred.sum().item() <= 1.01
