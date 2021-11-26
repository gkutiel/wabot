import fasttext

from datetime import datetime
from wabot.MsgEncoder import MsgEncoder
from wabot.parser import Msg


def test_SimpleSenderPredictor():
    from wabot.SimpleSenderPredictor import SimpleSenderPredictor
    model = fasttext.load_model('model.bin')
    msg_encoder = MsgEncoder(model, hidden_size=16)

    predictor = SimpleSenderPredictor(
        senders=['a', 'b', 'c'],
        msg_encoder=msg_encoder)

    pred = predictor('בוקר טוב')
    assert pred.shape == (3,)
    assert 0.99 <= pred.sum().item() <= 1.01
