import fasttext
from datetime import datetime

from wabot.parser import Msg


def test_MsgEncoder():
    from wabot.MsgEncoder import MsgEncoder
    model = fasttext.load_model('model.bin')
    encoder = MsgEncoder(model)
    encoded = encoder('בוקר טוב')

    assert encoded.shape == (32,)
