import re

from wabot.parser import Tokenizer
from wabot.train import Params


def test_MsgEncoder():
    from wabot.TextEncoder import TextEncoder
    encoder = TextEncoder(
        params=Params(hidden_size=16),
        lexicon={'a': 0, 'b': 1, 'c': 2},
        tokenizer=Tokenizer(sub=re.compile(r' ')))

    encoded = encoder('a b b c')

    assert encoded.shape == (16,)
