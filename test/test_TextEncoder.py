import re
from datetime import datetime

from wabot.parser import Msg, Tokenizer


def test_MsgEncoder():
    from wabot.TextEncoder import TextEncoder
    encoder = TextEncoder(
        lexicon={'a': 0, 'b': 1, 'c': 2},
        tokenizer=Tokenizer(sub=re.compile(r' ')))

    encoded = encoder('a b b c')

    assert encoded.shape == (16,)
