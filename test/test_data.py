from datetime import datetime
import re
from wabot.TextEncoder import TextEncoder
from wabot.parser import Msg, Tokenizer
from wabot.train import Params


def test_sender_dataloader():
    from wabot.data import sender_dataloader

    senders = {'a': 0, 'b': 1, 'c': 2}
    msgs = [Msg(datetime(2020, 1, i+1), 'a', 'Hello My World') for i in range(10)]
    tokenizer = Tokenizer(sub=re.compile(r' '))

    params = Params(batch_size=2, min_tokens=2)
    dl = sender_dataloader(
        params=params,
        msgs=msgs,
        senders=senders,
        text_encoder=TextEncoder(
            params=params,
            tokenizer=tokenizer,
            lexicon={'Hello': 0, 'My': 1, 'World': 2}))

    assert len(dl) == 5

    d = next(iter(dl))
    assert len(d) == 2

    sid, tokens = d
    assert sid.shape == (2,), sid
    assert tokens.shape == (2, 3), tokens
