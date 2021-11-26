import pytest
import re

from pytest import fixture
from datetime import datetime
from wabot.parser import Msg, get_senders, Tokenizer


@fixture()
def tokenizer():
    return Tokenizer(sub=re.compile(' '))


def text(date='01/01/18', time='00:00', sender='Bob', text='Hello World'):
    return f'''{date}, {time} - {sender}: {text}'''


def msg(datetime=datetime(2018, 1, 1, 0, 0), sender='Bob', text='Hello World'):
    return Msg(datetime, sender, text)


def test_Tokenizer(tokenizer):
    assert tokenizer('Hello World') == ['Hello', 'World']


def test_build_lexicon():
    from wabot.parser import build_lexicon

    assert build_lexicon(['a', 'a', 'b', 'c', 'c'], size=2) == {'a': 1, 'c': 2}


def test_date_time_msg_re():
    from wabot.parser import datetime_msg_re

    assert not datetime_msg_re.match(text(date='Dec 12'))

    match = datetime_msg_re.match(text())
    assert match
    assert match[1] == '01/01/18, 00:00'
    assert match[2] == 'Bob'
    assert match[3] == 'Hello World'

    match = datetime_msg_re.match(text(date='1/1/18', time='0:00'))
    assert match
    assert match[1] == '1/1/18, 0:00'
    assert match[2] == 'Bob'
    assert match[3] == 'Hello World'


def test_parse_line():
    from wabot.parser import parse_msg

    assert parse_msg(text()) == msg()
    assert parse_msg(text(time='0:00')) == msg(datetime(2018, 1, 1, 0, 0))

    with pytest.raises(ValueError):
        parse_msg(text(date='Dec 12'))


def test_parse():
    from wabot.parser import parse

    lines = [text(time=f'00:0{i}') for i in range(3)]

    parsed = list(parse(lines))
    assert len(parsed) == 3
    assert parsed[0].time == datetime(2018, 1, 1, 0, 0)
    assert parsed[0].text == 'Hello World'
    assert parsed[1].time == datetime(2018, 1, 1, 0, 1)
    assert parsed[1].text == 'Hello World'
    assert parsed[2].time == datetime(2018, 1, 1, 0, 2)
    assert parsed[2].text == 'Hello World'


def test_get_senders():
    assert get_senders([msg(), msg(), msg()]) == {'Bob': 0}

    senders = get_senders([msg(sender='A'), msg(sender='C'), msg(sender='B')])
    assert senders.keys() == {'A', 'B', 'C'}
    assert set(senders.values()) == {0, 1, 2}


def test_get_tokens(tokenizer):
    from wabot.parser import get_tokens

    msgs = [msg(text='a b'), msg(text='a c d')]
    assert get_tokens(msgs, tokenizer=tokenizer) == {'a', 'b', 'c', 'd'}


def test_to_sessions():
    from wabot.parser import get_sessions

    def dt(m):
        return datetime(2018, 1, 1, 0, m)

    msgs = [msg(datetime=dt(i)) for i in range(5)]
    assert len(list(get_sessions(msgs))) == 1

    msgs = [msg(datetime=dt(i)) for i in range(1, 20, 6)] + [msg(datetime=dt(i)) for i in range(31, 50, 7)]
    sessions = list(get_sessions(msgs))
    assert len(sessions) == 2
    assert len(sessions[0]) == 4
    assert len(sessions[1]) == 3
